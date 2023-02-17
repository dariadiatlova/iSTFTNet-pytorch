import os
import wandb
import time
import torch
import warnings
import argparse
import itertools
import numpy as np
import torch.nn.functional as F
import torch.multiprocessing as mp

from loguru import logger
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler, DataLoader

from src.models.stft import TorchSTFT
from src.util.env import AttrDict, build_env
from src.datasets.meldataset import MelDataset, get_mel_spectrogram, get_dataset_filelist
from src.util.utils import scan_checkpoint, load_checkpoint, save_checkpoint, setup_logger, load_config
from src.models.models import Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator, feature_loss, generator_loss,\
    discriminator_loss


warnings.simplefilter(action='ignore', category=FutureWarning)

torch.backends.cudnn.benchmark = True


def train(rank, args: argparse.Namespace, config: AttrDict):
    if config.num_gpus > 1:
        os.environ["MASTER_ADDR"] = config.dist_config["dist_addr"]
        os.environ["MASTER_PORT"] = config.dist_config["dist_port"]
        init_process_group(config.dist_config['dist_backend'], rank=rank,
                           world_size=config.dist_config['world_size'] * config.num_gpus)

    torch.cuda.manual_seed(config.seed)
    device = torch.device(f"cuda:{rank}")
    generator = Generator(config).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)
    stft = TorchSTFT(**config).to(device)
    optim_g = torch.optim.AdamW(generator.parameters(), config.learning_rate, betas=(config.adam_b1, config.adam_b2))
    optim_d = torch.optim.AdamW(itertools.chain(msd.parameters(), mpd.parameters()),
                                config.learning_rate, betas=(config.adam_b1, config.adam_b2))

    if rank == 0:
        logger.info(generator)
        os.makedirs(args.checkpoint_path, exist_ok=True)
        logger.info(f"checkpoints directory : {args.checkpoint_path}")

    if os.path.isdir(args.checkpoint_path):
        checkpoint_generator = scan_checkpoint(args.checkpoint_path, 'g_')
        checkpoint_discriminator = scan_checkpoint(args.checkpoint_path, 'do_')
    else:
        checkpoint_generator, checkpoint_discriminator = None, None

    steps = 0
    if checkpoint_generator is None or checkpoint_discriminator is None:
        state_dict_do = None
        last_epoch = -1
    else:
        state_dict_g = load_checkpoint(checkpoint_generator, device)
        state_dict_do = load_checkpoint(checkpoint_discriminator, device)
        generator.load_state_dict(state_dict_g['generator'])
        mpd.load_state_dict(state_dict_do['mpd'])
        msd.load_state_dict(state_dict_do['msd'])
        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])

    if config.num_gpus > 1:
        generator = DistributedDataParallel(generator, device_ids=[rank]).to(device)
        mpd = DistributedDataParallel(mpd, device_ids=[rank]).to(device)
        msd = DistributedDataParallel(msd, device_ids=[rank]).to(device)

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=config.lr_decay, last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=config.lr_decay, last_epoch=last_epoch)

    training_filelist, validation_filelist = get_dataset_filelist(args)
    trainset = MelDataset(training_filelist, shuffle=False if config.num_gpus > 1 else True, device=device, **config)
    train_sampler = DistributedSampler(trainset) if config.num_gpus > 1 else None
    train_loader = DataLoader(trainset, num_workers=config.num_workers, shuffle=False,
                              sampler=train_sampler, batch_size=config.batch_size, pin_memory=True, drop_last=True)

    if rank == 0:
        validset = MelDataset(validation_filelist, split=False, shuffle=False, **config)
        validation_loader = DataLoader(validset, num_workers=1, shuffle=False, sampler=None,
                                       batch_size=1, pin_memory=True, drop_last=True)
        wandb.init(project=config["wandb"]["project"])

    generator.train()
    mpd.train()
    msd.train()
    for epoch in range(max(0, last_epoch), args.training_epochs):
        if rank == 0:
            start = time.time()
            logger.info(f"Epoch: {epoch + 1}")

        if config.num_gpus > 1:
            train_sampler.set_epoch(epoch)

        for i, batch in enumerate(train_loader):
            if rank == 0:
                start_b = time.time()

            x, y, _, y_mel = batch
            x = x.to(device)
            y = y.to(device)
            y_mel = y_mel.to(device)
            y = y.unsqueeze(1)
            spec, phase = generator(x)
            y_g_hat = stft.inverse(spec, phase)
            y_g_hat_mel = get_mel_spectrogram(y_g_hat.squeeze(1), **config)
            optim_d.zero_grad()

            # MPD
            y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
            loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)

            # MSD
            y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
            loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

            loss_disc_all = loss_disc_s + loss_disc_f
            loss_disc_all.backward()
            optim_d.step()

            # Generator
            optim_g.zero_grad()

            # L1 Mel-Spectrogram Loss
            loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45

            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
            loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
            loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel

            loss_gen_all.backward()
            optim_g.step()

            if rank == 0:
                # Wandb & stdout logging
                if steps % args.wandb_log_interval == 0:
                    with torch.no_grad():
                        mel_error = F.l1_loss(y_mel, y_g_hat_mel).item()
                    wandb.log({"train_loss/gen_total_loss": loss_gen_all})
                    wandb.log({"train_loss/mel_error": mel_error})

                    logger.info(f'Steps : {steps}, Gen Loss Total : {np.round(loss_gen_all.item(), 3)}, '
                                f'Mel-Spec. Error : {np.round(mel_error, 3)}, s/b : {time.time() - start_b}')

                # checkpointing
                if steps % args.checkpoint_interval == 0 and steps != 0:
                    checkpoint_path = f"{args.checkpoint_path}/g_{steps:08d}"
                    save_checkpoint(checkpoint_path,
                                    {'generator': (generator.module if config.num_gpus > 1 else generator).state_dict()})
                    checkpoint_path = f"{args.checkpoint_path}/do_{steps:08d}"
                    save_checkpoint(checkpoint_path, 
                                    {'mpd': (mpd.module if config.num_gpus > 1 else mpd).state_dict(),
                                     'msd': (msd.module if config.num_gpus > 1 else msd).state_dict(),
                                     'optim_g': optim_g.state_dict(),
                                     'optim_d': optim_d.state_dict(),
                                     'steps': steps,
                                     'epoch': epoch})

                # Validation
                if steps % args.validation_interval == 0:
                    generator.eval()
                    torch.cuda.empty_cache()
                    val_err_tot = 0
                    with torch.no_grad():
                        for j, batch in enumerate(validation_loader):
                            x, y, _, y_mel = batch
                            spec, phase = generator(x.to(device))
                            y_g_hat = stft.inverse(spec, phase)
                            y_mel = y_mel.to(device)
                            y_g_hat_mel = get_mel_spectrogram(y_g_hat.squeeze(1), **config)
                            val_err_tot += F.l1_loss(y_mel, y_g_hat_mel).item()

                            if steps == 0:
                                wandb.log(
                                    {f"audio/{j}_gt": wandb.Audio(y[0].detach().cpu().numpy(),
                                                                  caption=f"audio/{j}_gt",
                                                                  sample_rate=config.sampling_rate)})
                            if steps % args.log_audio_interval == 0:
                                wandb.log(
                                    {f"audio/{j}_generated": wandb.Audio(y_g_hat[0].squeeze(0).detach().cpu().numpy(),
                                                                         caption=f"audio/{j}_generated",
                                                                         sample_rate=config.sampling_rate)})

                        val_err = val_err_tot / (j + 1)
                        wandb.log({"val_loss/mel_error": val_err})
                    generator.train()
            steps += 1

        scheduler_g.step()
        scheduler_d.step()
        
        if rank == 0:
            logger.info(f'Time taken for epoch {epoch + 1} is {int(time.time() - start)} sec\n')


def main():
    setup_logger()
    logger.info("Initializing Training Process...")

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_training_file', type=str)
    parser.add_argument('--input_validation_file', type=str)
    parser.add_argument('--config_path', default='config.json')
    parser.add_argument('--input_mels_dir', default='ft_dataset')
    parser.add_argument('--fine_tuning', default=False, type=bool)
    parser.add_argument('--checkpoint_path', default='/src/checkpoints')
    parser.add_argument('--training_epochs', default=3900, type=int)
    parser.add_argument('--wandb_log_interval', default=50, type=int, help="Once per n steps")
    parser.add_argument('--checkpoint_interval', default=5000, type=int, help="Once per n steps")
    parser.add_argument('--log_audio_interval', default=1000, type=int, help="Once per n steps")
    parser.add_argument('--validation_interval', default=50, type=int, help="Once per n steps")

    args = parser.parse_args()
    config = load_config(args.config_path)
    build_env(args.config_path, 'config.json', args.checkpoint_path)

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
        config.num_gpus = torch.cuda.device_count()
        config.batch_size = config.batch_size // config.num_gpus
        logger.info(f"Batch size per GPU : {config.batch_size}")

    if config.num_gpus > 1:
        mp.spawn(train, nprocs=config.num_gpus, args=(args, config,))
    else:
        train(0, args, config)


if __name__ == '__main__':
    main()
