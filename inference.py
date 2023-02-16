import os
import glob
import json
import torch
import argparse

from pathlib import Path
from loguru import logger
from scipy.io.wavfile import write

from env import AttrDict
from stft import TorchSTFT
from models import Generator
from utils import setup_logger, load_checkpoint, scan_checkpoint
from meldataset import get_mel_spectrogram, MAX_WAV_VALUE, load_wav


def inference(args: argparse.Namespace, config: AttrDict, device: str) -> None:
    generator = Generator(config).to(device)
    stft = TorchSTFT(**config).to(device)
    state_dict_g = load_checkpoint(args.checkpoint_file, device)
    generator.load_state_dict(state_dict_g['generator'])
    filelist = glob.glob(f"{args.input_wavs_dir}/*.wav")
    os.makedirs(args.output_dir, exist_ok=True)
    generator.eval()
    generator.remove_weight_norm()
    with torch.no_grad():
        for i, filename in enumerate(filelist):
            wav, sr = load_wav(filename)
            wav = wav / MAX_WAV_VALUE
            wav = torch.FloatTensor(wav).to(device)
            x = get_mel_spectrogram(wav.unsqueeze(0), **config)
            spec, phase = generator(x)
            y_g_hat = stft.inverse(spec, phase)
            audio = y_g_hat.squeeze()
            audio = audio * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype('int16')
            output_file = Path(args.output_dir) / f"{Path(filename).stem}_generated.wav"
            write(output_file, config.sampling_rate, audio)
            logger.info(output_file)


def main():
    setup_logger()
    logger.info('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_file', required=True)
    parser.add_argument('--input_wavs_dir', default='test_files')
    parser.add_argument('--output_dir', default='generated_files')
    args = parser.parse_args()

    config_file = os.path.join(os.path.split(args.checkpoint_file)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()

    json_config = json.loads(data)
    config = AttrDict(json_config)

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    inference(args, config, device)


if __name__ == '__main__':
    main()
