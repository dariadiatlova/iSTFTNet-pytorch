import os
import glob
import torch
import argparse
import onnxruntime
import numpy as np

from tqdm import tqdm
from pathlib import Path
from loguru import logger
from scipy.io.wavfile import write

from src.util.env import AttrDict
from src.models.stft import TorchSTFT
from src.models.models import Generator
from src.util.utils import setup_logger, load_checkpoint, load_config
from src.datasets.meldataset import get_mel_spectrogram, MAX_WAV_VALUE, load_wav


def inference(args: argparse.Namespace, config: AttrDict, device: str) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    stft = TorchSTFT(**config).to(device)

    if not args.onnx_inference:
        generator = Generator(config).to(device)
        state_dict_g = load_checkpoint(args.checkpoint_file, device)
        generator.load_state_dict(state_dict_g["generator"])
        generator.eval()
        generator.remove_weight_norm()
    else:
        ort_session = onnxruntime.InferenceSession(args.checkpoint_file, providers=[args.onnx_provider])

    if args.compute_mels:
        filelist = glob.glob(f"{args.input_wavs_dir}/*.wav")
    else:
        filelist = glob.glob(f"{args.input_mels_dir}/*.npy")
    with torch.no_grad():
        for filename in tqdm(filelist):
            if args.compute_mels:
                wav, sr = load_wav(filename)
                wav = wav / MAX_WAV_VALUE
                wav = torch.FloatTensor(wav).to(device)
                x = get_mel_spectrogram(wav.unsqueeze(0), **config)
            else:
                x = torch.FloatTensor(np.load(filename)).to(device)
            if not args.onnx_inference:
                spec, phase = generator(x)
            else:
                spec, phase = ort_session.run(None, {"input": x.detach().cpu().numpy().astype(np.float32)})
                spec = torch.FloatTensor(spec).to(device)
                phase = torch.FloatTensor(phase).to(device)
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
    parser.add_argument('--checkpoint_file', default="src/data/awesome_checkpoints/g_00975000")
    parser.add_argument('--onnx_inference', default=False, help="if True checkpoint file should be .onnx")
    parser.add_argument('--onnx_provider', default="CPUExecutionProvider",
                        help="https://onnxruntime.ai/docs/execution-providers/")
    parser.add_argument('--input_wavs_dir', default='src/data/deep_voices_wavs')
    parser.add_argument('--input_mels_dir', default='src/data/deep_voices_mels')
    parser.add_argument('--output_dir', default='src/data/generated_files')
    parser.add_argument('--config_path', default="src/config.json")
    parser.add_argument('--compute_mels', action=argparse.BooleanOptionalAction,
                        help="Pass --no-compute_mels if --input_mels_dir is specified and mels are precomputed.")
    args = parser.parse_args()

    if not args.input_wavs_dir and not args.input_mels_dir:
        logger.error("Mels directory or wav directory to get mels is required.")

    config = load_config(args.config_path)

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    inference(args, config, device)


if __name__ == '__main__':
    main()
