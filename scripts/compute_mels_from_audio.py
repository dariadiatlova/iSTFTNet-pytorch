import os
import glob
import torch
import argparse
import numpy as np

from tqdm import tqdm
from pathlib import Path

from src.util.env import AttrDict
from src.models.stft import TorchSTFT
from src.util.utils import load_config
from src.datasets.meldataset import load_wav, MAX_WAV_VALUE, get_mel_spectrogram


def main(args: argparse.Namespace, config: AttrDict):
    os.makedirs(args.output_mel_dirs, exist_ok=True)
    stft = TorchSTFT(**config)
    filelist = glob.glob(f"{args.input_wav_dirs}/*.wav")
    for filename in tqdm(filelist, total=len(filelist)):
        wav, sr = load_wav(filename)
        wav = wav / MAX_WAV_VALUE
        wav = torch.FloatTensor(wav)
        x = get_mel_spectrogram(wav.unsqueeze(0), **config).detach().cpu().numpy()
        mel_filename = Path(args.output_mel_dirs) / f"{Path(filename).stem}.npy"
        np.save(str(mel_filename), x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_wav_dirs', default="src/data/deep_voices")
    parser.add_argument('--output_mel_dirs', default="src/data/deep_mels")
    parser.add_argument('--config_path', default="src/config.json")
    args = parser.parse_args()
    config = load_config(args.config_path)
    main(args, config)
