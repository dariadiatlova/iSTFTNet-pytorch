import os
import math
import torch
import random
import numpy as np

from typing import Tuple, List
from argparse import Namespace
from scipy.io.wavfile import read
from librosa.util import normalize
from torch.utils.data import Dataset
from librosa.filters import mel as librosa_mel_fn

MAX_WAV_VALUE = 32768.0


def load_wav(full_path):
    sampling_rate, data = read(full_path)
    return data, sampling_rate


def dynamic_range_compression(x, clip_val: float = 1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None))


def dynamic_range_decompression(x):
    return np.exp(x)


def dynamic_range_compression_torch(x, clip_val: float = 1e-5):
    return torch.log(torch.clamp(x, min=clip_val))


def dynamic_range_decompression_torch(x):
    return torch.exp(x)


def spectral_normalize_torch(magnitudes: torch.Tensor):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes: torch.Tensor):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


def get_mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax,
                        center=False, **_) -> torch.Tensor:
    mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
    mel_basis = torch.from_numpy(mel).float().to(y.device)
    hann_window = torch.hann_window(win_size).to(y.device)
    pad_value = int((n_fft - hop_size) / 2)
    y = torch.nn.functional.pad(y.unsqueeze(1), (pad_value, pad_value), mode='reflect').squeeze(1)
    spectrogram = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window,
                             center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)
    spectrogram = torch.sqrt(spectrogram.pow(2).sum(-1) + 1e-9)
    mel_spectrogram = torch.matmul(mel_basis, spectrogram)
    normalized_mel_spectrogram = spectral_normalize_torch(mel_spectrogram)
    return normalized_mel_spectrogram


def get_dataset_filelist(args: Namespace) -> Tuple[List, List]:
    with open(args.input_training_file, 'r', encoding='utf-8') as f:
        training_files = [i[:-1] for i in f.readlines()]
    with open(args.input_validation_file, 'r', encoding='utf-8') as f:
        validation_files = [i[:-1] for i in f.readlines()]
    return training_files, validation_files


class MelDataset(Dataset):
    def __init__(self, training_files, segment_size, n_fft, num_mels, hop_size, win_size, sampling_rate, fmin, fmax,
                 seed, split=True, shuffle=True, device=None, fmax_loss=None, fine_tuning=False, input_mels_dir=None,
                 **_):
        random.seed(seed)
        self.fmin = fmin
        self.fmax = fmax
        self.split = split
        self.n_fft = n_fft
        self.device = device
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmax_loss = fmax_loss
        self.fine_tuning = fine_tuning
        self.audio_files = training_files
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.base_mels_path = input_mels_dir
        if shuffle:
            random.shuffle(self.audio_files)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str, torch.Tensor]:
        filename = self.audio_files[index]
        audio, sampling_rate = load_wav(filename)
        audio = audio / MAX_WAV_VALUE
        if not self.fine_tuning:
            audio = normalize(audio) * 0.95
        if sampling_rate != self.sampling_rate:
            raise ValueError(f"{sampling_rate} SR doesn't match target {self.sampling_rate} SR")
        audio = torch.FloatTensor(audio).unsqueeze(0)

        if not self.fine_tuning:
            if self.split:
                if audio.size(1) >= self.segment_size:
                    max_audio_start = audio.size(1) - self.segment_size
                    audio_start = random.randint(0, max_audio_start)
                    audio = audio[:, audio_start:audio_start+self.segment_size]
                else:
                    audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')

            mel = get_mel_spectrogram(audio, self.n_fft, self.num_mels, self.sampling_rate, self.hop_size,
                                      self.win_size, self.fmin, self.fmax, center=False)
        else:
            mel = np.load(os.path.join(self.base_mels_path, os.path.splitext(os.path.split(filename)[-1])[0] + '.npy'))
            mel = torch.from_numpy(mel)

            if len(mel.shape) < 3:
                mel = mel.unsqueeze(0)

            if self.split:
                frames_per_seg = math.ceil(self.segment_size / self.hop_size)

                if audio.size(1) >= self.segment_size:
                    mel_start = random.randint(0, mel.size(2) - frames_per_seg - 1)
                    mel = mel[:, :, mel_start:mel_start + frames_per_seg]
                    audio = audio[:, mel_start * self.hop_size:(mel_start + frames_per_seg) * self.hop_size]
                else:
                    mel = torch.nn.functional.pad(mel, (0, frames_per_seg - mel.size(2)), 'constant')
                    audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')

        mel_loss = get_mel_spectrogram(audio, self.n_fft, self.num_mels, self.sampling_rate, self.hop_size,
                                       self.win_size, self.fmin, self.fmax_loss, center=False)

        return mel.squeeze(), audio.squeeze(0), filename, mel_loss.squeeze()

    def __len__(self) -> int:
        return len(self.audio_files)
