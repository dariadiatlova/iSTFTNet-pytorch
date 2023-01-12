import random

import librosa
import torch
import torch.nn as nn
import torch.utils.data
import torchaudio

MIN_MEL_VALUE = 1e-05


def load_wav(wav_path, sr):
    wav, sample_rate = librosa.load(wav_path, sr=sr)
    return wav, sample_rate


class ComputeMel(nn.Module):
    def __init__(self, sample_rate, n_fft, hop_length, n_mels, power=1, center=False, f_min=0, f_max=None):
        super().__init__()
        self.compute_mel = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft,
                                                                hop_length=hop_length, n_mels=n_mels,
                                                                power=power, center=center,
                                                                f_min=f_min, f_max=f_max,
                                                                norm='slaney', mel_scale='slaney')
        self.hop_length = hop_length
        self.padding = n_fft - hop_length

    def forward(self, wav):
        padded_wav = nn.functional.pad(wav, (self.padding // 2, self.padding // 2), mode='reflect')
        self.compute_mel = self.compute_mel.to(wav.device)
        spectrogram = self.compute_mel.spectrogram(padded_wav)
        mel = torch.log(torch.clamp(self.compute_mel.mel_scale(spectrogram), min=MIN_MEL_VALUE))
        return mel


class MelDataset(torch.utils.data.Dataset):
    def __init__(self, training_files, segment_size, n_fft, num_mels, hop_size, win_size, sampling_rate,  fmin, fmax,
                 split=True, shuffle=True, device=None, fmax_loss=None, base_mels_path=None):
        self.audio_files = training_files
        random.seed(1234)
        if shuffle:
            random.shuffle(self.audio_files)
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.fmax_loss = fmax_loss
        self.device = device
        self.base_mels_path = base_mels_path
        self.mel_spectrogram = ComputeMel(sample_rate=self.sampling_rate, n_fft=self.n_fft,
                                          hop_length=self.hop_size, n_mels=self.num_mels, f_max=self.fmax)

    def __getitem__(self, index):
        filename = self.audio_files[index]
        audio, sampling_rate = load_wav(filename, self.sampling_rate)
        audio = torch.clip(torch.FloatTensor(audio).unsqueeze(0), -1, 1)
        if self.split:
            if audio.size(1) >= self.segment_size:
                max_audio_start = audio.size(1) - self.segment_size
                audio_start = random.randint(0, max_audio_start)
                audio = audio[:, audio_start:audio_start+self.segment_size]
            else:
                audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')

        mel = self.mel_spectrogram(audio)
        mel_loss = self.mel_spectrogram(audio)

        return mel.squeeze(), audio.squeeze(0), filename, mel_loss.squeeze()

    def __len__(self):
        return len(self.audio_files)
