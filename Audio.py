import numpy as np
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

import librosa
import matplotlib.pyplot as plt
from librosa import display
from torchaudio.utils import download_asset


def display_wave_plot(waveshow):
    fig, ax = plt.subplots()
    img = display.waveshow(waveshow, x_axis='time', ax=ax, color="blue")
    ax.set_title('Waveform')
    plt.show()


def display_note_plot(spec, mode='cqt'):
    fig, ax = plt.subplots()
    if mode == 'cqt':
        img = librosa.display.specshow(spec, x_axis='time', y_axis='cqt_note', hop_length=2048, ax=ax, fmin=27.5)
        ax.set_title('Power spectrogram')
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
    if mode == 'stft':
        img = librosa.display.specshow(spec, x_axis='time', y_axis='log', ax=ax)
        ax.set_title('Power spectrogram')
        fig.colorbar(img, ax=ax, format="%+2.0f dB")

    plt.show()


def plot_waveform(waveform, sr, title="Waveform", ax=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sr

    if ax is None:
        _, ax = plt.subplots(num_channels, 1)
    ax.plot(time_axis, waveform[0], linewidth=1)
    ax.grid(True)
    ax.set_xlim([0, time_axis[-1]])
    ax.set_title(title)


def plot_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto", interpolation="nearest")


def plot_fbank(fbank, title=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Filter bank")
    axs.imshow(fbank, aspect="auto")
    axs.set_ylabel("frequency bin")
    axs.set_xlabel("mel bin")


if __name__ == "__main__":
    w = "resources/wav/c_eb_c_eb_c_eb_c.wav"

    y, sr = librosa.load(w, duration=30, sr=22050)
    print(y.shape)
    cqt = librosa.cqt(y, sr=sr, n_bins=88, fmin=27.5, hop_length=2048)
    print(cqt.shape)
    spec = librosa.amplitude_to_db(np.abs(cqt), ref=np.min)
    print(spec)
    print(spec.shape)
    display_note_plot(spec)

    exit()
    torch.random.manual_seed(0)

    waveform, sample_rate = torchaudio.load(w)

    print(sample_rate)
    spectocgram = T.Spectrogram(n_fft=512, hop_length=512, win_length=512)
    spec = spectocgram(waveform)
    fig, axs = plt.subplots(1, 2)
    plot_waveform(waveform, sample_rate, title="Waveform", ax=axs[0])
    plot_spectrogram(spec[0], title="Spectogram", ax=axs[1])
    fig.tight_layout()
    plt.show()
