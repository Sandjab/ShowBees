import librosa
import numpy as np
from scipy import signal


class MFCC:
    def __init__(self, n_mfcc):
        self._n_mfcc = n_mfcc

    def __call__(self, sig, sr):
        melspec = librosa.feature.melspectrogram(sig, sr)
        return librosa.feature.mfcc(
            S=librosa.power_to_db(melspec),
            sr=sr,
            n_mfcc=self._n_mfcc
        )


class STFT:
    def __init__(self, n_fft=2048, hop_length=512, win_length=2048):
        self._n_fft = n_fft
        self._hop_length = hop_length
        self._win_length = win_length
        return

    def __call__(self, sig, sr):
        return np.abs(librosa.stft(sig,
                                   n_fft=self._n_fft,
                                   hop_length=self._hop_length,
                                   win_length=self._win_length,
                                   window='hann',
                                   center=True,
                                   dtype=np.complex64,
                                   pad_mode='reflect'
                                   )
                      )


class Welch:
    def __init__(self, nperseg):
        self._nperseg = nperseg

    def __call__(self, sig, sr):
        f, Px = signal.welch(sig, sr, nperseg=self._nperseg)


class Magic:
    def __init__(self, freq_min, freq_max, freq_step):
        self._freq_step = freq_step
        self._freq_min = freq_min
        self._freq_max = freq_max

    def __call__(self, sig, sr):
        f, Px = signal.welch(sig, sr, nperseg=sr / self._freq_step)
        start = int(self._freq_min / self._freq_step)
        end = int(self._freq_max / self._freq_step)
        return Px[start:end]


class FrequencySieve:
    def __init__(self, freq_step, freqs):
        self._freq_step = freq_step
        self._freqs = freqs

    def __call__(self, sig, sr):
        f, Px = signal.welch(sig, sr, nperseg=sr / self._freq_step)
        results = []
        for freq in self._freqs:
            results.append(Px[int(freq / self._freq_step)])

        return results
