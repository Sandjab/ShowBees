import librosa
# from scipy import signal


class MFCC:
    def __init__(self, n_mfcc):
        self._n_mfcc = n_mfcc

    def __call__(self, sig, sr):
        melspec = librosa.feature.melspectrogram(sig, sr)
        return librosa.feature.mfcc(
            S=librosa.power_to_db(melspec),
            sr=sr,
            n_mfcc=self._n_mfcc)
