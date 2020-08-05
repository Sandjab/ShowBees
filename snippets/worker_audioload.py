import librosa
from os import fspath

def worker(path):
    sig1, sr = librosa.core.load(fspath(path))
    sig2, sr = librosa.core.load(fspath(path))
    sig3, sr = librosa.core.load(fspath(path))
    sig4, sr = librosa.core.load(fspath(path))    
    return len(sig1) + len(sig2) + len(sig3) + len(sig4)