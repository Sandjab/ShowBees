def mfcc(sig, sr, n_mfcc=20):
    melspec = librosa.feature.melspectrogram(sig, sr)
    return librosa.feature.mfcc(
        S=librosa.power_to_db(melspec),
        sr=sr,
        n_mfcc=n_mfcc)


def welch(sig, sr, nperseg=2048):
    _, psd = signal.welch(sig, sr, nperseg=nperseg)
    return psd[0:31]


def extractor1(*args, **kwargs):
    return kwargs.get('P1', None)


def extractor2(x, P1, P2, P3='XX'):
    return P2


def extract_feature_from_sample(sample_path, extractor, *args, **kwargs):
    """Extract features from audio chunk

    Extract features from a sample audio file (chunk) using the provided
    extractor and parameters

    Args:
        sample_path (Path): Path to the audio chunk
        extractor (func): extractor function to be executed
        *kwargs : arbitrary keyword type arguments for extractor

    Returns:
        (object):

    """

    sig, sr = librosa.core.load(sample_path)

    return extractor(sig, sr, *args, **kwargs)


def extract_features_from_dataset(dataset_name, sample_names, output_path):
    # TODO : reference centralisée pour les paths
    # dataset_path = mooltipath('datasets', dataset_name)
    # samples_path = Path(dataset_path, 'samples')
    return
