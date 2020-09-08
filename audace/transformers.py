import re
from sklearn import preprocessing
from scipy.signal import butter, sosfilt
import numpy as np


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfilt(sos, data)
    return y


class BandPass:
    def __init__(self, lowcut, highcut, order=5):
        self._lowcut = lowcut
        self._highcut = highcut
        self._order = order
        return

    def __call__(self, s, sr):
        return butter_bandpass_filter(s,
                                      self._lowcut,
                                      self._highcut,
                                      sr,
                                      self._order)


class Identity:
    def __init__(self):
        return

    def __call__(self, s):
        return s


class Standardize:
    def __init__(self):
        return

    def __call__(self, s, *args, **kwargs):
        return preprocessing.scale(s)


class NormalizeRMS:
    def __init__(self, rms_level=0):
        self._rms_level = rms_level

    def __call__(self, s, *args, **kwargs):
        # linear rms level and scaling factor
        r = 10**(self._rms_level / 10.0)
        a = np.sqrt((len(s) * r**2) / np.sum(s**2))

        return s * a


class PreEmphasize:
    def __init__(self, coeff):
        self._coeff = coeff
        return

    def __call__(self, s, *args, **kwargs):
        return np.append(s[0], s[1:] - self._coeff * s[:-1])


class Decode:
    def __init__(self, dict):
        self._dict = dict
        return

    def __call__(self, key):
        return self._dict[key]


class StringMatcher:
    def __init__(self, regex):
        self._compiled_regex = re.compile(regex)

    def __call__(self, s):
        try:
            return self._compiled_regex.findall(s)[0]
        except IndexError:
            return None


class StringMapper:
    def __init__(self, rules):
        self._compiled_rules = []
        for rule in rules:
            pattern, target = rule
            self._compiled_rules.append((re.compile(pattern), target))

    def __call__(self, s):
        for rule in self._compiled_rules:
            pattern, target = rule
            if (re.search(pattern, s)):
                return target

        return None


class AsConstant:
    def __init__(self, value):
        self._value = value

    def __call__(self, _):
        return self._value


class Threshold:
    def __init__(self, threshold):
        self._threshold = threshold

    def __call__(self, value):
        return value >= self._threshold


class UpperCaser:
    def __init__(self):
        return

    def __call__(self, s):
        return s.upper()


class LowerCaser:
    def __init__(self):
        return

    def __call__(self, s):
        return s.lower()
