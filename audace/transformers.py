import re


class Identity:
    def __init__(self):
        return

    def __call__(self, value):
        return value


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
