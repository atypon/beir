from functools import reduce
from os.path import expandvars

import yaml


class DotDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = DotDict(value)

    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError:
            return super().__getattr__(k)
        if isinstance(v, dict):
            return DotDict(v)
        return v

    def __setattr__(self, k, v):
        if isinstance(v, dict):
            v = DotDict(v)
        self[k] = v

    def __getitem__(self, k):
        if isinstance(k, str) and "." in k:
            k = k.split(".")
        if isinstance(k, (list, tuple)):
            return reduce(lambda d, kk: d[kk], k, self)
        return super().__getitem__(k)

    def get(self, k, default=None):
        if isinstance(k, str) and "." in k:
            try:
                return self[k]
            except KeyError:
                return default
        return super().get(k, default=default)


def load_configurations(path: str) -> DotDict:
    """
    Used for parsing configuration files
    :param str path: path to conf file
    :returns DotDict: dictionary accessing fields with dot notation
    """
    with open(path) as f:
        cfg = yaml.safe_load(expandvars(f.read()))
    return DotDict(cfg)
