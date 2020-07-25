from pyprojroot import here
from pathlib import Path

rootpath = here()

def mooltipath(*args):
    return rootpath.joinpath(*args)