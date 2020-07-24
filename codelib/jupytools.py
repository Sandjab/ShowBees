from pyprojroot import here
from os import fspath
from os.path import join

rootpath = fspath(here())

def mooltipath(relpath=''):
    return join(rootpath, relpath)