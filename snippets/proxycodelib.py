""" Little helper allowing import of modules from the codelib directory and build of absolute path from project root, based on a marker file"""
import sys
from pyprojroot import here
from os import fspath
from os.path import join

sys.path.append(join(fspath(here()),'codelib'))
