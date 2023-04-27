""" Little helper allowing import of modules from the codelib directory and build of absolute path from project root, based on a marker file"""
import sys
import pyprojroot
from os import fspath

base_path = pyprojroot.find_root(pyprojroot.has_file(".kilroy"))

try:
    #sys.path.append(fspath(here(has_file=".kilroy", warn=True)))
    sys.path.append(fspath(base_path))
except (RecursionError):
    raise FileNotFoundError('Marker file (.kilroy) not found in project tree')