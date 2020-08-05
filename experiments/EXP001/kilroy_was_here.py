""" Little helper allowing import of modules from the codelib directory and build of absolute path from project root, based on a marker file"""
import sys
from pyprojroot import here
from os import fspath
from os.path import join
try:
    sys.path.append(fspath(here(project_files=['.kilroy'], warn=True)))
except (RecursionError):
    raise FileNotFoundError('Marker file (.kilroy) not found in project tree')