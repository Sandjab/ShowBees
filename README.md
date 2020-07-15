# ShowBees
## Beehive sound analysis

## Pre-requisites
In order to execute properly this github code on Windows, you'll need:

Jupyter notebook or Jupiter lab.

The following additional packages/librarie

numba version 0.48

Due to a bug in the current version of librosa, causing a `ModuleNotFoundError: No module named 'numba.decorators'` you'll need to downgrade from the latest numba version, and specifically install 0.48

`python -m pip install numba==0.48 --user`

soundFile

`python -m pip install soundFile --user`

to allow mp3 file management in soundfile

`conda install -c conda-forge ffmpeg`
