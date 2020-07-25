# ShowBees - Beehive sound analysis

This repository presents several experiments related to the interpretation of sounds produced by bees, as a mean of detecting beehives' health status and stress level.

---
## Pre-requisites
In order to execute properly this github code, you'll need **Jupyter notebook** or **Jupiter lab** (https://jupyter.org/), preferably part of a broader **Anaconda** install ( https://www.anaconda.com/products/individual ), as well as the following additional packages/libraries:

### numba

*Due to a bug in the current version of librosa, causing a `ModuleNotFoundError: No module named 'numba.decorators'` error message, it is mandatory to downgrade from the latest numba version, and specifically install* ***0.48***

Install it with `python -m pip install numba==0.48 --user`

*This may change in newer versions of librosa, so your mileage may vary. You may try to run notebooks and downgrade to 0.48 only if you met the aforementioned error message*

### soundFile

Install it with `python -m pip install soundFile --user`

*Note: to allow mp3 file management in a Windows Anaconda install of Jupyter you may need to install additional codecs:*

Install the needed codecs either with: `conda install -c conda-forge ffmpeg` or pip


### pyprojroot

Allows finding root directory in Python projects, just like the **R** `here` and `rprojroot` packages.

Install it using `conda install -c conda-forge pyprojroot` or pip

### checksumdir

To allow replicability checks, output directories of various processes within this repository are "signed" using a md5 hash. This reference hash is "frozen" in each experiment manifest.

As this hash is displayed at the end of each major process completion, one can infer at a glance if the obtained result shares the same hash as the reference experiment he or she is trying to replicate.

Install it using: `python -m pip install checksumdir --user`

You may now use it as follow:
```python
from checksumdir import dirhash

directory  = 'D://path/to/directory'
md5hash    = dirhash(directory, 'md5')

print(directory, md5hash)
```


### ipywidgets

This is needed in order to use nice widgets in the Jupyter environment, in particular progress bars, thus avoiding cluttering your notebooks' output with endless log lines just for progress tracking purpose.


Install it using `conda install -c conda-forge ipywidgets`

or
```
pip install ipywidgets
jupyter nbextension enable --py widgetsnbextension
```

## System Information

Experimental results documented in this repository were produced using the configuration below:

```
CPython 3.7.7
IPython 7.16.1

conda 4.8.3
numba 0.48.0
tensorflow 1.14.0
keras 2.3.1
scipy 1.5.0
sklearn 0.23.1
soundfile 0.10.3.post1
librosa 0.7.2

compiler   : MSC v.1916 64 bit (AMD64)
system     : Windows
release    : 10
machine    : AMD64
processor  : Intel64 Family 6 Model 142 Stepping 10, GenuineIntel
CPU cores  : 8
interpreter: 64bit
```

---
## Repository structure

The root directory containsthe following files:

- this **README.md** file
- a **.here** empty file, needed to mark the directory as top-level, thus allowing the mooltipath mechanism (described later in this document) to work from any subsdirectory of any depth 

And directories:

 
- an **experiments** directory containing:
  - one subdirectory EXPxyz_*<EXPERIMENT_NAME_IN_SNAKE_CASE>* for each experiment
  - a README.md file with the list of experiments (experiment name and short description), as well as an introduction to the experiments common process, if needed.

- each **experiments/EXPxyz_*<EXPERIMENT_NAME_IN_SNAKE_CASE>*** directory contains:
  - a README.md file with a detailed description of the experiment
  - the notebook(s) used to conduct the experiment,
  - optionnaly, output files produced during the notebook(s) execution. If needed, adhoc subdirectories may be used for organizational purposes.
  - optionnaly, python files specific to this experiment (see Rules below). If needed, adhoc subdirectories may be used for organizational purposes.  
  
- a **codelib** directory containing:
  - all python modules shared by the aforementioned notebooks,
  - a README.md with a short description of each module's purpose
  - Adhoc subdirectories may be used to organize modules per domain (e.g. preprocessing, classifiers, visualization, etc...)  
  
- a **datasets** directory containing:
  - one manifest (*.lof = **l**ist **o**f **f**iles) file per dataset, listing de raw data (sound) files part of this dataset   
  - one subdirectory per dataset where experiment datasets will be built from the reference dataset, unsing the coresponding manifests. These subdirectories don't exist at initialisation time and willbe created dynamically.  
  
*Note: Within the `datasets` directory, all subdirectory are .gitignored*
  
--- 
## Repository Guidelines

This repository has been built having in mind theses guidelines:

- A python code (.py) file must never be duplicated across experiments. The codelib directory specifically exists for the specific purpose of allowing common code sharing between experiments.
- Duplication of IPython code between notebooks is obviously allowed, but must be limited to:
  - parameters setting, orchestration of macro functions calls and results visualisation (notebooks' code should be sequential without any complex logic)
  - exploratory snippets not meant for experiment replication (non trustable). They should ideally reside in a `snippets` subdirectory
- Python code (.py) files specific to a given experiment should be avoided in this experiment directory (as the repository normally contains only related experiments, which are in fact various scenarii around a same domain, they should generally use common code). Eaxh time it's possible, the code must be made as generic an reusable as possible, at least in the context of the proposet repository structure
- `proxycodelib.py`(see below) file as well as implementations of notebooks as selfcontained regular python files for execution out of interactive python environment are to notable exceptions to the above rule.   
  
--- 
## How to replicate an experiment

To replicate an experiment, you should:

- **Build the input dataset from a reference dataset** : As datasets may be quite large, they are not saved in this repository, and must be build using the following steps: 
  - Get the reference dataset: Reference datasets must be retrieved from the internet. Links to these datasets are provided at the bottom of this file, as well as in the **datasets** README. Download them and save them in a directory somewhere on your computer.
  - Build the experiment dataset: 
    - Experiments datasets are subsets or mixtures of the reference datasets.
    - Their content is detailled in the corresponding manifest in the `datasets` directory.
    - They can be either built on the fly when running an experiment for the first time, or upfront using the generic notebook present in the `datasets` directory
    - Check the experiment dataset: Compute a MD5 hash over the directory you just created. This hash should be identical to the one provided in the experiment dataset detailed description. If not, check that the files list is correct. 
    - From this point, you have reasonably insured that you have the same initial conditions than the original experiment (and so, you can expect to obtain the same results;-) )
  
  
## How to add or extend an experiment  

In addition to the various functions available within the codelib directory modules, two specific mechanisms are provided to enable the proposed repository structure:

- **proxycodelib** : proxicodelib is a module that should be imported in any notebook willing to make use of a function from the codelib directory. It make the codelib modules accessible from anywhere in the repository.
- **mooltipath**: *mooltipath(*args)* is a function part of the codelib `jupytools` module.
  - When imported, it walks from the current directory up to the first directory containing a `.here` file and build an absolute path joining this root directory path with the list of paths passed in \*args (This operation is performed once).
  - As such, it allows to use stable paths relative this root directory, independant from the notebook location.
  - It is operating system agnostic and takes care of any needed path normalization (/ vs \\). So you can always use '/' as a separator when defining path strings. I

Ex:

Let's say this repository resides locally on your computer in C:\Users\me\ShowBees

And that you have a notebook in the experiments/EXP01 directory withing this repository (actually, this is not relevant)

Then

```python
import proxycodelib
from jupytools import mooltipath

# Without arguments
# will print the repository root directory C:/Users/me/ShowBees
print(mooltipath())

# if any argument has a leading slash, will build an absolute path
# directlyusing the last argument with leading slash
#
print(mooltipath('/a')) # will print C:\a
print(mooltipath('/a', '/b', '/c')) # will print C:\c
print(mooltipath('/a', '/b', 'c', '/d')) # will print C:\d

# if there is no argument with a leading slash (ony relative paths)
# will build an absolute path by joining these relative paths,
# using the repository root directory as a starting point
# (This is the most useful case)
#
print(mooltipath('a')) # will print C:/Users/me/ShowBees/a
print(mooltipath('a/b','c\\d/e', 'f/foo.bar')) # will print C:\Users\me\ShowBees\a\b\c\d\e\f\foo.bar
```





---
## Credits & Notes
- Most of this repository code is a reuse of the (either untouched, slightly modified or heavily refactored) code from:
  - **Nolasco, Ines** : https://www.researchgate.net/profile/Ines_Nolasco
  - **Khellouf, Leila** : https://fr.linkedin.com/in/leila-khellouf-174704197
  - **Fourer, Dominique** : https://www.researchgate.net/profile/Dominique_Fourer , https://fourer.fr/
  
- Reference datasets used in this repo are built from:
  - https://zenodo.org/record/1321278
  
- Documentation was generated with **Sphinx** ( https://www.sphinx-doc.org/ )

- Detailed notebook hardware, os, language and libraries signature was performed using **watermark** ( https://github.com/rasbt/watermark )

- Initial exploratory analysis was performed using:
  - **London University Sonic Visualiser** ( https://www.sonicvisualiser.org/ ) and **Vamp Plugins** ( https://www.vamp-plugins.org/download.html )
  - **Google Embedding Projector** ( https://projector.tensorflow.org/ )

- No animals were harmed in the making of this repository