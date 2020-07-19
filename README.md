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

Install the needed codecs with: `conda install -c conda-forge ffmpeg`

### checksumdir

To allow replicability checks, output directories of various processes within this repository are "signed" using a md5 hash. This reference hash is "frozen" in each experiment documentation.

As this hash is displayed at the end of each major process completion, one can check at a glance if the obtained result shares the same hash as the reference hash of the experiment he or she is trying to replicate.

Install it using: `python -m pip install checksumdir --user`

<ins>From python</ins>
```python
"""directory hash computation"""
from checksumdir import dirhash

directory  = 'D:/datasets/sounds/SANDBOX'
md5hash    = dirhash(directory, 'md5')

print(directory, md5hash)
```

<ins>From the command line</ins>
```
checksumdir /path/to/directory
```

### ipywidgets

This is needed in order to use nice widgets in the Jupyter environment, in particular progress bars, thus avoiding cluttering your notebooks' output with endless log lines just for progress tracking purpose.

```
pip install ipywidgets
jupyter nbextension enable --py widgetsnbextension
```
or

`conda install -c conda-forge ipywidgets`


---
## Repository structure

- the **experiments** directory contains:
  - one subdirectory EXPxyz for each experiment
  - a README.md file with the list of experiments (experiment code EXPxyz, name, and short description), as well as an introduction to the experiments common process, if needed.

- each **experiments/EXPxyz** directory contains:
  - a README.md file with a detailed description of the experiment
  - the notebook(s) used to conduct the experiment,
  - optionnaly, output files produced during the notebook(s) execution. If needed, adhoc subdirectories may be used for organizational purposes.
  - optionnaly, python files specific to this experiment (see Rules below). If needed, adhoc subdirectories may be used for organizational purposes.

- the **codelib** directory contains:
  - all python modules shared by the aforementioned notebooks,
  - a README.md with a short description of each module's purpose
  - Adhoc subdirectories may be used to organize modules per domain (e.g. preprocessing, classifiers, visualization, etc...)

--- 
## Repository Rules
- A python code (.py) file must never be duplicated across experiments. The codelib directory specifically exists for the purpose of allowing common code sharing between experiments.
- Duplication of IPython code between notebooks is obviously allowed, but must be limited to:
  - parameters setting, orchestration of macro functions calls and results visualisation (notebooks' code should be sequential without any complex logic)
  - exploratory snippets not meant for experiment replication (non trustable)
- Python code '.py) files specific to a given experiment should be avoided in experiment directory (as a repository normally contains only related experiments, which are in fact various scenarii around a same domain, they should generally use common code).
- Re-implementations of notebooks as selfcontained regular python files for execution out of interactive python environment is notable exception of the above rule. 
 
---
## Credits
- Most of this repository code is a reuse of the (either untouched, slightly modified or heavily refactored) code from:
  - **Nolasco, Ines** : https://www.researchgate.net/profile/Ines_Nolasco
  - **Khellouf, Leila** : https://fr.linkedin.com/in/leila-khellouf-174704197
  - **Fourer, Dominique** : https://www.researchgate.net/profile/Dominique_Fourer , https://fourer.fr/
  
- Reference datasets used in this repo are built from:
  - https://zenodo.org/record/1321278
  
- Documentation was generated with **Sphinx** ( https://www.sphinx-doc.org/ )

- Detailed notebook hardware, os, language and libraies signature was performed using **watermark** ( https://github.com/rasbt/watermark )

- Initial exploratory analysis was performed using:
  - **London University Sonic Visualiser** ( https://www.sonicvisualiser.org/ ) and **Vamp Plugins** ( https://www.vamp-plugins.org/download.html )
  - **Google Embedding Projector** ( https://projector.tensorflow.org/ )

- No animals were harmed in the making of this repository