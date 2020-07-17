# ShowBees - Beehive sound analysis

This repository presents several experiments related to the interpretation of sounds produced by bees, as a mean of detecting beehives' health status and stress level.

---
## Pre-requisites
In order to execute properly this github code, you'll need **Jupyter notebook** or **Jupiter lab** (https://jupyter.org/), as well as the following additional packages/libraries:

**numba** version 0.48 :

*Due to a bug in the current version of librosa, causing a `ModuleNotFoundError: No module named 'numba.decorators'` error message, it is mandatory to downgrade from the latest numba version, and specifically install 0.48*

Install it with `python -m pip install numba==0.48 --user`

*This may change in newer versions of librosa, so your mileage may vary. You may try to run notebooks and downgrade to 0.48 only if you met the aforementioned error message*
	

**soundFile**

Install it with `python -m pip install soundFile --user`

*Note: to allow mp3 file management in a Windows Anaconda install of Jupyter you may need to install additional codecs:*

Install the needed codecs with: `conda install -c conda-forge ffmpeg`


**ipywidgets**

This is needed in order to use nice widgets in the Jupyter environment, in particular progress bars, thus avoiding clustering your notebooks' output with endless log lines just for progress tracking purpose.

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
  - a README.md file with a detailed description the experiment
  - the notebook(s) used to conduct the experiment,
  - optionnaly, output files produced during the notebook(s) execution. If needed, adhoc subdirectories may be used for organizational purposes.
  - optionnaly, python files specific to this experiment (see Rules below). If needed, adhoc subdirectories may be used for organizational purposes.

- the **codelib** directory contains:
  - all python modules shared by the aforementioned notebooks,
  - a README.md with a short description of each module's purpose
  - subdirectories may be used to group modules per domain (e.g. preprocessing, classifiers, visualization, etc...)

--- 
## Repository Rules
- A python code file must never be duplicated across experiments. Tne codelib directory is here to allow common code sharing between experiments.
- Replication of IPython code between notebooks is obviously allowed, but must be limited to parameter setting and function calls (no complex code in notebooks)
- Python code specific to a given experiment should be avoided in .py files (a repository normally contains only related experiments, which are in fact various scenarii around a same domain, generally using common code).
 
---
## Credits
- Most of this repo code is a reuse of the (sometimes refactored) code from:
  - **Nolasco, Ines** : https://www.researchgate.net/profile/Ines_Nolasco
  - **Khellouf, Leila** : https://fr.linkedin.com/in/leila-khellouf-174704197
  - **Fourer, Dominique** : https://www.researchgate.net/profile/Dominique_Fourer , https://fourer.fr/
- Reference datasets used in this repo are:
  - https://zenodo.org/record/1321278
- Documentation was generated with Sphinx ( https://www.sphinx-doc.org/ )
- Detailed notebook watermarking was performed using https://github.com/rasbt/watermark )
- No animals were harmed in the making of this repository