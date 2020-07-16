# ShowBees - Beehive sound analysis


---
## Pre-requisites
In order to execute properly this github code, you'll need **Jupyter notebook** or **Jupiter lab**, as well as the following additional packages/libraries:

**numba** version 0.48 :

*Due to a bug in the current version of librosa, causing a `ModuleNotFoundError: No module named 'numba.decorators'` error message, it is mandatory to downgrade from the latest numba version, and specifically install 0.48*

Install it with `python -m pip install numba==0.48 --user`

*This may change in newer versions of librosa, so your mileage may vary. You may try to run notebooks and downgrade to 0.48 only if you met the aforementioned error message*
	

**soundFile**

Install it with `python -m pip install soundFile --user`

*Note: to allow mp3 file management in a Windows Anaconda install of Jupyter you may need to install additional codecs:*

Install the needed codecs with: `conda install -c conda-forge ffmpeg`


**ipywidgets**

This is needed in order to use nice widgets in the Jupyter environment, in particular progress bars, thus avoiding clustering you notebooks' output with endless log lines.

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
  - a README.md file with the list of experiments (experiment code EXPxyz, name, and short description)

- each **experiments/EXPxyz** directory contains:
  - a README.md file with a detailed description the experiment
  - the notebook(s) used to conduct the experiment,
  - optionnaly, output files produced during the notebook(s) execution

- the **codelib** directory contains:
  - all python modules needed by the aforementioned notebooks,
  - a README.md with a short description of each module's purpose