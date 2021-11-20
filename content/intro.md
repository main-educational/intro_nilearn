# Introduction

These two tutorials will introduce you to the [nilearn](https://nilearn.github.io/stable/index.html) 
library to manipulate and process fMRI data, and to [scikit-learn](https://scikit-learn.org/stable/) 
to apply machine learning techniques on your data.

It was developed for use at the [Montreal AI and Neuroscience (MAIN)](https://www.main2021.org/) 
conference in November 2021.

### Pre-requisites

You need to have access to a terminal with Python 3. 
If it not already the case, 
[here](https://realpython.com/installing-python/#how-to-check-your-python-version-on-windows) 
is a guide to install python 3 on any OS.

## Setup

Firstly, clone/download this repository to your machine and navigate to the directory.

```bash
git clone https://github.com/mtl-AI-neuroscience/intro_ML.git
cd intro_ML
```

We encourage you to use a virtual environment for this tutorial 
(and for all your projects, that's a good practice). 
To do this, run the following command in your terminal, it will create the
environment in a folder named `env_tuto`:

```bash
python3 -m venv env_tuto
```
Then the following command will activate the environment:

```bash
source env_tuto/bin/activate
```
Finally, you can install the required libraries:

```bash
pip install -r requirements.txt
```

## Usage

Now that you are all set, you can run the notebooks with the command:

```bash
jupyter notebook
```

## Acknowledgements

This tutorial was presented by 
[François Paugam](https://github.com/FrancoisPgm),
and [Hao-Ting Wang](https://wanghaoting.com/).

It is rendered here using [Jupyter Book](https://github.com/jupyter/jupyter-book),
<!-- with compute infrastructure provided by the [Canadian Open Neuroscience Platform (CONP)](http://conp.ca). -->

The content of notebook was iterated based on work from the past presentors:
[Pierre Bellec](https://simexp.github.io/lab-website/),
[Elizabeth DuPre](https://elizabeth-dupre.com),
[Greg Kiar](http://gkiar.me),
and [Jake Vogel](https://scholar.google.ca/citations?user=1m6yqlwAAAAJ&hl=en).

Past versions of this tutorial:
[MAIN 2018](https://brainhack101.github.io/introML-book/intro), 
[Neurohackademy 2020](https://emdupre.github.io/nha2020-nilearn/01-data-structures.html)

Introduction to fMRI is taken from:
[PSY3018 Méthodes en neurosciences cognitives](https://psy3018.github.io/intro.html)
