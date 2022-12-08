# Introduction

```{figure} ../images/nilearn-logo.png
---
height: 250px
---
```

These two tutorials will introduce you to the [nilearn](https://nilearn.github.io/stable/index.html) 
library to manipulate and process fMRI data, and to [scikit-learn](https://scikit-learn.org/stable/) 
to apply machine learning techniques on your data.

It was developed for use at the [Montreal AI and Neuroscience (MAIN) 
educational workshop](https://main-educational.github.io) in [November 2021](https://www.main2021.org/) and [December 2022](https://www.main2022.org).

## Setup

### I know how to use Python, just give me the thing

You can download the jupyter notebook with the download icon on the top right of each page. 
Here's the basic requirements you need to install:

```
nilearn
matplotlib
seaborn
jupyter-book
```


### I need more step by step help....

::::{note}

You need to have access to a terminal with Python 3. 
If you have setup your environment based on instructions of [MAIN educational installation guide](https://main-educational.github.io/installation.html), you are good to go 🎉

If it not already the case, 
[here](https://realpython.com/installing-python/#how-to-check-your-python-version-on-windows) 
is a quick guide to install python 3 on any OS.
::::

1. Clone/download this repository to your machine and navigate to the directory.

    ```bash
    git clone https://github.com/main-educational/intro_nilearn.git
    cd intro_nilearn
    ```

2. We encourage you to use a virtual environment for this tutorial 
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

3. Navigate to the content of the book:
    ```bash
    cd content/
    ```

    Now that you are all set, you can run the notebooks with the command:

    ```bash
    jupyter notebook
    ```
    Click on the `.md` files. They will be rendered as jupyter notebooks 🎉


## Acknowledgements

This tutorial is presented by 
[Yasmin Mzayek](https://github.com/ymzayek),
and [Hao-Ting Wang](https://github.com/htwangtw).

It is rendered here using [Jupyter Book](https://github.com/jupyter/jupyter-book),
<!-- with compute infrastructure provided by the [Canadian Open Neuroscience Platform (CONP)](http://conp.ca). -->

The content of notebook was iterated based on work from the past presentors:
[François Paugam](https://github.com/FrancoisPgm),
[Pierre Bellec](https://simexp.github.io/lab-website/),
[Elizabeth DuPre](https://elizabeth-dupre.com),
[Greg Kiar](http://gkiar.me),
and [Jake Vogel](https://scholar.google.ca/citations?user=1m6yqlwAAAAJ&hl=en).

Past versions of this tutorial:
[MAIN 2018](https://brainhack101.github.io/introML-book/intro), 
[Neurohackademy 2020](https://emdupre.github.io/nha2020-nilearn/01-data-structures.html)

Introduction to fMRI is taken from:
[PSY3018 Méthodes en neurosciences cognitives](https://psy3018.github.io/intro.html)
