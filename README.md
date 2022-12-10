# intro_nilearn

[![Jupyter Book Badge](https://jupyterbook.org/badge.svg)](https://main-educational.github.io/intro_nilearn/intro.html) 
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/main-educational/intro_nilearn/HEAD)
[![GitHub size](https://img.shields.io/github/repo-size/main-educational/intro_nilearn)](https://github.com/main-educational/intro_nilearn/archive/master.zip)
[![GitHub issues](https://img.shields.io/github/issues/main-educational/intro_nilearn?style=plastic)](https://github.com/main-educational/intro_nilearn)
[![GitHub PR](https://img.shields.io/github/issues-pr/main-educational/intro_nilearn)](https://github.com/main-educational/intro_nilearn/pulls)
[![License](https://img.shields.io/github/license/main-educational/intro_nilearn)](https://github.com/main-educational/intro_nilearn)



This is the jupyter book of [Machine learning in functional MRI using Nilearn](https://main-educational.github.io/intro_nilearn/intro.html).
See the introduction of the jupyter book for more details, and acknowledgements.
The website is built with the [jupyter book](https://jupyterbook.org/) project, and deployed using github.

### Build the book

If you want to build the book locally:

- Clone this repository
- Run `pip install -r binder/requirements.txt` (it is recommended to run this command in a virtual environment)
- For a clean build, remove `content/_build/`
- Run `jb build content/`

An html version of the jupyter book will be automatically generated in the folder `content/_build/html/`.

### Hosting the book

The html version of the book is hosted on the `gh-pages` branch of this repo. Navigate to your local build and run,
- `ghp-import -n -p -f content/_build/html`

This will automatically push your build to the `gh-pages` branch. 
More information on this hosting process can be found [here](https://jupyterbook.org/publish/gh-pages.html#manually-host-your-book-with-github-pages).
