---
layout: docs
docid: "installation"
title: "Installation"
permalink: /docs/installation.html
subsections:
  - title: Install
    id: install
---

<a id="install"> </a>

pyGrams.py has been developed to work on both Windows and MacOS. To install:

1. Please make sure Python 3.6 is installed and set in your path.  

   To check the Python version default for your system, run the following in command line/terminal:

   ```
   python --version
   ```

   **_Note_**: If Python 2.x is the default Python version, but you have installed Python 3.x, your path may be setup to use `python3` instead of `python`.

2. To install pyGrams packages and dependencies, from the root directory (./pyGrams) run:

   ``` 
   pip install -e .
   ```

   This will install all the libraries and then download their required datasets (namely NLTK's data). Once installed, 
   setup will run some tests. If the tests pass, the app is ready to run. If any of the tests fail, please email [ons.patent.explorer@gmail.com](mailto:ons.patent.explorer@gmail.com) with a screenshot of the failure so that we may get back to you, or alternatively open a [GitHub issue here](https://github.com/datasciencecampus/pyGrams/issues).