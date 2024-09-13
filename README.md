# pyFishSenseDev [![Python package](https://github.com/UCSD-E4E/pyFishSenseDev/actions/workflows/python-package.yml/badge.svg)](https://github.com/UCSD-E4E/pyFishSenseDev/actions/workflows/python-package.yml)
This repository contains a Python implementation of the Fishsense lite pipeline. It contains tools for camera calibration, laser calibration, and fish length calculation. This pipeline assumes that you already have access to all the data necessary to run this pipeline. As this code is meant to be a temporary measure, the user may need to edit some file paths, so apologies in advance. 

## After cloning
1. Install the `poetry` packages.  `poetry install`.
2. Install `git annex`.  On Ubuntu, execute `sudo apt install git-annex`.
3. Install `git-annex-remote-synology` from [git-annex-remote-synology](https://github.com/UCSD-E4E/git-annex-remote-synology).
4. Run `git-annex-remote-synology setup --hostname e4e-nas.ucsd.edu`
5. Run `git annex enableremote synology`.
6. Download data `git annex copy --from=synology`.

## Development Dependencies
We provide a `docker` container which has the dependencies pre-installed.  In order to use this, please ensure you have `docker` installed on your system.  When running Visual Studio Code, ensure that you use the option to reopen in the container.  This step will time for the intial setup.

## Execute Unit Tests
```
poetry run pytest
```
