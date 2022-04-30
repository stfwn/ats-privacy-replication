# Replication study of "Privacy-preserving Collaborative Learning with Automatic Transformation Search"

[![DOI](https://zenodo.org/badge/445183551.svg)](https://zenodo.org/badge/latestdoi/445183551)


This is the code behind a replication study of [Privacy-preserving
Collaborative Learning with Automatic Transformation
Search](https://arxiv.org/abs/2011.12505). The code for the original
implementation can be found [here](https://github.com/gaow0007/ATSPrivacy).

## Setup

1. Set up the conda environment.

```
conda env create -f environment.yml
conda activate reproducing-ats
```
2. (Optional) [Download our logs and
   checkpoints](https://archive.org/details/ats-privacy-replication) and unzip
   in the root directory of this repository, so that you now have a `logs`
   folder.
3. Read the usage section below or dive straight into our `report.ipynb` Jupyter notebook.

## Usage

The entrypoint for this project is `main.py`. It has three main capabilities:

- **Search:** perform a policy search.
- **Train:** train a model on a dataset, optionally using a particular policy
  and/or alternative defense.
- **Attack:** reconstruct an image given a gradient from a model.

See the prerendered `report.ipynb` Jupyter notebook for a tour through its
functionality. If you [download our logs and
checkpoints](https://archive.org/details/ats-privacy-replication), you can
fully rerender our figures and tables.

```
usage: main.py [-h] [--data-dir DATA_DIR] {search,train,attack,test} ...

optional arguments:
  -h, --help            show this help message and exit
  --data-dir DATA_DIR

command:
  {search,train,attack,test}
                        Action to execute
    search              Automatic transformation search
    train               Model training
    attack              Perform reconstruction attack
    test                Test a model
```
