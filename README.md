# sweets
Workflow for creating interferograms from Sentinel-1 geocoded SLCs.



## Install

The following will install `sweets` into a conda environment.

1. Download source code:
```bash
git clone https://github.com/opera-adt/sweets.git && cd sweets
```
2. Install dependencies:
```bash
conda env create --file conda-env.yml
```

or if you have an existing environment:
```bash
conda env update --name my-existing-env --file conda-env.yml
```

3. Install `sweets` via pip:
```bash
conda activate sweets-env
python -m pip install .
```


## Usage


## License

This software is licensed under your choice of BSD-3-Clause or Apache-2.0 licenses. See the accompanying LICENSE file for further details.

SPDX-License-Identifier: BSD-3-Clause OR Apache-2.0
