# sweets
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/isce-framework/sweets/main.svg)](https://results.pre-commit.ci/latest/github/isce-framework/sweets/main)

Workflow for creating unwrapped interferograms from Sentinel-1 geocoded SLCs.

## Install

`sweets` is available to install via conda-forge:

```bash
mamba install -c conda-forge sweets
```

Alternatively, the following will install `sweets` into a conda environment.

1. Download source code:
```bash
git clone https://github.com/opera-adt/sweets.git && cd sweets
```
2. Install dependencies:
```bash
mamba env create --file conda-env.yml
```

or if you have an existing environment:
```bash
mamba env update --name my-existing-env --file conda-env.yml
```

3. Install `sweets` via pip:
```bash
conda activate sweets-env
python -m pip install .
```

## Usage

From the command line, installing will create a `sweets` executable. You can run `sweets --help` to see the available options.
```bash
sweets --help
```
To configure a workflow, the minimum inputs are
- the bounding box of the area of interest in degrees longitude/latitude as (left, bottom right top)
- the start date (and end date, or it is assumed to be today)
- the track (relative orbit) number.

For example:

```bash
sweets config  --bbox -102.2 32.15 -102.1 32.22 --start 2022-12-15 --end 2022-12-29 --track 78
```
This will make a YAML configuration file (by default `sweets_config.yaml`). You can inspect it to see all the configuration defaults.

Then you can kick off the workflow using
```bash
sweets run sweets_config.yaml
```

### Configuration from Python

Alternatively, you can configure everything in python:
```python
from sweets.core import Workflow
bbox = (-102.3407 31.9909 -101.9407 32.3909)
start = "2020-01-01"  # can be strings or datetime objects
track = 78
w = Workflow(bbox=bbox, asf_query=dict(start=start, end=end, relativeOrbit=track))
w.run()
```

You can also save the workflow to a config file for later use/to inspect or change parameters:
```
w.to_yaml()  # Saves to sweets_config.yml for inspection/tweaking
```

If you want to run this later from the config, you can do
```python
w = Workflow.from_yaml("sweets_config.yml")
w.run()
```

You can also print an empty config file to edit any parameters manually

```bash
sweets config --print-empty
```

## License

This software is licensed under your choice of BSD-3-Clause or Apache-2.0 licenses. See the accompanying LICENSE file for further details.

SPDX-License-Identifier: BSD-3-Clause OR Apache-2.0
