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

## Setup

You need a `~/.netrc` file with NASA Earthdata credentials to download data from ASF:

```
machine urs.earthdata.nasa.gov
    login <username>
    password <password>
```

Register at https://urs.earthdata.nasa.gov/users/new if you don't have an account.

### Optional: CDSE credentials

To download Sentinel-1 SLC granules from the [Copernicus Data Space Ecosystem (CDSE)](https://dataspace.copernicus.eu/) instead of ASF, you need CDSE credentials. These can be provided in one of two ways:

- Add an entry to your `~/.netrc` file:
  ```
  machine dataspace.copernicus.eu
      login <cdse-username>
      password <cdse-password>
  ```
- Or set environment variables:
  ```bash
  export CDSE_USERNAME=<cdse-username>
  export CDSE_PASSWORD=<cdse-password>
  ```

A free CDSE account can be registered at [dataspace.copernicus.eu](https://dataspace.copernicus.eu/).


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

### Using CDSE for Sentinel-1 Download

By default, Sentinel-1 SLC granules are downloaded from the [Alaska Satellite Facility (ASF)](https://asf.alaska.edu/). As an alternative, particularly suited for European users or those operating within the European cloud ecosystem, granules can be downloaded from [CDSE](https://dataspace.copernicus.eu/) by passing `--download-source CDSE`:

```bash
sweets config --bbox -102.2 32.15 -102.1 32.22 --start 2022-12-15 --end 2022-12-29 --track 78 --download-source CDSE
```

Or in a YAML config file, set:
```yaml
download_source: CDSE
```

### Configuration from Python

Alternatively, you can configure everything in python:
```python
from sweets.core import Workflow
bbox = (-102.3407, 31.9909, -101.9407, 32.3909)
start = "2020-01-01"  # can be strings or datetime objects
track = 78
w = Workflow(bbox=bbox, asf_query=dict(start=start, end=end, relativeOrbit=track))
w.run()
```

To use CDSE for downloads:
```python
w = Workflow(
    bbox=bbox,
    download_source="CDSE",
    asf_query=dict(start=start, end=end, relativeOrbit=track),
)
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
