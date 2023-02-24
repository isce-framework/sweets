# sweets
Workflow for creating unwrapped interferograms from Sentinel-1 geocoded SLCs.



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

For the unwrapping portion, it is assumed you have installed [SNAPHU](https://web.stanford.edu/group/radar/softwareandlinks/sw/snaphu/) and have `snaphu` in your path.


## Usage

From the command line, installing will create a `sweets` executable. You can run `sweets --help` to see the available options.
```bash
sweets --help
```
You need to specify the bounding box (left, bottom, right, top) of the area of interest, the start dates (and end date, or it is assumed to be today), and the track number.

```bash
sweets --bbox -102.3407 31.9909 -101.9407 32.3909 --start "2022-10-15" --track 78
```

Or you can set all the parameters in python:
```python
from sweets.core import Workflow
bbox = (-102.3407 31.9909 -101.9407 32.3909)
start = "2020-01-01"  # can be strings or datetime objects
track = 78
w = Workflow(bbox=bbox, start=start, start=start, track=track)
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



## License

This software is licensed under your choice of BSD-3-Clause or Apache-2.0 licenses. See the accompanying LICENSE file for further details.

SPDX-License-Identifier: BSD-3-Clause OR Apache-2.0
