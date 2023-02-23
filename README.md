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

```python
from sweets.core import Workflow

# Pick your area of interest as a bounding box
bbox = [-120, 34.5, -118.5, 35.5]  # random one
# Pick your start/ending dates and track (relative orbit) number
# dates can be strings or datetime objects
start, end = "2020-01-01", "2020-12-31"
# Pick which track (relative orbit) you want to process
track = "2020-01-01", "2020-12-31", 78

w = Workflow(
    bbox=bbox,
    start=start,
    end=end,
    track=track,
    # (optional) Set the maximum temporal baseline (in days) for interferograms
    max_temporal_baseline=180,
)

# Run the workflow
w.run()
```


## License

This software is licensed under your choice of BSD-3-Clause or Apache-2.0 licenses. See the accompanying LICENSE file for further details.

SPDX-License-Identifier: BSD-3-Clause OR Apache-2.0
