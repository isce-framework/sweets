# Unreleased


# [0.2.0](https://github.com/opera-adt/dolphin/compare/v0.2.0...v0.3.0) - 2023-08-23

**Fixed**
- Geometry/`static layers` file creation from new COMPASS changes

**Dependencies**
- Upgraded `pydantic >= 2.1`
- Pinned minimum dolphin version due to mamba weirdness

# [0.1.0](https://github.com/isce-framework/sweets/commits/v0.1.0) - 2023-08-22


First version of processing workflow.

**Added**
- Created modules for DEM creation, ASF data download, geocoding SLCs, interferogram creation, and unwrapping.
- Created basic plotting utilities for interferograms and unwrapped interferograms.
- CLI commands `sweets config` and `sweets run` to configure a workflow and to run it.
