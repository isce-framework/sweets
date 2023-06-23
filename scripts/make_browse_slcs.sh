#!/bin/bash

# Enable common error handling options.
set -o errexit
set -o nounset
set -o pipefail

USAGE="Usage: $0 -d|--input-dir [-x <x pct>] [-y <y pct>] [-o|--output-dir <output directory>] [-P|--max-procs <max procs>]"

# Set default subsampling values
x_pct=10
y_pct=20
output_dir="."
max_procs=1

# Parse input arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
    -x)
        x_pct="$2"
        shift
        shift
        ;;
    -y)
        y_pct="$2"
        shift
        shift
        ;;
    -d | --input-dir)
        input_dir="$2"
        shift
        shift
        ;;
    -o | --output-dir)
        output_dir="$2"
        shift
        shift
        ;;
    -P | --max-procs)
        max_procs="$2"
        shift
        shift
        ;;
    -h | --help)
        echo "$USAGE"
        exit 0
        ;;
    *)
        echo "Unknown option: $key"
        exit 1
        ;;
    esac
done

# Check if input directory was provided
if [[ -z "${input_dir:-}" ]]; then
    echo "Input directory is required"
    echo "$USAGE"
    exit 1
fi

echo "Input dir: $input_dir"
echo "Output dir: $output_dir"
mkdir -p $output_dir

# Find .h5 files and pass them to xargs
find "$input_dir" -name "t[0-9]*.h5" | xargs --max-args 1 --max-procs $max_procs -I {} bash -c '
    # Extract the filename and form the GDAL string
    filename=$(basename -- "$1")
    filename="${filename%.*}"
    gdal_str="DERIVED_SUBDATASET:LOGAMPLITUDE:NETCDF:$1:\"data/VV\""
    output_file="$2/${filename}.tif"
    echo "$gdal_str -> $output_file"

    # Check if the output file already exists, skip if it does
    if [ -f "$output_file" ]; then
        echo "File $output_file already exists, skipping..."
    else

        # Use gdal_translate to subsample and save as .tif
        gdal_translate -q -ot Float32 -of GTiff -co TILED=YES -co COMPRESS=LZW -r near -outsize $3% $4% "$gdal_str" "$output_file"
    fi

' bash {} $output_dir $x_pct $y_pct
