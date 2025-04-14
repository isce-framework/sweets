# source: https://github.com/conda-forge/numpy-feedstock/issues/347#issuecomment-2746317575
#!/bin/bash
set -e

LIB_PATH="${PIXI_PROJECT_ROOT}/.pixi/envs/default/lib"

# Find all dylib files
find "$LIB_PATH" -name "*.dylib" -o -name "*.so" | while read -r library; do
    echo "Processing $library..."

    # Extract all LC_RPATH entries
    rpaths=$(otool -l "$library" | grep -A2 LC_RPATH | grep "path " | awk '{print $2}')

    # Create a temporary file to track seen rpaths
    temp_file=$(mktemp)

    # Check for duplicates and remove them
    echo "$rpaths" | while read -r rpath; do
        if [[ -z "$rpath" ]]; then
            continue
        fi

        if grep -q "^$rpath$" "$temp_file"; then
            echo "  Removing duplicate RPATH: $rpath"
            install_name_tool -delete_rpath "$rpath" "$library" || true
        else
            echo "$rpath" >> "$temp_file"
            echo "  Keeping RPATH: $rpath"
        fi
    done

    # Re-sign the library
    echo "  Re-signing $library"
    codesign --force --sign - "$library" || echo "  Warning: Could not sign $library"

    # Clean up the temporary file
    rm -f "$temp_file"

    echo "Done with $library"
    echo "-----------------------"
done

echo "All libraries processed!"
