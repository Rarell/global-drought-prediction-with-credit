#!/bin/bash

# Organize the raw and aggregated MODIS files collected and made by modis_downloader.py
# Final organization:
# ./
#  |---raw/
#  |------raw_variable/
#  |---------year/
#  |------------tile/
#  |---------------*.hdf
#  |---global/
#  |------variable/
#  |---------year/
#  |------------*.nc

# For raw/ directory: raw_variable uses all variables contained in the .hdf files
# e.g., evaporation_and_potential_evaporation, ndvi_and_evi

# The global/ directory contains the MODIS data aggregated to a 0.05 degree x 0.05 degree 
# global map and saved to .nc files.

# Inputs:
#    $1: String of the MODIS product being organized (e.g., MOD16A2GF is the product for ET and PET)

MODIS_PRODUCT=$1

# Determine some directory information based on the MODIS product
if [[ "$MODIS_PRODUCT" == "MOD16A2GF" ]]; then
    RAWDIRECTORY="./raw/evaporation_and_potential_evaporation/"
    GLOBALDIRECTORY=("./global/evaporation/" "./global/potential_evaporation/")
    VARIABLES=("modis.evaporation" "modis.potential_evaporation")
elif [[ "$MODIS_PRODUCT" == "MOD09A1" ]]; then
    RAWDIRECTORY="./raw/reflectance/"
    GLOBALDIRECTORY=("./global/evi/" "./global/ndvi/" "./global/lai/" "./global/fpar/")
    VARIABLES=("modis.evi" "modis.ndvi" "modis.lai" "modis.fpar")
elif [[ "$MODIS_PRODUCT" == "MOD15A2H" ]]; then
    RAWDIRECTORY="./raw/lai_and_fpar/"
    GLOBALDIRECTORY=("./global/lai/" "./global/fpar/")
    VARIABLES=("modis.lai" "modis.fpar")
fi
#### NEED TO ADD: Land cover, DSR


#tiles=find ./raw/evaporation_and_potential_evaporation/ -minidepth 1 -maxdepth 1 -type d {'*'}
# Start with an empty array
TILES=()

# Search all files in the raw/evaporation_and_potential_evaporation/2000/ directory 
# (any random directory with all the tiles will suffice)
for FILE in ./raw/evaporation_and_potential_evaporation/2000/*; do
    # Check if the file is a directory ([[ -d $FILE]])
    # If FILE is a directory (&&), then add $FILE to the array (+=)
    # Note the full directory is ./raw/evaporation_and_potential_evaporation/h00v07/, ...
    # only the tile reference is desired, which start at index 49 and ends at index 53.
    [[ -d $FILE ]] && TILES+=("${FILE:49:53}")
done

for YEAR in {2000..2024}; do
    echo $YEAR
    # Construct the string for the MODIS filenames
    for TILE in ${TILES[@]}; do
        STRING="${MODIS_PRODUCT}*"
        STRING+="${YEAR}"
        STRING+="*${TILE}*"
        echo $STRING
        # Construct the path to the tile directory
        find raw -maxdepth 1 -name $STRING | while read -r FILE ; do
            echo ${FILE:4:100}
            OUTPATH="$RAWDIRECTORY"
            OUTPATH+="${YEAR}/"
            OUTPATH+="${TILE}/"
            echo $OUTPATH
            # Make sure the file exists
            if [ ! -d "$OUTPATH" ]; then
                mkdir $OUTPATH
            fi
            #OUTPATH+="${FILE:4:100}"
            #echo $OUTPATH

            # Move MODIS files into their respective tile information
            mv $FILE $OUTPATH 
        done
    done
    
    echo "Moving aggregated files"
    
    for n in "${!VARIABLES[@]}"; do
        # Construct the strong for aggregated filenames
        STRING="${VARIABLES[n]}*"
        STRING+="${YEAR}*.nc"
        find . -maxdepth 1 -name "$STRING" | while read -r FILE ; do
            echo ${FILE:2:100}
            # Construct the path to the directory for the aggregated data files
            OUTPATH="${GLOBALDIRECTORY[n]}"
            OUTPATH+="${YEAR}/"
            #OUTPATH+="${FILE:2:100}"

            # Move the data files to their directory
            mv $FILE $OUTPATH
        done
    done
done

