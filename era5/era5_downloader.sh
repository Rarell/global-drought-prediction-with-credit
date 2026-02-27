#!/bin/bash
# Heavily based on code created by Dr. Andrew Fagg
# Modified: Stuart Edris
#1;95;0c
# Collect ERA5 data for a set year and aggregate them unto daily temporal scale
# Grids saved as .nc files with lat and lon
# 
# Reasonable partitions: debug_5min, debug_30min, normal/32gb_20core/64gb_24core
#

#SBATCH --partition=normal
#SBATCH --cpus-per-task=10
# memory in MB
#SBATCH --mem=3072
# The %04a is translated into a 4-digit number that encodes the SLURM_ARRAY_TASK_ID. 
#SBATCH --output=outputs/era5_download_exp%04a_stdout.txt
#SBATCH --error=outputs/era5_download_%04a_stderr.txt
#SBATCH --time=28:00:00
#SBATCH --job-name=era5_download
#SBATCH --mail-user=sgedris@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/scratch/rarrell/ERA5
##SBATCH --array=0-23
#
#################################################



### Code to run on the supercomputer
. /home/rarrell/tf_setup.sh
conda activate tf

# Note only one set of experiments should be uncommented at per batch run
hostname

# Collect the data for a given year

# Sample to download ERA5 from Copernicus store for a single level variable
python -u download_data.py \
--variable surface_net_solar_radiation \
--var_sname_era ssr \
--years $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID \
--era5_dataset reanalysis-era5-single-levels \
--process

# Sample to download ERA5 from Copernicus store for variables on pressure levels
python -u download_data.py \
--variable u_component_of_wind \
--var_sname_era u \
--years $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID \
--era5_dataset reanalysis-era5-pressure-levels \
--pressure_level 500 \
--process

# Sample for downloading ERA5 from Google Earth engine for variables on pressure levels
python era5_downloader_from_google_engine.py \
--variables u_component_of_wind \
--var_snames_era u \
--years $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID \
--era5_dataset date-variable-pressure_level \
--pressure_level 500 \
--process

# Sample for downloading ERA5 from Google Earth engine for a single level variable
python era5_downloader_from_google_engine.py \
--variables 10m_wind_gust_since_previous_post_processing \
--var_snames_era fg10 \
--years $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID \
--era5_dataset date-variable-single_level \
--process
