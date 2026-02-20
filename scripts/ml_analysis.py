'''Main script for conducting postprocessing analysis
after the CREDIT model is trained and predictions made.
Create figures to evaluate model performance, variability,
and examine case studies.

TODO:
'''
import gc
import os, sys, warnings
import numpy as np
import pandas as pd
import multiprocessing as mp
import zarr
import imageio
import time
from tqdm import tqdm
from argparse import ArgumentParser
from datetime import datetime, timedelta
from netCDF4 import Dataset
from glob import glob

from metric_calculations import calculate_metric, calculate_rpc, calculate_acc_in_space, calculate_rmse_in_space
from fd_calculations import calculate_climatology, calculate_sm_percentiles, calculate_sesr, calculate_fdii
from data_loading import load_climatology, load_persistence, load_metrics
from utils import subset_data, get_metric_information, new_sort
from plotting import (
    plot_metric,
    plot_violin, 
    make_score_cards, 
    make_comparison_map,
    make_anomaly_map, 
    make_comparison_subset_map, 
    make_anomaly_subset_map,
    make_metric_error_plots,
    make_histogram_scatter_plot
)

# Load zarr files in a similar method as zarr2 (there are currently errors without this)
zarr.config.set({'default_zarr_format': 2})            

# Define all currently supported subsets
subsets = ['nh', 'sh', 'tropics', 'africa', 'africa_nh', 'africa_sh', 'africa_tropics', 'conus', 'south central']

# Define all upper air and diagnostic variables (surface variables are the ones not included in these)
upper_air_variables = ['u', 'v', 'z', 'q_tot']
diagnostic_variables = ['ndvi', 'evi', 'lai', 'fpar', 'swvl1', 'swvl2', 'swvl3', 'swvl4', 'fdii1', 'fdii2', 'fdii3', 'fdii4', 'sesr', 'tp_30d']

# Define the variables skip in the analysis
skip_variables = ['time', 'latitude', 'longitude', 'forecast_step', 'datetime', 'time_new']

# Define the short name (used in nc files) for all the variables
short_names = {
    '200 mb u wind': 'u', '200_mb_u_wind': 'u', '500 mb u': 'u', '500_mb_u_wind': 'u',
    '200 mb v wind': 'u', '200_mb_v_wind': 'v', '500 mb v': 'u', '500_mb_v_wind': 'v',
    '200 mb geopotential': 'z', '200_mb_geopotential': 'z', '500 mb geopotential': 'z', '500_mb_geopotential': 'z',
    '200 mb total specific humidity': 'q_tot', '200_mb_total_specific_humidity': 'q_tot', 
    '500 mb total specific humidity': 'q_tot', '500_mb_total_specific_humidity': 'q_tot',
    'temperature': 'tair',
    'pressure': 'sp',
    'dewpoint': 'd2m',
    'precipitation': 'tp', 
    'precipitation_7day': 'tp_7d', '7 day precipitation': 'tp_7d',
    'precipitation_14day': 'tp_14d', '14 day precipitation': 'tp_14d',
    'precipitation_30day': 'tp_30d', '30 day precipitation': 'tp_30d',
    'evaporation': 'e',
    'potential evaporation': 'pev', 'potential_evaporation': 'pev',
    'evi': 'evi',
    'ndvi': 'ndvi',
    'lai': 'lai',
    'fpar': 'fpar',
    'high vegetation coverage': 'cvh', 'high_vegetation_coverage': 'cvh',
    'low vegetation coverage': 'cvl', 'low_vegetation_coverage': 'cvl',
    'radiation': 'ssr',
    'wind speed': 'ws', 'wind_speed': 'ws',
    'wind gusts': 'fg10', 'wind_gusts': 'fg10',
    'soil moisture 1': 'swvl1', 'soil_moisture_1': 'swvl1',
    'soil moisture 2': 'swvl2', 'soil_moisture_2': 'swvl2', 
    'soil moisture 3': 'swvl3', 'soil_moisture_3': 'swvl3',
    'soil moisture 4': 'swvl4', 'soil_moisture_4': 'swvl4',
    'fdii1': 'fdii1', 
    'fdii2': 'fdii2', 
    'fdii3': 'fdii3', 
    'fdii4': 'fdii4',
    'sesr': 'sesr'
}

# Define the path for each rotation
path_to_rotation = {
    0: '/ourdisk/hpc/ai2es/sedris/results/rotation_00/',#conv_results/',#rotation_00/',
    1: '/ourdisk/hpc/ai2es/sedris/results/rotation_01/',
    2: '/ourdisk/hpc/ai2es/sedris/results/rotation_02/',
    3: '/ourdisk/hpc/ai2es/sedris/results/rotation_03/',
    4: '/ourdisk/hpc/ai2es/sedris/results/rotation_04/',
    5: '/ourdisk/hpc/ai2es/sedris/results/rotation_05/',
    6: '/ourdisk/hpc/ai2es/sedris/results/rotation_06/',
    7: '/ourdisk/hpc/ai2es/sedris/results/rotation_07/',
    8: '/ourdisk/hpc/ai2es/sedris/results/rotation_08/',
    9: '/ourdisk/hpc/ai2es/sedris/results/rotation_09/',
    10: '/ourdisk/hpc/ai2es/sedris/results/rotation_10/',
    11: '/ourdisk/hpc/ai2es/sedris/results/rotation_11/',
    'single_run': '/ourdisk/hpc/ai2es/sedris/results/upsampling/'# '/ourdisk/hpc/ai2es/sedris/results/one_experiment_run/'
}

# Test years to load for each corresponding rotation
test_years = {
    0: [2023, 2024],
    1: [2001, 2002],
    2: [2003, 2004],
    3: [2005, 2006],
    4: [2007, 2008],
    5: [2009, 2010],
    6: [2011, 2012],
    7: [2013, 2014],
    8: [2015, 2016],
    9: [2017, 2018],
    10: [2019, 2020],
    11: [2021, 2022],
    'single_run': [2020, 2021]
}

def make_climatology_and_persistence_metrics(args, timestamp) -> None:
    '''
    Calculate performance metrics for climatology and persistence for one day/set of predictions
    and save them to .nc files

    Inputs:
    :param args: Dictionary of parser arguments specified at the terminal. 
                 --subset, --subset, --apply_mask, --data_path, --prediction_path and --single_experiment_run are key arguments for this analysis
    :param timestamp: Datetime describing the day the predictions are for
    '''

    # Obtain the path to a set of performance metric data that is known to exist
    path = path_to_rotation[0]
    forecast_length = 90 # Preset forecast length

    # For increased generality, determine the rotation from the timestamp
    for key in test_years.keys():
        if timestamp.year in test_years[key]:
            rot = key

    # Construct the filename for the climatology and persistence datasets
    base = '%04d-%02d-%02dT00Z.nc'%(timestamp.year, timestamp.month, timestamp.day)
    clim_filename = 'clim_%s'%base if args.subset is None else '%s_clim_%s'%base
    persist_filename = 'persist_%s'%base if args.subset is None else '%s_persist_%s'%base
    print(clim_filename)

    if (os.path.exists('%s/climatology/rotation_%02d/%s'%(args.prediction_path, rot, clim_filename)) & 
        os.path.exists('%s/persistence/rotation_%02d/%s'%(args.prediction_path, rot, persist_filename))):
        # If the files exist already, then there is no need to run this again
        return

    # Initial list metrics; load in a test set of metrics that is known to exist
    metrics_initial = pd.read_csv('%s/forecasts/metrics/%04d-01-01T00Z.csv'%(path, test_years[0][0]), sep = ',',
                                  header = 0, index_col = 0, nrows = forecast_length) # Read 90 rows for all forecasts 

    fh = metrics_initial['forecast_step']

    # Initialize the dataset
    climatology_metrics = {}
    persistence_metrics = {}
    print(timestamp)

    # Perform computations for each metric and variable
    for metric in metrics_initial.columns:
        if metric in skip_variables: # These include time and forecast_step, which is what the metrics are plotted against
            continue
        
        # Parse the metric into its individual parts
        metric_name, var_name, level = get_metric_information(metric)

        if (var_name == 'Overall') & (args.subset is not None):
            continue

        # print(metric, var_name, metric_name)
        # print('Working on ' + var_name + ' climatology and persistence predictions')# for: ' + (datetime(args.year, args.month, 1)+timedelta(days=day)).isoformat())

        # For overall variables, skip the climatology and persistence calculations (no variable to compare against)
        if var_name == 'Overall':
            climatology = None
            persistence = None

        # Climatology and persistence calculations
        else:
            # Obtain the type of data for the variable (i.e., which zarr file the true label is in)
            if var_name in diagnostic_variables:
                data_type = 'diagnostic'
            elif var_name in upper_air_variables:
                data_type = 'upper_air'
            else:
                data_type = 'surface'
            
            # Load in climatology and persistence predictions
            tmp_climatology = load_climatology(var_name, 
                                               timestamp.month, 
                                               timestamp.day, 
                                               Ndays = forecast_length, 
                                               level = level, 
                                               clim_year = timestamp.year,
                                               type = 'running', 
                                               path = args.data_path)
            tmp_persistence = load_persistence(var_name, 
                                               data_type, 
                                               timestamp.year, 
                                               timestamp.month, 
                                               timestamp.day, 
                                               Ndays = forecast_length, 
                                               level = level, 
                                               type = 'standard', 
                                               path = args.data_path)

            # Calculate the performance metric for climatology and persistence
            climatology = calculate_metric(tmp_climatology, 
                                           metric_name, 
                                           var_name, 
                                           timestamp.year, 
                                           timestamp.month, 
                                           timestamp.day, 
                                           data_type, 
                                           subset = args.subset,
                                           Ndays = forecast_length, 
                                           apply_mask = args.apply_mask,
                                           level = level, 
                                           path = args.data_path)
            persistence = calculate_metric(tmp_persistence, 
                                           metric_name, 
                                           var_name, 
                                           timestamp.year, 
                                           timestamp.month, 
                                           timestamp.day, 
                                           data_type, 
                                           subset = args.subset,
                                           Ndays = forecast_length, 
                                           apply_mask = args.apply_mask,
                                           level = level, 
                                           path = args.data_path)

            climatology = np.array(climatology)
            persistence = np.array(persistence)
            # climatology = np.stack(climatology, axis = 0)
            # climatology = np.nanmean(climatology, axis = 0)
            # persistence = np.stack(persistence, axis = 0)
            # persistence = np.nanmean(persistence, axis = 0)
        
        # Store the climatology and persistence metrics
        climatology_metrics[metric] = climatology
        persistence_metrics[metric] = persistence
        #print('Climatology: ', climatology)
        #print('Persistence: ', persistence)

    # Save the results to a .nc file
    with Dataset('%s/climatology/rotation_%02d/%s'%(args.prediction_path, rot, clim_filename), 'w') as nc:
        nc.createDimension('forecast_length', size = forecast_length)
        nc.createDimension('overall_length', size = 1)

        for key in climatology_metrics.keys():
            if len(key) < 5: # length of the key is less than 5 only when the metric_name is used
                             # (acc, rmse, mse, mae), i.e., for overall metrics, when var_name = None
                nc.createVariable(key, np.float64, ('overall_length', ))
                nc.variables[key][:] = climatology_metrics[key]
            else:
                # Create and write the metric information
                nc.createVariable(key, climatology_metrics[key].dtype, ('forecast_length', ))
                nc.variables[key][:] = climatology_metrics[key][:]

    with Dataset('%s/persistence/rotation_%02d/%s'%(args.prediction_path, rot, persist_filename), 'w') as nc:
        nc.createDimension('forecast_length', size = forecast_length)
        nc.createDimension('overall_length', size = 1)

        for key in persistence_metrics.keys():
            if len(key) < 5: # length of the key is less than 5 only when the metric_name is used
                             # (acc, rmse, mse, mae), i.e., for overall metrics, when var_name = None
                nc.createVariable(key, np.float64, ('overall_length', ))
                nc.variables[key][:] = persistence_metrics[key]
            else:
                # Create and write the metric information
                nc.createVariable(key, persistence_metrics[key].dtype, ('forecast_length', ))
                nc.variables[key][:] = persistence_metrics[key][:]


def make_subset_metrics(args, timestamp) -> None:
    '''
    Calculate performance metrics, same as what CREDIT performs, for individual subsetted regions for 
    one day/prediction set and save results to .csv files in the same format as the CREDIT files

    Inputs:
    :param args: Dictionary of parser arguments specified at the terminal. 
                 --subset, --apply_mask, --data_path and --single_experiment_run are key arguments for this analysis
    :param timestamp: Datetime describing the day the predictions are for
    '''

    # For increased generality, determine the rotation from the timestamp
    if args.single_experiment_run:
        rot = 'single_run'
    else:
        for key in test_years.keys():
            if timestamp.year in test_years[key]:
                rot = key
    
    # Obtain the path to a set of performance metric data that is known to exist
    path = path_to_rotation[0]
    forecast_length = 90 # Preset forecast length

    # Construct the filename for performance metrics
    subset_filename = '%04d-%02d-%02dT00Z_%s.csv'%(timestamp.year, timestamp.month, timestamp.day, args.subset)

    # Initial list metrics; load in a test set of metrics that is known to exist
    metrics_initial = pd.read_csv('%s/forecasts/metrics/%04d-01-01T00Z.csv'%(path, test_years[0][0]), sep = ',',
    # metrics_initial = pd.read_csv('%s/forecasts/metrics/2020-01-01T00Z.csv'%(path_to_rotation[rot]), sep = ',',
                                  header = 0, index_col = 0, nrows = forecast_length) # Read 90 rows for all forecasts 

    fh = metrics_initial['forecast_step']

    # Obtain the path to the performance metric data
    path = path_to_rotation[rot]

    if os.path.exists('%s/forecasts/metrics/%s'%(path, subset_filename)):
        # If the files exist already, then there is no need to run this again
        return
    
    # Collect all the file names for the predictions
    date_str = timestamp.strftime('%Y-%m-%dT%HZ')
    files = glob('%s/forecasts/%s/pred_*.nc'%(path, date_str), recursive = True)

    files_sorted = new_sort(files)

    # Loop through predictions files
    y_pred = {}
    for n, file in enumerate(files_sorted):
        # Load the predictions
        fd_indices = False
        with Dataset(file, 'r') as nc:
            time_pred = nc.variables['time'][:]
            time_pred = datetime(1900,1,1) + timedelta(hours = time_pred.item())

            # Load in data and subset it
            y_pred[n] = {}
            for var in nc.variables.keys():
                if var in skip_variables:
                    # Skip the description variables
                    continue
                else:
                    y_pred[n][var] = nc.variables[var][:].squeeze()
                    # if (var == 'sesr') | ('fdii' in var):
                    #     fd_indices = True

    # Convert the dictionary from y_pred[forecast_day][variable_and_metric] (shape lat x lon) to dict[variable_and_metric] (shape of forecast_day, lat, lon)
    y_subset_pred = {}
    for var in y_pred[0].keys():
        y_subset_pred[var] = []
        for fh in y_pred.keys():
            # Get the variable and add it to a list
            y_subset_pred[var].append(y_pred[fh][var])

        # Convert the list to an array, which will stack along axis = 0 for shape forecast_day x lat x lon
        y_subset_pred[var] = np.array(y_subset_pred[var])

    # For each metric and variable, calculate the performance of y_pred for a subsetted region
    metrics_subset = {}

    # Add fd indices to the columns if needed so that they are calculated
    if fd_indices is True:
        print('Adding FD columns')
        metrics_initial['acc_sesr'] = 0; metrics_initial['acc_fdii1'] = 0; metrics_initial['acc_fdii2'] = 0; metrics_initial['acc_fdii3'] = 0; metrics_initial['acc_fdii4'] = 0
        metrics_initial['rmse_sesr'] = 0; metrics_initial['rmse_fdii1'] = 0; metrics_initial['rmse_fdii2'] = 0; metrics_initial['rmse_fdii3'] = 0; metrics_initial['rmse_fdii4'] = 0
        metrics_initial['mse_sesr'] = 0; metrics_initial['mse_fdii1'] = 0; metrics_initial['mse_fdii2'] = 0; metrics_initial['mse_fdii3'] = 0; metrics_initial['mse_fdii4'] = 0
        metrics_initial['mae_sesr'] = 0; metrics_initial['mae_fdii1'] = 0; metrics_initial['mae_fdii2'] = 0; metrics_initial['mae_fdii3'] = 0; metrics_initial['mae_fdii4'] = 0

    # Perform calculations for each metric and variable
    for metric in metrics_initial.columns:
        if metric in skip_variables: # These include time and forecast_step, which is what the metrics are plotted against
            continue
        
        # Parse the metric into its individual parts
        metric_name, var_name, level = get_metric_information(metric)

        if (var_name == 'Overall'):
            continue

        # Obtain the type of data for the variable (i.e., which zarr file the true label is in)
        if var_name in diagnostic_variables:
            data_type = 'diagnostic'
        elif var_name in upper_air_variables:
            data_type = 'upper_air'
        else:
            data_type = 'surface'

        # Reduce to 3D for upper air variables
        if level is not None:
            tmp = y_subset_pred[var_name][:,level,:,:]
        else:
            tmp = y_subset_pred[var_name]

        # Perform the metric calculation based on the subsetted region
        metric_calculation = calculate_metric(tmp, 
                                              metric_name, 
                                              var_name,
                                              timestamp.year, 
                                              timestamp.month, 
                                              timestamp.day, 
                                              data_type, 
                                              subset = args.subset, 
                                              Ndays = forecast_length, 
                                              apply_mask = args.apply_mask,
                                              level = level, 
                                              path = args.data_path)
        
        metric_calculation = np.array(metric_calculation)

        # Add the metric to the final dictionary
        metrics_subset[metric] = metric_calculation

    # Convert the dictionary to a pandas df to write to the same csv format as the original metric files
    metrics_subset_df = pd.DataFrame(metrics_subset)

    # Write the metrics to csv files
    metrics_subset_df.to_csv('%s/forecasts/metrics/%s'%(path, subset_filename), index = True)
    
    return

def add_fd_indices(
        direct, 
        sm, 
        dates_all, 
        one_year, 
        esr_means, 
        esr_stds, 
        mask
        ) -> None:
    '''
    Calculate flash drought (FD) indices (SESR, and FDII at multiple soil depths) and append them
    to a .nc prediction file

    Inputs:
    :param direct: Directory path to where the .nc files are located
    :param sm: Directory of soil moisture datasets
               Each key (1, 2, ...) is a depth level of soil moisture, 
               and has a list (one for each year) of time x lat x lon arrays of 
               SM for the respective depth
    :param dates_all: List/array of datetimes for all valid dates in the full dataset (i.e., in the SM time series)
    :param one_year: List/array of datetimes for one complete year
    :param esr_means: Array of climatological means of the evaporative stress ratio 
                      for grid point and day of year (np.ndarray with shape time x lat x lon)
    :param esr_stds: Array of climatological standard deviations of the evaporative stress ratio 
                     for grid point and day of year (np.ndarray with shape time x lat x lon)
    :param mask: Land-sea mask (0 for sea grids and 1 for land grid points)
    '''

    # Collect all the .nc files
    nc_files = glob('%s/pred_*.nc'%direct, recursive = True)

    # Initialize datasets that will be loaded from new files
    data = {}
    data['e'] = []; data['pev'] = []; data['swvl1'] = []; data['swvl2'] = []; data['swvl3'] = []; data['swvl4'] = []
    dates = []
    fd_indices_calculated = []

    # Load each .nc prediction file
    for file in new_sort(nc_files):
        # Load the prediction data
        with Dataset(file, 'r') as nc:
            # Determine if the FD indices have already been calculated for the .nc file
            if (
                ('sesr' in nc.variables.keys()) & 
                ('fdii1' in nc.variables.keys()) & 
                ('fdii2' in nc.variables.keys()) & 
                ('fdii3' in nc.variables.keys()) & 
                ('fdii4' in nc.variables.keys())
                ):
                fd_indices_calculated.append(True)
            else:
                fd_indices_calculated.append(False)

            # Load FD related variables
            data['e'].append(nc.variables['e'][:])
            data['pev'].append(nc.variables['pev'][:])
            data['swvl1'].append(nc.variables['swvl1'][:])
            data['swvl2'].append(nc.variables['swvl2'][:])
            data['swvl3'].append(nc.variables['swvl3'][:])
            data['swvl4'].append(nc.variables['swvl4'][:])
            
            # Date information
            date_tmp = nc.variables['time'][:]
            dates.append(datetime(1900,1,1) + timedelta(hours = int(date_tmp)))

    # Skip if FD indices were already calculated for all files
    # if np.nansum(fd_indices_calculated) == len(nc_files):
    #     print('All FD indices already calculated; skipping %s'%direct)
    #     return

    # Concenate loaded data into a single set of arrays
    data['e'] = np.concatenate(data['e'])
    data['pev'] = np.concatenate(data['pev'])
    data['swvl1'] = np.concatenate(data['swvl1'])
    data['swvl2'] = np.concatenate(data['swvl2'])
    data['swvl3'] = np.concatenate(data['swvl3'])
    data['swvl4'] = np.concatenate(data['swvl4'])
    dates = np.array(dates)

    # Calculate SESR
    data['sesr'] = calculate_sesr(data['e'], data['pev'], dates, esr_means, esr_stds, one_year)

    # Calculate SM percentiles
    smp1 = calculate_sm_percentiles(data['swvl1'], sm[1], dates, dates_all, mask = mask)
    smp2 = calculate_sm_percentiles(data['swvl2'], sm[2], dates, dates_all, mask = mask)
    smp3 = calculate_sm_percentiles(data['swvl2'], sm[3], dates, dates_all, mask = mask)
    smp4 = calculate_sm_percentiles(data['swvl3'], sm[4], dates, dates_all, mask = mask)

    # Calculate FDII
    data['fdii1'], _, _ = calculate_fdii(smp1, dates, apply_runmean = True, mask = mask)
    data['fdii2'], _, _ = calculate_fdii(smp2, dates, apply_runmean = True, mask = mask)
    data['fdii3'], _, _ = calculate_fdii(smp3, dates, apply_runmean = True, mask = mask)
    data['fdii4'], _, _ = calculate_fdii(smp4, dates, apply_runmean = True, mask = mask)

    # Append the calculated FD variables
    for t, file in enumerate(new_sort(nc_files)):
        # If the FD indices for this .nc file have been made, skip this iteration
        # if fd_indices_calculated[t]:
        #     print('Continuing to next file')
        #     continue

        # Setup the data to be shape 1 x lat x lon, to mimic the variables in the original .nc files
        data_tmp = {}
        fd_indices = ['sesr', 'fdii1', 'fdii2', 'fdii3', 'fdii4']
        T, I, J = data['e'].shape
        for key in fd_indices:
            data_tmp[key] = data[key][np.newaxis,t,:,:].astype(np.float32)
            # data_tmp[key] = np.zeros((1, I, J))
            # data_tmp[key][0,:,:] = data[key][t,:,:]

        # Append the FD indices to the .nc file 
        with Dataset(file, 'a') as nc:
            for key in fd_indices:
                # Make the new variable
                if key not in nc.variables.keys():
                    nc.createVariable(key, data_tmp[key].dtype, ('time', 'latitude', 'longitude'))

                # If the variable already exists, this will overwrite existing data
                # (useful for making corrections if needed)
                nc.variables[key][:] = data_tmp[key][:]

            # nc.createVariable('sesr', data_tmp['sesr'].dtype, ('time', 'latitude', 'longitude'))
            # nc.createVariable('fdii1', data_tmp['fdii1'].dtype, ('time', 'latitude', 'longitude'))
            # nc.createVariable('fdii2', data_tmp['fdii2'].dtype, ('time', 'latitude', 'longitude'))
            # nc.createVariable('fdii3', data_tmp['fdii3'].dtype, ('time', 'latitude', 'longitude'))
            # nc.createVariable('fdii4', data_tmp['fdii4'].dtype, ('time', 'latitude', 'longitude'))

            # nc.variables['sesr'][:] = data_tmp['sesr'][:]
            # nc.variables['fdii1'][:] = data_tmp['fdii1'][:]
            # nc.variables['fdii2'][:] = data_tmp['fdii2'][:]
            # nc.variables['fdii3'][:] = data_tmp['fdii3'][:]
            # nc.variables['fdii4'][:] = data_tmp['fdii4'][:]


def make_fd_indices(args) -> None:
    '''
    Calculate and add flash drought (FD) indices (the standardized evaporative stress ratio [SESR], 
    and the flash drought intensity index [FDII] for all soil depths) to a set of model predictions
    (.nc files are appended with the indices)

    Inputs:
    :param args: Dictionary of parser arguments specified at the terminal. 
                 --nprocesses, --fd_task_number, and --single_experiment_run are key arguments for this analysis
    '''

    # Define a few variables
    path_to_sm = '/ourdisk/hpc/ai2es/sedris/credit_datasets'
    days_per_year = 365 # daily data; this number can be adjusted for different temporal resolutions

    # All years in the time series
    all_years = np.arange(2001, 2025)
    N_leap_days = np.sum((all_years % 4) == 0) # Note datetimes follow the scheme of a leap day every 4 years, even if it isn't exactly correct
    T_full = (days_per_year * all_years.size) + N_leap_days

    # Construct datetime array of dates in the dataset
    date_initial = datetime(all_years[0], 1, 1)
    dates_all = np.array([date_initial + timedelta(days = t) for t in range(T_full)])

    # Load a land-sea mask; this is used to skip sea points can save computation time and resources
    with Dataset('%s/aridity_mask_reduced.nc'%path_to_sm ,'r') as nc:
        mask = nc.variables['aim'][:]

    print('Loading SM datasets')
    sm = {}
    sm[1] = []; sm[2] = []; sm[3] = []; sm[4] = [] # Initializing sm for different depths
    e = []
    pet = []

    # Load a full soil moisture time series (required for percentile calculations)
    for y in all_years:
        print(y)
        root = zarr.open_group('%s/diagnostic.%04d.zarr'%(path_to_sm, y), mode = 'r')
        
        # Load SM at each depth as float32s to help conserve RAM usage
        sm[1].append(root['swvl1'][:].astype(np.float32))
        sm[2].append(root['swvl2'][:].astype(np.float32))
        sm[3].append(root['swvl3'][:].astype(np.float32)) 
        sm[4].append(root['swvl4'][:].astype(np.float32))

        # Load ET and PET to construct climatological means and standard deviations
        root_surf = zarr.open_group('%s/surface.%04d.zarr'%(path_to_sm, y), mode = 'r')
        e.append(root_surf['e'][:].astype(np.float32))
        pet.append(root_surf['pev'][:].astype(np.float32))
        # print('Size of each  is %f MB, totalling %f MB'%(sys.getsizeof(root['swvl1'][:].astype(np.float32))/1024/1024,
        #                                                  (sys.getsizeof(root['swvl1'][:].astype(np.float32)) + sys.getsizeof(root['swvl2'][:].astype(np.float32)) + sys.getsizeof(root['swvl3'][:].astype(np.float32)) + sys.getsizeof(root['swvl4'][:].astype(np.float32)) + sys.getsizeof(root_surf['e'][:].astype(np.float32)) + sys.getsizeof(root_surf['pev'][:].astype(np.float32)))/1024/1024))

    print('Concatenating')

    # Merge ET and PET into a single dataset each
    e = np.concatenate(e)
    pet = np.concatenate(pet)
    # print('Total size: %f MB'%((sys.getsizeof(sm[1]) + sys.getsizeof(sm[2]) + sys.getsizeof(sm[3]) + sys.getsizeof(sm[4]) + sys.getsizeof(e) + sys.getsizeof(pet))/1024/1024))
    
    # Make the means and standard deviations for ESR
    print('Making climatologies')
    esr_means, esr_stds, one_year = calculate_climatology(e, pet, dates_all, days_per_year+1)

    # Clear some of the larger datasets
    del e, pet
    gc.collect()
    print('Collecting directories')

    # Grab the .nc prediction filenames
    if np.invert(args.single_experiment_run):
        all_direct = []
        # Collect directories of nc files for all rotations
        for rot in rotations:
            path = path_to_rotation[rot]
            all_direct.append(new_sort(glob('%s/forecasts/*Z'%path, recursive = True)))
        
        # Concatenate into a single list of directories
        all_direct = np.concatenate(all_direct)
    else:
        # Collect the directories with predictions
        path = path_to_rotation[rotations[0]]
        all_direct = new_sort(glob('%s/forecasts/*Z'%path, recursive = True))

    # print(np.sort(all_direct))

    if args.nprocesses > 1:
        # Set parameters for multiprocessing
        param_args = [(direct, sm, dates_all, one_year, esr_means, esr_stds, mask) for direct in new_sort(all_direct)]
        # param_args = [(direct) for direct in np.sort(all_direct)]

        # Add FD indices to .nc files with nprocesses processes
        with mp.Pool(args.nprocesses) as pool:
            data = pool.starmap(add_fd_indices, param_args)
            # data = pool.map(add_fd_indices, param_args)
    else:
        # If only 1 process, add FD indices one file at a time

        # If no task number is specified, process all files via loop
        if args.fd_task_number == -1:
            for direct in new_sort(all_direct):
                add_fd_indices(direct, 
                               sm, 
                               dates_all, 
                               one_year, 
                               esr_means, 
                               esr_stds, 
                               mask)
        else:
            # If a task number is specified, divide the number of files to work on by 12
            # This is designed to work with SLURM's task ID system, to split the work
            # into twelve parts and allow multiprocessing without creating large strain
            # on the RAM (due to being split across several nodes instead of working on 1)

            # Divide the number directories to work on by 12
            Niter = 12
            Ntasks = len(all_direct)/Niter

            # Starting index of files to work on
            start_ind = int(Niter * args.fd_task_number)

            # Ending index of files to work on (note a special case for the last task)
            if (len(all_direct) - start_ind) < Niter:
                end_ind = len(all_direct)
            else:
                end_ind = int(Niter * (args.fd_task_number + 1))

            # Loop through directories for the specified task ID
            for direct in new_sort(all_direct)[start_ind:end_ind]:
                add_fd_indices(direct, 
                               sm, 
                               dates_all, 
                               one_year, 
                               esr_means, 
                               esr_stds, 
                               mask)

    return

def make_variation_plots(args) -> None:
    '''
    Load in true and predicted data an examine analyze the variability of the predicted data

    Function creates a set of violin plots of climate anomalies and RPC for predicted
    variables and predicted variable anomalies as a function of lead time

    Inputs:
    :param args: Dictionary of parser arguments specified at the terminal. 
                 --apply_mask, --rotations, and --single_experiment_run are key arguments for this analysis
    '''
     # Variables and subsets to plot
    variables_to_plot = ['tp', 'tp_30d', 'e', 'pev', 'ndvi', 'evi', 'lai', 'fpar', 'd2m', 'tair',# 'sp',
                            'swvl1', 'swvl2', 'swvl3', 'swvl4']
    subsets = ['africa', 'africa_nh', 'africa_sh', 'africa_tropics', 'nh', 'sh', 'tropics', 'conus']

    # Initialize datasets
    true_datasets = {}
    true_subset_data = {}
    true_subset_anom_data = {}
    pred_datasets = {}
    rpc = {}
    rpc_anom = {}

    climatology = {}

    # Load land-sea and aridity mask if required
    if args.apply_mask:
        with Dataset('%s/land_reduced.nc'%args.data_path ,'r') as nc:
            mask_lsm = nc.variables['lsm'][:]

        with Dataset('%s/aridity_mask_reduced.nc'%args.data_path ,'r') as nc:
            mask_aim = nc.variables['aim'][:]

        mask = np.where((mask_lsm == 1) & (mask_aim == 1), 1, 0)

    # Load climatological means (for determining anomalies)
    root = zarr.open_group('%s/climatology.zarr'%(args.data_path), mode = 'r')
    # Load all variables in the climatology set
    for var in variables_to_plot:
        climatology[var] = root[var][:]

        # Apply the mask if necessary 
        # (this will carry over to all remaining datasets since NaN operated with anything else yields a NaN)
        if args.apply_mask:
            for t in range(climatology[var].shape[0]):
                climatology[var][t,:,:] = np.where(mask == 1, climatology[var][t,:,:], np.nan)
        T, I, J = climatology[var].shape
    times = root['time'][:]

    # Initialize all the dictionary values to lists (one for each possible subset and variable)
    for sub in subsets:
        true_datasets[sub] = {}
        true_subset_data[sub] = {}
        true_subset_anom_data[sub] = {}
        pred_datasets[sub] = {}
        rpc[sub] = {}
        rpc_anom[sub] = {}

        for var in variables_to_plot:
            true_datasets[sub][var] = []
            true_subset_data[sub][var] = []
            true_subset_anom_data[sub][var] = []
            pred_datasets[sub][var] = []
            rpc[sub][var] = []
            rpc_anom[sub][var] = []
            # pred_datasets[sub][var] = np.ones((forecast_length, I, J)) * np.nan
    
    # Construct datetimes for one year (for extracting climatology dates)
    one_year = np.array([datetime.fromisoformat(date) for date in times])
    months_one_year = np.array([date.month for date in one_year])
    days_one_year = np.array([date.day for date in one_year])

    all_dates = []

    if args.single_experiment_run:
        years = np.arange(2020, 2022+1) # Add an extra year so further predictions into the next year are examined

    # Collect the true labels
    for y in years:
        print(y)
        # Open the zarr files
        root = zarr.open_group('%s/diagnostic.%04d.zarr'%(args.data_path, y), mode = 'r')
        root_surf = zarr.open_group('%s/surface.%04d.zarr'%(args.data_path, y), mode = 'r')

        # Collect lat and lon
        lat = root['latitude'][:,0]
        lon = root['longitude'][0,:]

        # Get datetime information to construct datetimes for all valid dates
        times = root['time'][:]
        dates = np.array([datetime.fromisoformat(date) for date in times])
        all_dates.append(dates)
        ind = []
        for date in dates:
            ind.append(np.where((date.month == months_one_year) & (date.day == days_one_year))[0][0])

        # Collect the data for each variables
        for var in variables_to_plot:
            variable = root[var][:] if var in diagnostic_variables else root_surf[var][:]

            # Determine the climate anomalies
            anom = variable - climatology[var][ind,:,:]

            # Add the data for each subset
            for sub in subsets:
                sub_data, _, _ = subset_data(anom, lat, lon, sub)
                sub_var, _, _ = subset_data(variable, lat, lon, sub)

                # Save the subset data in its gridded form for RPC calculations
                true_subset_data[sub][var].append(sub_var)
                true_subset_anom_data[sub][var].append(sub_data)

                # Spatial average to shape is (time)
                sub_data = np.nanmean(sub_data, axis = -1)
                sub_data = np.nanmean(sub_data, axis = -1)

                # Add to dataset
                true_datasets[sub][var].append(sub_data)
        
    # Construct the array of all valid datetimes
    all_dates = np.concatenate(all_dates)

    # Obtain the total number of datetimes in the rotation
    years_tmp = np.array([date.year for date in all_dates])
    ind_rotation = np.concatenate([np.where(year == years_tmp)[0] for year in years[:-1]])
    # print(all_dates[ind_rotation])

    # Concatenate the true datasets into arrays of time x lat x lon for each subset and variable
    for sub in subsets:
        for var in variables_to_plot:
            true_subset_data[sub][var] = np.concatenate(true_subset_data[sub][var])
            true_subset_anom_data[sub][var] = np.concatenate(true_subset_anom_data[sub][var])

    # Collect the directories of predictions to load
    if np.invert(args.single_experiment_run):
        all_direct = []
        # Collect directories of .nc files for all rotations in the single experiment run
        for rot in rotations:
            path = path_to_rotation[rot]
            all_direct.append(glob('%s/forecasts/*Z'%path, recursive = True))
        
        # Concatenate into a single list of directories
        all_direct = np.concatenate(all_direct)
    else:
        # Collect the directories with predictions
        path = path_to_rotation[rotations[0]]
        all_direct = glob('%s/forecasts/*Z'%path, recursive = True)
        
    # Collect the prediction datasets
    for direct in tqdm(new_sort(all_direct)):
        # Collect all the .nc files in a directory
        nc_files = glob('%s/pred_*.nc'%direct, recursive = True)

        # Initialize temperary dictionaries that will store information for each forecast step
        # This is done so that the final loaded data will have shape time x forecast_length rather than 
        # stack everything to a time*forecast_length shape
        tmp_variable = {}
        tmp_rpc = {}
        tmp_rpc_anom = {}
        for sub in subsets:
            tmp_variable[sub] = {}
            tmp_rpc[sub] = {}
            tmp_rpc_anom[sub] = {}
            for var in variables_to_plot:
                tmp_variable[sub][var] = []
                tmp_rpc[sub][var] = []
                tmp_rpc_anom[sub][var] = []

        # Load each .nc prediction file
        for file in new_sort(nc_files):
            # Load the prediction data
            with Dataset(file, 'r') as nc:
                lat = nc.variables['latitude'][:]
                lon = nc.variables['longitude'][:]

                # Load the forecast hour/lead time and convert to an index
                fh = nc.variables['forecast_hour'][:]
                fh = int(fh/24 - 1)

                # Date information for the currently loaded forecast
                date_tmp = nc.variables['time'][:]
                date = datetime(1900,1,1) + timedelta(hours = int(date_tmp))

                # Find the index for the corresponding climatology date
                ind = np.where((date.month == months_one_year) & (date.day == days_one_year))[0]

                # Find the index in the true labels that correspond to the current prediction
                ind_obs = np.where(date == all_dates)[0]
                # print(ind_obs)

                for var in variables_to_plot:
                    variable = nc.variables[var][0,:,:]

                    anom = variable - climatology[var][ind[0],:,:]
                    # for sub in subsets:
                    #     tmp = np.array([pred_datasets[sub][var][fh,:,:], anom])
                    #     pred_datasets[sub][var][fh,:,:] = np.nansum(tmp, axis = 0)
                    for sub in subsets:
                        # Subset the current anomalies and original dataset
                        sub_data, _, _ = subset_data(anom, lat, lon, sub)
                        sub_var, _, _ = subset_data(variable, lat, lon, sub)
                        # print(true_subset_data[sub][var].shape)
                        
                        # Calculate the Ratio of Principal Components (RPC) to compare variation 
                        # between original variables and anomalies
                        rpc_tmp = calculate_rpc(true_subset_data[sub][var][ind_obs[0],:,:], sub_var)
                        rpc_anom_tmp = calculate_rpc(true_subset_anom_data[sub][var][ind_obs[0],:,:], sub_data)

                        # Save the RPC information
                        tmp_rpc[sub][var].append(rpc_tmp)
                        tmp_rpc_anom[sub][var].append(rpc_anom_tmp)

                        # Spatial average anomalies to obtain shape forecast_length (after the loop is finished)
                        sub_data = np.nanmean(sub_data, axis = -1)
                        sub_data = np.nanmean(sub_data, axis = -1)

                        # Add to dataset
                        tmp_variable[sub][var].append(sub_data)

        # Convert the lists to arrays (shape forecast_length) and add them to the final datasets
        for sub in subsets:
            for var in variables_to_plot:
                tmp = np.array(tmp_variable[sub][var])
                rpc_tmp = np.array(tmp_rpc[sub][var])
                rpc_anom_tmp = np.array(tmp_rpc_anom[sub][var])
                pred_datasets[sub][var].append(tmp)#np.nanmean(tmp, axis = 0))
                rpc[sub][var].append(rpc_tmp)
                rpc_anom[sub][var].append(rpc_anom_tmp)

    # For each variable subset, make a trio of figures to examine model variability performance
    for var in variables_to_plot:
        for sub in subsets:
            # Ensure all datasets are arrays for plotting
            true_datasets[sub][var] = np.concatenate(true_datasets[sub][var])[ind_rotation]
            pred_datasets[sub][var] = np.array(pred_datasets[sub][var])
            rpc[sub][var] = np.array(rpc[sub][var])
            rpc_anom[sub][var] = np.array(rpc_anom[sub][var])
            # pred_datasets[sub][var] = pred_datasets[sub][var]/len(all_direct)
            # pred_datasets[sub][var], _, _ = subset_data(pred_datasets[sub][var], lat, lon, sub)

            # Nfor, I, J = pred_datasets[sub][var].shape

            # true_datasets[sub][var] = np.nanmean(true_datasets[sub][var], axis = 0).flatten()
            # pred_datasets[sub][var] = pred_datasets[sub][var].reshape(Nfor, I*J)

            # Ensure all datasets are the correct shape
            print(var, sub)
            print(true_datasets[sub][var].shape)
            print(pred_datasets[sub][var].shape)
            print(rpc[sub][var].shape)
            print(rpc_anom[sub][var].shape)
            
            # Make savename, and violin plots plots
            savename = 'violin_plot_%s_%s.png'%(sub, var)
            plot_violin(true_datasets[sub][var], pred_datasets[sub][var], 
                        var, path = args.figure_path, savename = savename)

            # Make the RPC plots for the original variables
            savename = 'rpc_plot_%s_%s.png'%(sub, var)
            plot_metric(rpc[sub][var], 
                        np.arange(1, rpc[sub][var].shape[1] + 1), 
                        'rpc', 
                        var, 
                        climatology = None, 
                        persistence = None,
                        month = None, 
                        year = None, 
                        add_variation = True,
                        path = args.figure_path, 
                        savename = savename)
            
            # Make the RPC plots for the anomalies
            savename = 'rpc_plot_%s_%s_anom.png'%(sub, var)
            plot_metric(rpc_anom[sub][var], 
                        np.arange(1, rpc[sub][var].shape[1] + 1), 
                        'rpc', 
                        var, 
                        climatology = None, 
                        persistence = None,
                        month = None, 
                        year = None, 
                        add_variation = True,
                        path = args.figure_path, 
                        savename = savename)


def make_case_study_plots(
        args, 
        variable, 
        case_study_start, 
        case_study_end, 
        pred_path, 
        verbose = True
        ) -> None:
    '''
    Create a set of figures for a case study for each forecast step in a model prediction. Turn figures into a .gif video

    Inputs:
    :param args: Dictionary of parser arguments defined in the terminal
                 For this function, important arguments are --variable, --subset --apply_mask, --data_path, and --figure_path
    :param variable: Name of the variable the case study is made for (must be a key in short_names)
    :param case_study_start: Datetime of the starting date in the case study
    :param case_study_end: Datetime of the ending date of the case study
    :param pred_path: Directory path pointing to forecasts directory (where the predictions are stored)
    :param verbose: Boolean; Output progress at various points in the function
    '''

    # Determine the short name (name used in zarr and nc files) of the variable
    sname = short_names[variable.lower()]

    # Clarify plotting name if upper air variables are used
    if (('200 mb' in args.variable) | ('200_mb' in args.variable)) & ('u' in args.variable.lower()):
        plot_sname = '200 mb ' + sname
    elif (('500 mb' in args.variable) | ('500_mb' in args.variable)) & ('u' in args.variable.lower()):
        plot_sname = '500 mb ' + sname
    elif (('200 mb' in args.variable) | ('200_mb' in args.variable)) & ('v' in args.variable.lower()):
        plot_sname = '200 mb ' + sname
    elif (('500 mb' in args.variable) | ('500_mb' in args.variable)) & ('v' in args.variable.lower()):
        plot_sname = '500 mb ' + sname
    elif (('200 mb' in args.variable) | ('200_mb' in args.variable)) & ('geopotential' in args.variable.lower()):
        plot_sname = '200 mb ' + sname
    elif (('500 mb' in args.variable) | ('500_mb' in args.variable)) & ('geopotential' in args.variable.lower()):
        plot_sname = '500 mb ' + sname
    elif (('200 mb' in args.variable) | ('200_mb' in args.variable)) & ('humidity' in args.variable.lower()):
        plot_sname = '200 mb ' + sname
    elif (('500 mb' in args.variable) | ('500_mb' in args.variable)) & ('humidity' in args.variable.lower()):
        plot_sname = '500 mb ' + sname
    else:
        plot_sname = sname

    # Collect type of true labels
    if sname in diagnostic_variables:
        data_type = 'diagnostic'
    elif sname in upper_air_variables:
        data_type = 'upper_air'
    else:
        data_type = 'surface'

    # Load the true labels
    root = zarr.open('%s/%s.%04d.zarr/'%(args.data_path, data_type, case_study_start.year))
    y_true = root[sname][:]
    lat = root['latitude'][:]
    lon = root['longitude'][:]
    times = root['time'][:]
    times = pd.to_datetime(times).to_numpy(dtype = datetime)

    # In the event the case study extends into the following year
    if case_study_end.year > case_study_start.year:
        # If the predictions extend into the next year, load the next year of data to use for predictions
        root_new = zarr.open_group('%s/%s.%04d.zarr/'%(args.data_path, data_type, case_study_end.year))
        y_true_new = root_new[sname][:]
        times_new = root_new['time'][:]
        times_new = pd.to_datetime(times_new).to_numpy(dtype = datetime)
        
        # Concatenate the new year of data onto the current year
        y_true = np.concatenate([y_true, y_true_new])
        times = np.concatenate([times, times_new])

    # Collect climatology means
    clim_root = zarr.open('%s/climatology.zarr/'%args.data_path)
    climatology = clim_root[sname][:]

    # Select the level for upper air variables
    if ('200 mb' in variable) | ('200_mb' in variable):
        y_true = y_true[:,0,:,:]
        climatology = climatology[:,0,:,:]
    elif ('500 mb' in variable) | ('500_mb' in variable):
        y_true = y_true[:,1,:,:]
        climatology = climatology[:,1,:,:]

    # Reverse the lattitude (so the maps appears the right way up)
    lat = lat[::-1, :]
    
    # Collect the prediction files for the case study
    files = glob('%s/forecasts/%04d-%02d-%02dT00Z/pred_*.nc'%(pred_path, 
                                                              case_study_start.year, 
                                                              case_study_start.month, 
                                                              case_study_start.day), recursive = True)
    
    # Make the error in space maps (i.e., maps of RMSE and ACC)
    if verbose:  
        print('Making error map')
    make_metric_error_plots(y_true, 
                            files, 
                            times, 
                            lat, 
                            lon, 
                            sname, 
                            plot_sname, 
                            args.subset, 
                            variable, 
                            args.apply_mask, 
                            args.data_path, 
                            args.figure_path)

    # Loop through predictions
    for file in new_sort(files):
        # Load model predictions
        with Dataset(file, 'r') as nc:
            fh = nc.variables['forecast_hour'][:]
            time_pred = nc.variables['time'][:]
            time_pred = datetime(1900,1,1) + timedelta(hours = time_pred.item())
            # Select upper air level if necessary
            if ('200 mb' in args.variable) | ('200_mb' in args.variable):
                y_pred = nc.variables[sname][0,0,:,:]
            elif ('500 mb' in args.variable) | ('500_mb' in args.variable):
                y_pred = nc.variables[sname][0,1,:,:]
            else:
                y_pred = nc.variables[sname][0,:,:]

        # Find the corresponding true label
        time_ind = np.where(time_pred == times)[0]
        y = y_true[time_ind[0],:,:]
        if time_ind[0] > 366: # For some case studies, this may be Jan - March of the next year,
                                # so the index will need to be reduced 365/366 to fit in the valid 
                                # climatology indices
            time_ind[0] = time_ind[0] - 366 if (case_study_start.year % 4) == 0 else time_ind[0] - 365

        clim_pred = climatology[time_ind[0],:,:]

        if verbose:
            print('Making figure for', time_pred.isoformat())
        # Make case study plots for the globe
        if args.subset is not None:
            # Make the maps of the variables
            make_comparison_subset_map(y, 
                                       y_pred, 
                                       clim_pred, 
                                       lat, 
                                       lon, 
                                       time_pred, 
                                       args.subset,
                                       var_name = plot_sname, 
                                       forecast_hour = fh, 
                                       path = args.figure_path, 
                                       savename = '%s_%s_forecast_hour_%04d.png'%(args.subset, variable, fh))
            figure_fns = '%s_%s_forecast_hour_*.png'%(args.subset, variable)

            # Determine climate anomalies
            y_anom = y - clim_pred
            y_pred_anom = y_pred - clim_pred

            # Make the climate anomaly maps
            make_anomaly_subset_map(y_anom, 
                                    y_pred_anom, 
                                    lat, 
                                    lon, 
                                    time_pred, 
                                    args.subset,
                                    var_name = plot_sname, 
                                    forecast_hour = fh,
                                    path = args.figure_path, 
                                    savename = '%s_%s_anomaly_forecast_hour_%04d.png'%(args.subset, variable, fh))
            anomaly_figure_fns = '%s_%s_anomaly_forecast_hour_*.png'%(args.subset, variable)

            # Apply masks if necessary (removes redundant points from scatterplot and histogram calculations)
            if args.apply_mask:
                # with Dataset('%s/aridity_mask.nc'%args.data_path, 'r') as nc:
                with Dataset('%s/aridity_mask_reduced.nc'%args.data_path, 'r') as nc:
                    # ai_mask = nc.variables['aim'][:,:-1,:] # Note the aridity mask also masks out the oceans
                    ai_mask = nc.variables['aim'][:]
                
                # with Dataset('%s/land.nc'%args.data_path, 'r') as nc:
                with Dataset('%s/land_reduced.nc'%args.data_path, 'r') as nc:
                    # lsm_mask = nc.variables['lsm'][:,:-1,:] # Includes other sea points not in the aridity mask (e.g., Great Lakes)
                    lsm_mask = nc.variables['lsm'][:]

                # Apply the masks
                y_pred = np.where(ai_mask == 1, y_pred, np.nan)
                y_pred = np.where(lsm_mask == 1, y_pred, np.nan)

                y = np.where(ai_mask == 1, y, np.nan)
                y = np.where(lsm_mask == 1, y, np.nan)

            # Subset data
            y_pred, _, _ = subset_data(y_pred, lat[:,0], lon[0,:], args.subset)
            y, _, _ = subset_data(y, lat[:,0], lon[0,:], args.subset)

            # Make histograms and scatterplots
            make_histogram_scatter_plot(y, 
                                        y_pred, 
                                        plot_sname, 
                                        fh/24, #max_counts = 60,
                                        path = args.figure_path, 
                                        savename = '%s_%s_2d_histogram_forecast_hour_%04d.png'%(args.subset, variable, fh))
            histogram_figure_fns = '%s_%s_2d_histogram_forecast_hour_*.png'%(args.subset, variable)

        # Make case study plots for a specified subset region
        else:
            # Make the maps of the variables
            make_comparison_map(y, 
                                y_pred, 
                                clim_pred, 
                                lat, 
                                lon, 
                                time_pred,
                                var_name = plot_sname, 
                                forecast_hour = fh,
                                path = args.figure_path, 
                                savename = '%s_forecast_hour_%04d.png'%(variable, fh))
            figure_fns = '%s_forecast_hour_*.png'%variable

            # Determine climate anomalies
            y_anom = y - clim_pred
            y_pred_anom = y_pred - clim_pred

            # Make the climate anomaly maps
            make_anomaly_map(y_anom, 
                             y_pred_anom, 
                             lat, 
                             lon, 
                             time_pred,
                             var_name = plot_sname, 
                             forecast_hour = fh,
                             path = args.figure_path, 
                             savename = '%s_anomaly_forecast_hour_%04d.png'%(variable, fh))
            anomaly_figure_fns = '%s_anomaly_forecast_hour_*.png'%(variable)
            
            # Apply masks if necessary (removes redundant points from scatterplot and histogram calculations)
            if args.apply_mask:
                # with Dataset('%s/aridity_mask.nc'%args.data_path, 'r') as nc:
                with Dataset('%s/aridity_mask_reduced.nc'%args.data_path, 'r') as nc:
                    # ai_mask = nc.variables['aim'][:,:-1,:] # Note the aridity mask also masks out the oceans
                    ai_mask = nc.variables['aim'][:]
                
                # with Dataset('%s/land.nc'%args.data_path, 'r') as nc:
                with Dataset('%s/land_reduced.nc'%args.data_path, 'r') as nc:
                    # lsm_mask = nc.variables['lsm'][:,:-1,:] # Includes other sea points not in the aridity mask (e.g., Great Lakes)
                    lsm_mask = nc.variables['lsm'][:]

                # Apply the masks
                y_pred = np.where(ai_mask == 1, y_pred, np.nan)
                y_pred = np.where(lsm_mask == 1, y_pred, np.nan)

                y = np.where(ai_mask == 1, y, np.nan)
                y = np.where(lsm_mask == 1, y, np.nan)
            
            # Make histograms and scatterplot
            make_histogram_scatter_plot(y, 
                                        y_pred, 
                                        plot_sname, 
                                        fh/24, #max_counts = 300,
                                        path = args.figure_path, 
                                        savename = '%s_2d_histogram_forecast_hour_%04d.png'%(variable, fh))
            histogram_figure_fns = '%s_2d_histogram_forecast_hour_*.png'%(variable)
        
    #  Collect all the newly created figures filenames and turm them into a sorted list
    images = []
    figures = glob('%s/%s'%(args.figure_path, figure_fns), recursive = True)
    figures = np.sort(figures)
    for figure in figures:
        images.append(imageio.imread(figure))

    histogram_images = []
    figures = glob('%s/%s'%(args.figure_path, histogram_figure_fns), recursive = True)
    figures = np.sort(figures)
    for figure in figures:
        histogram_images.append(imageio.imread(figure))

    anomaly_images = []
    figures = glob('%s/%s'%(args.figure_path, anomaly_figure_fns), recursive = True)
    figures = np.sort(figures)
    for figure in figures:
        anomaly_images.append(imageio.imread(figure))

    # Convert the list of figures into gifs
    sname = '%s_gif.mp4'%variable if args.subset is None else '%s_%s_gif.mp4'%(args.subset, variable)
    imageio.mimsave('%s/%s'%(args.figure_path, sname), images, format = 'FFMPEG')

    sname = '%s_histogram_gif.mp4'%variable if args.subset is None else '%s_%s_histogram_gif.mp4'%(args.subset, variable)
    imageio.mimsave('%s/%s'%(args.figure_path, sname), histogram_images, format = 'FFMPEG')

    sname = '%s_anomaliy_gif.mp4'%variable if args.subset is None else '%s_%s_anomaly_gif.mp4'%(args.subset, variable)
    imageio.mimsave('%s/%s'%(args.figure_path, sname), anomaly_images, format = 'FFMPEG')



if __name__ == '__main__':
    # Create a parser to parse and process command line inputs
    description = 'Display forecast results from a CREDIT model'
    parser = ArgumentParser(description=description, fromfile_prefix_chars='@')
    parser.add_argument('--rotations', type = int, default = -1, help = 'Number of rotations to use in the analysis. -1 gets all rotations')
    parser.add_argument('--single_experiment_run', action = 'store_true', help = 'Makle results based on a single rotation/experiment with test years of 2020 and 2021')
    parser.add_argument('-var', '--variable', type = str, default = 'soil moisture 1', help = 'Full variable name of the variable being plotted in the case study')
    parser.add_argument('--subset', type = str, default = 'none', help = 'Subset the data to a given region before calculating metrics')
    parser.add_argument('--apply_mask', action = 'store_true', help = 'Apply land-sea and aridity masks to certain variables')
    parser.add_argument('--case_study_start_date', type = str, default = '2020-03-01', help = 'Start date of the case study to examine (must be in a format pd.to_datetime can read)')

    # Pathing arguments
    parser.add_argument('--prediction_path', type = str, default = '/scratch/rarrell/credit_model', help = 'Path to credit predictions, where the forecasts directory is')
    parser.add_argument('--data_path', type = str, default = '/ourdisk/hpc/ai2es/sedris/credit_datasets', help = 'Path to true labels and climatology dataset')
    parser.add_argument('--figure_path', type = str, default = '/ourdisk/hpc/ai2es/sedris/scripts/figures', help = 'path to where the created figures are stored')

    # SLURM and other multiprocessing arguments
    parser.add_argument('--nprocesses', type = int, default = 1, help = 'Number of multiprocesses to use when calculating metrics for subsets')
    parser.add_argument('--calculate_fd_indices', action = 'store_true', help = 'Calculate FD indices (SESR and FDII) with ET, PET, and SM predictions and replace old forecasts with them (may be time consuming)')
    parser.add_argument('--fd_task_number', type = int, default = -1, help = 'ID number indicating which files to loop through in FD index calculation. -1 = loop through all files')

    # Type of analyses to conduct arguments
    parser.add_argument('--make_clim_metrics', action = 'store_true', help = 'Make climatology and persistence metric calculations and save this results (note this is very time consuming)')
    parser.add_argument('--make_subset_metrics', action = 'store_true', help = 'Make metric calculations on a subset of the globe and save this results (note this is very time consuming)')
    parser.add_argument('--make_performance_plots', action = 'store_true', help = 'Make the performance plots (performance metrics vs forecast length)')
    parser.add_argument('--make_variation_plots', action = 'store_true', help = 'Make the plots comparing variation of data/data anomalies')
    parser.add_argument('--make_score_cards', action = 'store_true', help = 'Make the score cards')
    parser.add_argument('--make_case_studies', action = 'store_true', help = 'Make case study plots for multiple variables (metric distribution histograms, spatial distribution maps, model prediction maps and gifs)')

    # Parse the arguments
    args = parser.parse_args()

    if args.single_experiment_run:
        # Test years in the single experiment run.
        years = np.array([2020, 2021])

        # Rotation for a single experiment
        rotations = ['single_run']
    else:

        # Get all the rotations to use in the analysis
        if args.rotations == -1:
            rotations = np.arange(len(path_to_rotation))
        else:
            rotations = np.arange(args.rotations)

        # Determine all the years used in defined rotations
        years = []
        for rot in rotations:
            year_list = test_years[rot]
            [years.append(year) for year in year_list]

        years = np.sort(years)

    # Preset the forecast length
    forecast_length = 90

    # Determine if subset is a known subset; if not set to ignore it
    if args.subset.lower() not in subsets:
        args.subset = None

    # Get the short name of the variable
    sname = short_names[args.variable.lower()]

    # Make all datetimes in the selected rotations
    rotation_dates = []
    for year in years:
        # Start date in the year
        start_tmp = datetime(year, 1, 1)
        end_tmp = datetime(year, 4, 4) if year == 2024 else datetime(year, 12, 31) # Special case for last year in the dataset
        N_days = (end_tmp - start_tmp).days+1
        
        # Construct the array of datetimes for given year
        dates_tmp = np.array([start_tmp + timedelta(days = day) for day in range(N_days)])
        rotation_dates.append(dates_tmp)

    
    # Concatenate all years into a single array for all dates in the rotations
    rotation_dates = np.concatenate(rotation_dates)
    print(rotation_dates)

    # Collect all months in the used rotations
    months = np.array([date.month for date in rotation_dates])

    # Get all the datetimes available in the full dataset will be done for
    start_date = datetime(2001, 1, 1); end_date = datetime(2024, 12, 31) - timedelta(days = forecast_length+1)
    N_days = (end_date - start_date).days+1
    
    # Construct the array of datetimes
    all_dates = np.array([start_date + timedelta(days = day) for day in range(N_days)])

    # Make FD indices if requested
    if args.calculate_fd_indices:
        make_fd_indices(args)

    # Make climatology and persistence metrics if requested
    if args.make_clim_metrics:
        
        print('Calculating performance metrics for climatology and persistence')
        if args.nprocesses > 1:
            # Use multiprocessing to make climatology and persistence metrics for multiple timestamps at a time

            # Set the argument parameters
            param_args = [(args, timestamp) for timestamp in all_dates[:-1]]

            # Make the climatology and persistence metrics via multiprocessing
            with mp.Pool(args.nprocesses) as pool:
                data = pool.starmap(make_climatology_and_persistence_metrics, param_args)
        else:
            # Make the climatology and persistence metrics one timestamp at a time
            for timestamp in tqdm(all_dates):
                make_climatology_and_persistence_metrics(args, timestamp)

    # Make performance metrics over a specific region if requested
    if args.make_subset_metrics:
        # Make sure there is a designated subset to run this
        if args.subset is None:
            assert "--make_subset_metrics cannot be run when --subset is None"

        print('Calculating performance metrics for a subset region')
        if args.nprocesses > 1:
            # Make the subset metrics for multiple timestamps simultaneously via multiprocessing

            # Set the function parameters
            param_args = [(args, timestamp) for timestamp in rotation_dates]

            # Use multiprocessing to make multiple subset metrics
            with mp.Pool(args.nprocesses) as pool:
                    data = pool.starmap(make_subset_metrics, param_args)
        else:
            # Make the subset metrics one timestamp at a time
            for timestamp in tqdm(rotation_dates):
                make_subset_metrics(args, timestamp)


    # Initial list of metrics metrics; load in a test set of metrics that is known to exist
    metrics_initial = pd.read_csv('%s/forecasts/metrics/%04d-01-01T00Z.csv'%(path_to_rotation[0], test_years[0][0]), sep = ',',
                                  header = 0, index_col = 0, nrows = forecast_length) # Read 90 rows for all forecasts 
    
    fh = metrics_initial['forecast_step'] # Collect the forecast hours/days (used as x axis in some plots)

    # Load performance calculations in time and forecast day if any of the plots using them will be made
    if args.make_performance_plots | args.make_score_cards:
        metric_list, clim_list, persist_list = load_metrics(rotations, 
                                                            all_dates, 
                                                            test_years, 
                                                            subset = args.subset, 
                                                            forecast_length = forecast_length, 
                                                            path_to_data = args.prediction_path, 
                                                            paths_to_rotation = path_to_rotation)

    # Make performance plots if requested
    if args.make_performance_plots:

        # Make the plots for each variable and performance metric
        for metric in metrics_initial.columns:

            if metric in skip_variables: # These include time and forecast_step, which is what the metrics are plotted against
                continue
            
            # Parse the metric information into its individual parts
            metric_name, var_name, level = get_metric_information(metric)

            # Clarify plotting name if multiple upper air variables are used
            if level == 0:
                plot_sname = '200 mb ' + var_name
            elif level == 1:
                plot_sname = '500 mb ' + var_name
            else:
                plot_sname = var_name

            if (var_name == 'Overall'):# & (args.subset is not None):
                continue

            # Average over all predictions
            # metric_plot = np.nanmean(metric_list[metric], axis = 0)
            # clim_plot = np.nanmean(clim_list[metric], axis = 0)
            # persist_plot = np.nanmean(persist_list[metric], axis = 0)

            # First plot is over all months
            metric_plot = metric_list[metric]
            clim_plot = clim_list[metric]
            persist_plot = persist_list[metric]

            # Make the first set of performance plots
            savename = '%s_in_forecast_hour.png'%metric if args.subset is None else '%s_%s_in_forecast_hour.png'%(args.subset, metric)
            plot_metric(metric_plot, fh, metric_name, plot_sname, 
                        climatology = clim_plot, persistence = persist_plot,
                        month = None, year = None, add_variation = True,
                        path = args.figure_path, savename = savename)
            
            # Average over all NH spring and summer predictions
            ind = np.where( (months >= 3) & (months <= 8) )[0]

            # metric_plot = np.nanmean(metric_list[metric][ind,:], axis = 0)
            # clim_plot = np.nanmean(clim_list[metric][ind,:], axis = 0)
            # persist_plot = np.nanmean(persist_list[metric][ind,:], axis = 0)
            metric_plot = metric_list[metric][ind,:]
            clim_plot = clim_list[metric][ind,:]
            persist_plot = persist_list[metric][ind,:]

            # Make the second set of performance plots for summer months
            savename = '%s_summer_in_forecast_hour.png'%metric if args.subset is None else '%s_%s_summer_in_forecast_hour.png'%(args.subset, metric)
            plot_metric(metric_plot, fh, metric_name, plot_sname, 
                        climatology = clim_plot, persistence = persist_plot,
                        month = 6, year = None, add_variation = True,
                        path = args.figure_path, savename = savename)

            # Average over all NH fall and winter predictions
            ind = np.where( (months >= 9) | (months <= 2) )[0]

            # metric_plot = np.nanmean(metric_list[metric][ind,:], axis = 0)
            # clim_plot = np.nanmean(clim_list[metric][ind,:], axis = 0)
            # persist_plot = np.nanmean(persist_list[metric][ind,:], axis = 0)
            metric_plot = metric_list[metric][ind,:]
            clim_plot = clim_list[metric][ind,:]
            persist_plot = persist_list[metric][ind,:]

            # Make the final set of performance plots for winter months
            savename = '%s_winter_in_forecast_hour.png'%metric if args.subset is None else '%s_%s_winter_in_forecast_hour.png'%(args.subset, metric)
            plot_metric(metric_plot, fh, metric_name, plot_sname, 
                        climatology = clim_plot, persistence = persist_plot,
                        month = 12, year = None, add_variation = True,
                        path = args.figure_path, savename = savename)
            
    # Score card calculations
    if args.make_score_cards:
        # List of metrics and variables to make score cards of
        metrics_to_plot = ['rmse', 'mae', 'acc']
        variables_to_plot = ['tp', 'tp_30d', 'e', 'pev', 'ndvi', 'evi', 'lai', 'fpar', 'd2m', 'tair', 'sp',
                             'swvl1', 'swvl2', 'swvl3', 'swvl4']#, 'fdii1', 'fdii2', 'fdii3', 'fdii4', 'sesr']

        # Determine some colorbar information based on the metric
        for metric in metrics_to_plot:
            if metric == 'acc':
                cmin = -10; cmax = 10; cint = 1
            else:
                cmin = -15; cmax = 15; cint = 1

            # Make the score cards for each of the each metric and variable
            for variable in variables_to_plot:
                savename = 'score_card_%s_%s.png'%(metric, variable) if args.subset is None else 'score_card_%s_%s_%s.png'%(metric, variable, args.subset)
                make_score_cards(clim_list['%s_%s'%(metric, variable)], metric_list['%s_%s'%(metric, variable)],
                                 months, variable = variable, score = metric, 
                                 subset = 'none' if args.subset == None else args.subset,
                                 cmin = cmin, cmax = cmax, cint = cint, 
                                 path = args.figure_path, savename = savename)

    # Make the plots evaluating model variation if requested
    if args.make_variation_plots:
        make_variation_plots(args)
       
    # Make the case study plots if requested
    if args.make_case_studies:
        # Turn the start day of the case study into a datetime
        case_study_start = pd.to_datetime(args.case_study_start_date)
        case_study_start = case_study_start.to_pydatetime()

        case_study_end = case_study_start + timedelta(days = forecast_length)

        # Determine what rotation the predictions are in
        for key in test_years.keys():
            if case_study_start.year in test_years[key]:
                rot = key

        pred_path = path_to_rotation[rot]

        # Variables to plot for case study
        variables_to_plot = ['temperature', 'dewpoint',
                             'precipitation', 'precipitation_30day', 
                             'evaporation', 'potential_evaporation', 
                             'ndvi', 'evi', 'lai', 'fpar', 
                             'soil_moisture_1', 'soil_moisture_2', 
                             'fdii1', 'fdii2', 
                             'sesr']
        
        if args.nprocesses > 1:
            # Use multiprocessing to make case study plots for multiple variables simultaneously

            # Set parameters for the case study plots
            param_args = [(args, variable, case_study_start, case_study_end, pred_path, False) for variable in variables_to_plot]

            # Make case study plots with nprocesses
            with mp.Pool(args.nprocesses) as pool:
                data = pool.starmap(make_case_study_plots, param_args)
        else:
            # Make the case study plots one variable at a time
            for variable in tqdm(variables_to_plot):
                make_case_study_plots(args, variable, case_study_start, case_study_end, pred_path, verbose = True)
        


    
