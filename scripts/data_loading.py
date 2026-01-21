import numpy as np
import pandas as pd
import zarr

from glob import glob
from netCDF4 import Dataset
from datetime import datetime, timedelta

from utils import get_metric_information, new_sort

# Define the variables skip in the analysis
skip_variables = ['time', 'latitude', 'longitude', 'forecast_step', 'datetime']

### Main TODO:
###    Make zarr name open more generic
zarr.config.set({'default_zarr_format': 2})

def load_climatology(var_name, month, day, Ndays = 30, level = None, clim_year = 2000, type = 'standard', path = 'test_data'):
    '''
    Load 30-days of climatology data
    '''

    # Load the climatology data for a given variable
    root = zarr.open_group('%s/climatology.zarr/'%path, mode = 'r')
    climatology = root[var_name][:]
    times = root['time'][:]
    times = pd.to_datetime(times).to_numpy(dtype = datetime)

    if level is not None:
        climatology = climatology[:,level,:,:]

    if (month == 12) & (day == 31):
        start_day = datetime(1999, month, day) + timedelta(days = 1) # Special case for end of year
    else:
        start_day = datetime(2000, month, day) + timedelta(days = 1)
    end_day = start_day + timedelta(days = Ndays)
    

    if type == 'standard':
        # Make climatology forecasts based on the start day only
        ind = np.where(times == start_day)[0]
        tmp = climatology[ind,:,:]
        tmp_clim = np.ones((Ndays, climatology.shape[1], climatology.shape[2]))
        for t in range(Ndays):
            tmp_clim[t,:,:] = tmp

        climatology = tmp_clim
    elif type == 'running':
        # Extract the data for 1 day after initial time (persistence forecast = predicting what happened the day before)
        if (end_day.year > start_day.year):
            # If the predictions are near the end of year (the prediction window extends into the next year)
            # make sure to include everything in the next year
            end_day_new = datetime(end_day.year-1, end_day.month, end_day.day)

            ind = np.where(times >= start_day)[0]
            ind_tmp = np.where(times <= end_day_new)[0]
            ind = np.concatenate([ind, ind_tmp])
            end_day = end_day_new


        else:
            ind = np.where((times >= start_day) & (times <= end_day))[0]

        clim_year = clim_year + 1 if month >= 10 else clim_year # A correction if the prediction starts at the end of the year

        # Collect the index for the leap day
        leap_day = datetime(2000,2,29)
        ind_leap_day = np.where(times == leap_day)[0]
        leap_year = ((clim_year % 4) == 0) | ((clim_year + 1) % 4 == 0)
        last_day = ((month == 12) & (day == 31))
        if np.invert(leap_year) & (ind_leap_day in ind):
            # Remove the leap day climatology for non-leap years 
            ind = np.delete(ind, ind_leap_day == ind)

            # Add a new index to make up for the lost day and maintain length 91
            # Note het correction above collects an extra day for months >= 10, so that one does not need an extra day
            ind = np.append(ind, ind[-1]+1) if (month < 3) | last_day else ind
        elif (leap_year & (ind_leap_day in ind)) & (month >= 9) & np.invert(last_day):
            ind = ind[:-1]

        climatology = climatology[ind,:,:]
        climatology = climatology[1:]

    return climatology


def load_persistence(var_name, data_type, year, month, day, Ndays = 30, level = None, type = 'standard', path = 'test_data'):
    '''
    Load persistence forecast for 30-days

    TODO:
    Add if block for u and v wind
    '''

    # Load the data for a given variable
    root = zarr.open_group('%s/%s.%04d.zarr/'%(path, data_type, year), mode = 'r')
    persistence = root[var_name][:]
    times = root['time'][:]
    times = pd.to_datetime(times).to_numpy(dtype = datetime)

    if level is not None:
        persistence = persistence[:,level,:,:]

    start_day = datetime(year, month, day) # + timedelta(days = 1)
    end_day = start_day + timedelta(days = Ndays)

    if type == 'standard':
        # Make persistence forecasts based on the start day only
        ind = np.where(times == start_day)[0]

        tmp = persistence[ind,:,:]
        tmp_persist = np.ones((Ndays, persistence.shape[1], persistence.shape[2]))
        for t in range(Ndays):
            tmp_persist[t,:,:] = tmp

        persistence = tmp_persist

    elif type == 'running':
        # Extract the data for 1 day after initial time (persistence forecast = predicting what happened the day before)
        if end_day.year > start_day.year:
            # If the predictions extend into the next year, load the next year of data to use for predictions
            root_new = zarr.open_group('%s/%s.%04d.zarr/'%(path, data_type, end_day.year), mode = 'r')
            persistence_new = root_new[var_name][:]
            times_new = root_new['time'][:]
            times_new = pd.to_datetime(times_new).to_numpy(dtype = datetime)

            if level is not None:
                persistence_new = persistence_new[:,level,:,:]
            
            # Concatenate the new year of data onto the current year
            persistence = np.concatenate([persistence, persistence_new])
            times = np.concatenate([times, times_new])

        ind = np.where((times >= start_day) & (times <= end_day))[0]
        persistence = persistence[ind,:,:]
        persistence = persistence[1:]

    return persistence

def load_metrics(rotations, dates, rotation_years, subset = None, forecast_length = 90, 
                 path_to_data = './results', 
                 paths_to_rotation = {0: './results/rotation_00', 'single_run': '../results/one_experiment_run'}):
    '''
    '''

    # Initial list metrics; load in a test set of metrics that is known to exist
    if subset is None:
        metrics_initial = pd.read_csv('%s/forecasts/metrics/%04d-01-01T00Z.csv'%(paths_to_rotation[0], 2023), sep = ',',
        # metrics_initial = pd.read_csv('%s/forecasts/metrics/2020-01-01T00Z.csv'%(paths_to_rotation['single_run']), sep = ',',
                                    header = 0, index_col = 0, nrows = forecast_length) # Read 90 rows for all forecasts 
    else: 
        metrics_initial = pd.read_csv('%s/forecasts/metrics/%04d-01-01T00Z_africa.csv'%(paths_to_rotation[0], 2023), sep = ',',
        # metrics_initial = pd.read_csv('%s/forecasts/metrics/2020-01-01T00Z.csv'%(paths_to_rotation['single_run']), sep = ',',
                                    header = 0, index_col = 0, nrows = forecast_length) # Read 90 rows for all forecasts 
    
    # Loop through the rotations and load performance metric based on said rotations, for plotting
    metrics = {}
    climatology = {}
    persistence = {}
    n = 0
    for rot in rotations:
        # Determine the path
        data_path = '%s/forecasts/metrics'%paths_to_rotation[rot]

        rot_years = rotation_years[rot]

        # Make all datetimes in the selected rotations
        start_tmp = datetime(rot_years[0], 1, 1)
        end_tmp = datetime(rot_years[-1], 4, 4) if rot_years[-1] == 2024 else datetime(rot_years[-1], 12, 31) # Special case for last year in the dataset
        N_days = (end_tmp - start_tmp).days+1
            
        rotation_dates = np.array([start_tmp + timedelta(days = day) for day in range(N_days)])

        # Collect all files to load
        if subset is None:
            files = glob('%s/*T00Z.csv'%(data_path), recursive = True)
            if rot == 'single_run':
                clim_files = np.concatenate([
                    glob('%s/climatology/rotation_10/clim_2020*.nc'%(path_to_data), recursive = True),
                    glob('%s/climatology/rotation_11/clim_2021*.nc'%(path_to_data), recursive = True)
                ])
                persist_files = np.concatenate([
                    glob('%s/persistence/rotation_10/persist_2020*.nc'%(path_to_data), recursive = True),
                    glob('%s/persistence/rotation_11/persist_2021*.nc'%(path_to_data), recursive = True)
                ])
            else:
                clim_files = glob('%s/climatology/rotation_%02d/clim_*.nc'%(path_to_data, rot), recursive = True)
                persist_files = glob('%s/persistence/rotation_%02d/persist_*.nc'%(path_to_data, rot), recursive = True)
        else:
            files = glob('%s/*T00Z_%s.csv'%(data_path, subset), recursive = True)
            if rot == 'single_run':
                clim_files = np.concatenate([
                    glob('%s/climatology/rotation_10/%s_clim_2020*.nc'%(path_to_data, subset), recursive = True),
                    glob('%s/climatology/rotation_11/%s_clim_2021*.nc'%(path_to_data, subset), recursive = True)
                ])
                persist_files = np.concatenate([
                    glob('%s/persistence/rotation_10/%s_persist_2020*.nc'%(path_to_data, subset), recursive = True),
                    glob('%s/persistence/rotation_11/%s_persist_2021*.nc'%(path_to_data, subset), recursive = True)
                ])
            else:
                clim_files = glob('%s/climatology/rotation_%02d/%s_clim_*.nc'%(path_to_data, rot, subset), recursive = True)
                persist_files = glob('%s/persistence/rotation_%02d/%s_persist_*.nc'%(path_to_data, rot, subset), recursive = True)
        
        # Select out only those climatology and persistence files that correspond to the rotation being loaded
        ind = [np.where(date == dates)[0][0] for date in rotation_dates]
        print('ind length: ', len(ind))
        if len(files) > len(clim_files):
            files = new_sort(files)[:len(clim_files)]
        #     clim_files = new_sort(clim_files)[:len(files)]
        #     persist_files = new_sort(persist_files)[:len(files)]
        # else:
        #     clim_files = new_sort(clim_files)
        #     persist_files = new_sort(persist_files)
        clim_files = new_sort(clim_files)#[:-2] if subset is None else new_sort(clim_files)#[ind] #if (subset is None) & np.invert(rot == 0) else new_sort(clim_files)
        persist_files = new_sort(persist_files)#[:-2] if subset is None else new_sort(persist_files)#[ind] #if (subset is None) & np.invert(rot == 0) else new_sort(persist_files)

        print(len(clim_files))
        print(len(files))

        # Load the metric data + climatology and persistence data
        for m, file in enumerate(new_sort(files)):
            metrics[n] = pd.read_csv(file, sep = ',',
                                    names = metrics_initial.columns, 
                                    index_col = 0, 
                                    header = 0,
                                    nrows = forecast_length)
            
            # Load climatology metrics
            with Dataset(clim_files[m], 'r') as nc:
                climatology[n] = {}
                for key in nc.variables.keys():
                    climatology[n][key] = nc.variables[key][:]

            # Load persistence metrics
            with Dataset(persist_files[m], 'r') as nc:
                persistence[n] = {}
                for key in nc.variables.keys():
                    persistence[n][key] = nc.variables[key][:]

            n = n + 1
    
    print(n)

    # Reorganize data to time x forecast day
    metric_list = {}
    clim_list = {}
    persist_list = {}
    for metric in metrics_initial.columns:
        if metric in skip_variables: # These include time and forecast_step, which is what the metrics are plotted against
            continue
        
        # Collect individual metric information
        metric_name, var_name, level = get_metric_information(metric)

        if (var_name == 'Overall') & (subset is not None):
            continue

        # Convert dictionary information of a variable and metric into arrays of size N_forecasts x forecast_length
        metric_list[metric] = []
        clim_list[metric] = []
        persist_list[metric] = []
        for key in metrics.keys():
            # Get the metric
            metric_list[metric].append(metrics[key][metric])
            clim_list[metric].append(climatology[key][metric])
            persist_list[metric].append(persistence[key][metric])

        metric_list[metric] = np.array(metric_list[metric])
        clim_list[metric] = np.array(clim_list[metric])
        persist_list[metric]= np.array(persist_list[metric])

    # print(metric_list.keys())

    return metric_list, clim_list, persist_list
