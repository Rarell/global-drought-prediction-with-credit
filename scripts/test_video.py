'''
Hard assumptions: 1 year of forecasts

Command line:  python Scripts/test_video.py -Y 2018 -m 6 --forecast_days 30 --subset africa -var temperature --apply_mask

NOTE: This is version 3
TODO:
    Think about error maps; primes based on means of the 90 predictions, or climatology means?
    LAM foreast evaluation
'''

# Library impots
import os, sys, warnings
import gc, re
import pickle
import numpy as np
import pandas as pd
import multiprocessing as mp
from netCDF4 import Dataset
from itertools import product
from scipy import stats
from scipy import interpolate
from scipy import signal
from scipy.special import gamma
from netCDF4 import Dataset, num2date
from datetime import datetime, timedelta
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from netCDF4 import Dataset
from datetime import datetime, timedelta
from argparse import ArgumentParser
from glob import glob
import zarr
import imageio

from metric_calculations import calculate_metric, calculate_acc_in_space, calculate_rmse_in_space
from data_loading import load_climatology, load_persistence
from plotting import plot_metric, make_comparison_map, make_comparison_subset_map, make_error_map, make_error_subset_map, make_histogram, make_histogram_scatter_plot
from utils import subset_data, get_metric_information, new_sort

### Main TODO:
###    Make zarr name open more generic
zarr.config.set({'default_zarr_format': 2})            

subsets = ['africa', 'south central']
upper_air_variables = ['u', 'v', 'z', 'q_tot']
diagnostic_variables = ['ndvi', 'evi', 'lai', 'fpar', 'swvl1', 'swvl2', 'swvl3', 'swvl4', 'fdii1', 'fdii2', 'fdii3', 'fdii4', 'sesr', 'tp_30d']
skip_variables = ['time', 'latitude', 'longitude', 'forecast_step', 'datetime']
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

def calculate_subset_metrics(date, metric_names, var_names, levels, metrics_initial, prediction_path):
    """
    """

    metrics_new = {}
    for metric in metrics_initial.columns:
        metrics_new[metric] = []

    # Load in data
    date_str = date.strftime('%Y-%m-%dT%HZ')
    files = glob('%s/forecasts/%s/pred_*.nc'%(prediction_path, date_str), recursive = True)
    print(date_str)

    files_sorted = new_sort(files)
    # Loop through predictions
    for file in files_sorted:
        # Load nc
        with Dataset(file, 'r') as nc:
            time_pred = nc.variables['time'][:]
            time_pred = datetime(1900,1,1) + timedelta(hours = time_pred.item())

            # Load in data and subset it
            y_pred = {}
            for var in nc.variables.keys():
                if var in skip_variables:
                    # Skip the description variables
                    continue
                else:
                    y_pred[var] = nc.variables[var][:]


        # Collect metrics
        m = 0
        for metric in metrics_initial.columns:
            if (var in skip_variables) | (var_names[m] == 'Overall'): # These are what metrics are plotted against
                continue

            # Collect the type of variable (diagnostic, upper air, or surface) from which to collect the true labels
            if var_names[m] in diagnostic_variables:
                data_type = 'diagnostic'
            elif var_names[m] in upper_air_variables:
                data_type = 'upper_air'
            else:
                data_type = 'surface'

            # Reduce to 3D for upper air variables
            if levels[m] is not None:
                tmp = y_pred[var_names[m]][:,levels[m],:,:]
            else:
                tmp = y_pred[var_names[m]]

            # Calculate the forecast metrics
            metrics_new[metric].append(calculate_metric(tmp, metric_names[m], var_names[m],
                                                        time_pred.year, time_pred.month, time_pred.day, data_type, 
                                                        subset = args.subset, Ndays = args.forecast_days, apply_mask = args.apply_mask,
                                                        level = levels[m], prediction_dataset = 'subset', path = args.data_path))
            
            m = m+1 # This method is needed instead of enumerate because forecast_hour and datetime can skew the index m 

    return metrics_new, date

# Function to make the metric performance plots
def make_metric_plots(args):
    '''
    Make the metric plots

    # Hard assumptions: 
    - Files start at Jan. 1
    - Predictions start on first of the month

    TODO:
    - Add calculations for overall metrics when subsetting
    '''

    # Initial list metrics
    metrics_initial = pd.read_csv('%s/forecasts/metrics/%04d-%02d-01T00Z.csv'%(args.prediction_path, args.year, args.month), sep = ',',
    #metrics_initial = pd.read_csv('forecasts/metrics/metrics%04d-%02d-01T00Z.csv'%(args.year, args.month), sep = ',',
                                  header = 0, index_col = 0, nrows = args.forecast_days) # Read 21 rows for all forecasts 
    list_of_metrics = metrics_initial.columns
    fh = metrics_initial['forecast_step']

    # Get metric and variable names
    var_names = []
    levels = []
    metric_names = []
    for metric in metrics_initial.columns:
        if metric in skip_variables: # These include time and forecast_step, which is what the metrics are plotted against
            continue
        
        metric_name, var_name, level = get_metric_information(metric)

        metric_names.append(metric_name)
        var_names.append(var_name)
        levels.append(level)

    # Subset
    metrics = {}
    if args.subset is not None:
        # Create datetimes for all days in the month
        start_day = datetime(args.year, args.month, 1)
        end_day = datetime(args.year, args.month+1, 1)
        num_days = (end_day - start_day).days
        dates = np.array([start_day+timedelta(days = n) for n in range(num_days)])
        dates = dates[:-1] # Exclude the last day (day 1 on month+1)

        # Test
        # Make arg_params
        param_args = [(date, metric_names, var_names, levels, metrics_initial, args.prediction_path) for date in dates]
        with mp.Pool(args.nprocesses) as pool:
            calculated_metrics = pool.starmap(calculate_subset_metrics, param_args)
        for n, date in enumerate(dates):
            metrics[n] = {}
            for metric in metrics_initial.columns:
                metrics[n][metric] = []
            for datum in calculated_metrics:
                date_metric = datum[1]
                if date_metric == date:
                    # Turn the metrics into arrays
                    for metric in metrics_initial.columns: 
                        metrics[n][metric] = np.array(datum[0][metric]).squeeze()
                else:
                    continue


        # Loop through each day to load the appropiate forecasts
        # for n, date in enumerate(dates):
        #     metrics[n] = {}
        #     for metric in metrics_initial.columns:
        #         metrics[n][metric] = []

        #     # Load in data
        #     date_str = date.strftime('%Y-%m-%dT%HZ')
        #     files = glob('%s/forecasts/%s/pred_*.nc'%(args.prediction_path, date_str), recursive = True)
        #     print(date_str)

        #     files_sorted = new_sort(files)
        #     # Loop through predictions
        #     for file in files_sorted:
        #         # Load nc
        #         with Dataset(file, 'r') as nc:
        #             time_pred = nc.variables['time'][:]
        #             time_pred = datetime(1900,1,1) + timedelta(hours = time_pred.item())

        #             # Load in data and subset it
        #             y_pred = {}
        #             for var in nc.variables.keys():
        #                 if var in skip_variables:
        #                     # Skip the description variables
        #                     continue
        #                 else:
        #                     y_pred[var] = nc.variables[var][:]


        #         # Collect metrics
        #         m = 0
        #         for metric in metrics_initial.columns:
        #             if (var in skip_variables) | (var_names[m] == 'Overall'): # These are what metrics are plotted against
        #                 continue

        #             # Collect the type of variable (diagnostic, upper air, or surface) from which to collect the true labels
        #             if var_names[m] in diagnostic_variables:
        #                 data_type = 'diagnostic'
        #             elif var_names[m] in upper_air_variables:
        #                 data_type = 'upper_air'
        #             else:
        #                 data_type = 'surface'

        #             # Reduce to 3D for upper air variables
        #             if levels[m] is not None:
        #                 tmp = y_pred[var_names[m]][:,levels[m],:,:]
        #             else:
        #                 tmp = y_pred[var_names[m]]

        #             # Calculate the forecast metrics
        #             metrics[n][metric].append(calculate_metric(tmp, metric_names[m], var_names[m],
        #                                                         time_pred.year, time_pred.month, time_pred.day-2, data_type, 
        #                                                         subset = args.subset, Ndays = args.forecast_days, apply_mask = args.apply_mask,
        #                                                         level = levels[m], path = args.data_path))
                    
        #             m = m+1 # This method is needed instead of enumerate because forecast_hour and datetime can skew the index m 
            
        #     # Turn the metrics into arrays
        #     for metric in metrics_initial.columns: 
        #         metrics[n][metric] = np.array(metrics[n][metric]).squeeze()
        print(metrics.keys())
        #print(metrics['0'].keys())    

    else:
        # Global metrics

        # Determine what row to start examining in the metrics file
        # start_days = (datetime(args.year, args.month, 1) - datetime(args.year, 1, 1)).days
        # start_row = start_days*args.forecast_days+1

        # Load in all the metric files
        #metric_files = glob('forecasts/metrics/metrics%04d-%02d*.csv'%(args.year, args.month), recursive = True)
        metric_files = glob('%s/forecasts/metrics/%04d-%02d*.csv'%(args.prediction_path, args.year, args.month), recursive = True)
        for n, file in enumerate(np.sort(metric_files)):
            metrics[n] = pd.read_csv(file, sep = ',',
                                     names = metrics_initial.columns, 
                                     index_col = 0, 
                                     header = 0,
                                     nrows = args.forecast_days)
                                     #skiprows = args.forecast_days*n+start_row, nrows = args.forecast_days)
            
    # Climatology and persistence calculations
    m = 0
    for metric in metrics_initial.columns:
        if metric in skip_variables: # These are what metrics are plotted against
            continue

        if (var_names[m] == 'Overall') & (args.subset is not None):
            continue

        print(m, metric, var_names[m], metric_names[m])
        print('Working on ' + var_names[m] + ' climatology and persistence predictions')# for: ' + (datetime(args.year, args.month, 1)+timedelta(days=day)).isoformat())

        if var_names[m] == 'Overall':
            climatology = None
            persistence = None
        else:
            if var_names[m] in diagnostic_variables:
                data_type = 'diagnostic'
            elif var_names[m] in upper_air_variables:
                data_type = 'upper_air'
            else:
                data_type = 'surface'
            
            climatology = []
            persistence = []
            start = datetime(args.year, args.month, 1)
            for day in fh:
                # Load in persistence/climatology
                date = start + timedelta(day - 1)
                tmp_climatology = load_climatology(var_names[m], date.month, date.day, 
                                                    Ndays = args.forecast_days, level = levels[m], 
                                                    type = 'running', path = args.data_path)
                tmp_persistence = load_persistence(var_names[m], data_type, date.year, date.month, date.day, 
                                                    Ndays = args.forecast_days, level = levels[m], 
                                                    type = 'standard', path = args.data_path)

                # Calculate metric for persistence/climatology
                climatology.append(calculate_metric(tmp_climatology, metric_names[m], var_names[m], 
                                                    date.year, date.month, date.day, 
                                                    data_type, Ndays = args.forecast_days, apply_mask = args.apply_mask,
                                                    level = levels[m], path = args.data_path))
                persistence.append(calculate_metric(tmp_persistence, metric_names[m], var_names[m], 
                                                    date.year, date.month, date.day, 
                                                    data_type, Ndays = args.forecast_days, apply_mask = args.apply_mask,
                                                    level = levels[m], path = args.data_path))

            climatology = np.stack(climatology, axis = 0)
            climatology = np.nanmean(climatology, axis = 0)
            persistence = np.stack(persistence, axis = 0)
            persistence = np.nanmean(persistence, axis = 0)
        #print('Climatology: ', climatology)
        #print('Persistence: ', persistence)
        # For each metric, make a list of all performances (e.g., all RMSE arrays for tair will be in a list)
        # this is for making a spagetti plot
        metric_list = []
        for key in metrics.keys():
            # Get the metric
            metric_list.append(metrics[key][metric])

        metric_list = np.array(metric_list)
        print(metric_list.shape)

        # For upper air variables, add the pressure level to the name
        if var_names[m] in upper_air_variables:
            var_names[m] = '200 mb '+ var_names[m] if levels[m] == 0 else '500 mb '+ var_names[m] 

        # Make a plot of metrics
        savename = '%s_in_forecast_hour.png'%metric if args.subset is None else '%s_%s_in_forecast_hour.png'%(args.subset, metric)
        plot_metric(metric_list, fh, metric_names[m], var_names[m], 
                    climatology = climatology, persistence = persistence,
                    month = args.month, year = args.year,
                    path = args.figure_path, savename = savename)
        
        m = m+1

    return

def make_metric_error_plots(ytrue, files, time, lat, lon,
                            sname, plot_sname, subset, variable, data_path, figure_path):
    '''
    '''

    # Need to get sname and args.variable and lat and lon and times
    time = np.array([t.to_pydatetime() for t in time])

    y_pred = []
    time_pred = []

    # Load the latitude weights
    with Dataset('%s/lat_and_lons.nc'%data_path, 'r') as nc:
        tmp = nc.variables['latitude'][:]
        tmp = np.cos(tmp*np.pi/180)

    # exampe weights to the same shape as the variable data
    latitude_weights = np.ones((ytrue.shape[1], ytrue.shape[2]))
    for j in range(latitude_weights.shape[-1]):
        latitude_weights[:,j] = tmp

    # Load in the predictions
    for file in new_sort(files):
        with Dataset(file, 'r') as nc:
            time_pred_tmp = nc.variables['time'][:]
            time_pred.append(datetime(1900,1,1) + timedelta(hours = time_pred_tmp.item()))
            
            if ('200 mb' in variable) | ('200_mb' in variable):
                y_pred.append(nc.variables[sname][0,0,:,:])
            elif ('500 mb' in variable) | ('500_mb' in variable):
                y_pred.append(nc.variables[sname][0,1,:,:])
            else:
                y_pred.append(nc.variables[sname][0,:,:])

    y_pred = np.array(y_pred)
    
    time_ind = np.where( (time >= time_pred[0]) & (time <= time_pred[-1]) )[0]
    ytrue = ytrue[time_ind,:,:]

    # Calculate the error metrics
    acc = calculate_acc_in_space(y_pred, ytrue, latitude_weights = latitude_weights)
    rmse = calculate_rmse_in_space(y_pred, ytrue, latitude_weights = latitude_weights)


    # make the plots
    if subset is not None:
        make_error_subset_map(acc, rmse, lat, lon, subset,
                              var_name = plot_sname,
                              path = figure_path, savename = '%s_%s_error_map.png'%(args.subset, variable))
        
        if args.apply_mask:
            # with Dataset('%s/aridity_mask.nc'%data_path, 'r') as nc:
            with Dataset('%s/aridity_mask_reduced.nc'%data_path, 'r') as nc:
                # ai_mask = nc.variables['aim'][:,:-1,:] # Note the aridity mask also masks out the oceans
                ai_mask = nc.variables['aim'][:]
            
            # with Dataset('%s/land.nc'%data_path, 'r') as nc:
            with Dataset('%s/land_reduced.nc'%data_path, 'r') as nc:
                # lsm_mask = nc.variables['lsm'][:,:-1,:] # Includes other sea points not in the aridity mask (e.g., Great Lakes)
                lsm_mask = nc.variables['lsm'][:]

            # Apply the masks
            acc = np.where(ai_mask == 1, acc, np.nan)
            acc = np.where(lsm_mask == 1, acc, np.nan)

            rmse = np.where(ai_mask == 1, rmse, np.nan)
            rmse = np.where(lsm_mask == 1, rmse, np.nan)
            
        make_histogram(rmse, plot_sname, 'rmse', path = figure_path, savename = '%s_%s_rmse_histogram.png'%(args.subset, variable))

    else:
        make_error_map(acc, rmse, lat, lon,
                       var_name = plot_sname, globe = True,
                       path = figure_path, savename = '%s_error_map.png'%(variable))
        
        if args.apply_mask:
            # with Dataset('%s/aridity_mask.nc'%data_path, 'r') as nc:
            with Dataset('%s/aridity_mask_reduced.nc'%data_path, 'r') as nc:
                # ai_mask = nc.variables['aim'][:,:-1,:] # Note the aridity mask also masks out the oceans
                ai_mask = nc.variables['aim'][:]
            
            # with Dataset('%s/land.nc'%data_path, 'r') as nc:
            with Dataset('%s/land_reduced.nc'%data_path, 'r') as nc:
                # lsm_mask = nc.variables['lsm'][:,:-1,:] # Includes other sea points not in the aridity mask (e.g., Great Lakes)
                lsm_mask = nc.variables['lsm'][:]

            # Apply the masks
            acc = np.where(ai_mask == 1, acc, np.nan)
            acc = np.where(lsm_mask == 1, acc, np.nan)

            rmse = np.where(ai_mask == 1, rmse, np.nan)
            rmse = np.where(lsm_mask == 1, rmse, np.nan)

        make_histogram(rmse, plot_sname, 'rmse', path = figure_path, savename = '%s_rmse_histogram.png'%(variable))
    return

if __name__ == '__main__':
    description = 'Display forecast results from a CREDIT model'
    parser = ArgumentParser(description=description, fromfile_prefix_chars='@')
    parser.add_argument('--movie_type', type = str, default = 'show_forecast_hours', help = 'Make gif of model forecasts (movie_type = show_forecast_hours) or only use initial 1 day forecasts (movie_type = show_initial times)')
    parser.add_argument('--sname', type = str, default = 'swvl1', help = 'Name of the variable in the nc and zarr files')
    parser.add_argument('-var', '--variable', type = str, default = 'soil moisture 1', help = 'Full variable name of the variable being plotted in the gif')
    parser.add_argument('-Y', '--year', type = int, default = 2014, help= 'Year of the results displayed')
    parser.add_argument('-m', '--month', type = int, default = 6, help = 'Month of the results to be displayed')
    parser.add_argument('--forecast_days', type = int, default = 21, help = 'Number of days the model forecasted for')
    parser.add_argument('--subset', type = str, default = 'none', help = 'Subset the data to a given region before calculating metrics')
    parser.add_argument('--apply_mask', action = 'store_true', help = 'Apply land-sea and aridity masks to certain variables')
    parser.add_argument('--skip_metric_plots', action = 'store_true', help = 'Skip the metric plots against the forecast hour')

    parser.add_argument('--prediction_path', type = str, default = '/scratch/rarrell/credit_model', help = 'Path to credit predictions, where the forecasts directory is')
    parser.add_argument('--data_path', type = str, default = '/ourdisk/hpc/ai2es/sedris/credit_datasets', help = 'Path to true labels and climatology dataset')
    parser.add_argument('--figure_path', type = str, default = '/ourdisk/hpc/ai2es/sedris/scripts/Figures', help = 'path to where the created figures are stored')

    parser.add_argument('--nprocesses', type = int, default = 1, help = 'Number of multiprocesses to use when calculating metrics for subsets')

    # Parse the arguments
    args = parser.parse_args()

    # Determine if subset is a known subset; if not set to ignore it
    if args.subset.lower() not in subsets:
        args.subset = None
    
    sname = short_names[args.variable.lower()]

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

    # The type of mp4 to make
    #movie_type = 'show_forecast_hours' # Acceptable types: show_initial_times, show_forecast_hours
    if args.movie_type == 'show_forecast_hours':
        if not args.skip_metric_plots:
            make_metric_plots(args)

        # Collect true labels
        if sname in diagnostic_variables:
            data_type = 'diagnostic'
        elif sname in upper_air_variables:
            data_type = 'upper_air'
        else:
            data_type = 'surface'
        root = zarr.open('%s/%s.%04d.zarr/'%(args.data_path, data_type, args.year))
        y_true = root[sname][:]
        lat = root['latitude'][:]
        lon = root['longitude'][:]
        times = root['time'][:]
        times = pd.to_datetime(times).to_numpy(dtype = datetime)

        # Collect climatology predictions
        clim_root = zarr.open('%s/climatology.zarr/'%args.data_path)
        climatology = clim_root[sname][:]

        # Select the level for upper air variables
        if ('200 mb' in args.variable) | ('200_mb' in args.variable):
            y_true = y_true[:,0,:,:]
            climatology = climatology[:,0,:,:]
        elif ('500 mb' in args.variable) | ('500_mb' in args.variable):
            y_true = y_true[:,1,:,:]
            climatology = climatology[:,1,:,:]

        # Reverse the lattitude
        lat = lat[::-1, :]

        # Create a list of predictions
        files = glob('%s/forecasts/%04d-%02d-01T00Z/pred_*.nc'%(args.prediction_path, args.year, args.month), recursive = True)

        # Make the error maps    
        print('Making error map')
        make_metric_error_plots(y_true, files, times, lat, lon, sname, plot_sname, args.subset, args.variable, args.data_path, args.figure_path)

        # Loop through predictions
        for file in new_sort(files):
            # Load nc
            with Dataset(file, 'r') as nc:
                fh = nc.variables['forecast_hour'][:]
                time_pred = nc.variables['time'][:]
                time_pred = datetime(1900,1,1) + timedelta(hours = time_pred.item())
                if ('200 mb' in args.variable) | ('200_mb' in args.variable):
                    y_pred = nc.variables[sname][0,0,:,:]
                elif ('500 mb' in args.variable) | ('500_mb' in args.variable):
                    y_pred = nc.variables[sname][0,1,:,:]
                else:
                    y_pred = nc.variables[sname][0,:,:]

            # Find the corresponding true label
            time_ind = np.where(time_pred == times)[0]
            y = y_true[time_ind[0],:,:]
            clim_pred = climatology[time_ind[0],:,:]

            print('Making figure for', time_pred.isoformat())
            # Make plots
            if args.subset is not None:
                # Make the comparison maps
                make_comparison_subset_map(y, y_pred, clim_pred, lat, lon, time_pred, args.subset,
                                           var_name = plot_sname, forecast_hour = fh,
                                           path = args.figure_path, savename = '%s_%s_forecast_hour_%04d.png'%(args.subset, args.variable, fh))
                figure_fns = '%s_%s_forecast_hour_*.png'%(args.subset, args.variable)

                # Make histogram scatterplot
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

                make_histogram_scatter_plot(y, y_pred, plot_sname, fh/24, #max_counts = 60,
                                            path = args.figure_path, savename = '%s_%s_2d_histogram_forecast_hour_%04d.png'%(args.subset, args.variable, fh))
                histogram_figure_fns = '%s_%s_2d_histogram_forecast_hour_*.png'%(args.subset, args.variable)

            else:
                # Make the comparison maps
                make_comparison_map(y, y_pred, clim_pred, lat, lon, time_pred,
                                    var_name = plot_sname, globe = True, forecast_hour = fh,
                                    path = args.figure_path, savename = '%s_forecast_hour_%04d.png'%(args.variable, fh))
                figure_fns = '%s_forecast_hour_*.png'%args.variable
                
                # Make histogram scatterplot
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
                
                make_histogram_scatter_plot(y, y_pred, plot_sname, fh/24, #max_counts = 300,
                                            path = args.figure_path, savename = '%s_2d_histogram_forecast_hour_%04d.png'%(args.variable, fh))
                histogram_figure_fns = '%s_2d_histogram_forecast_hour_*.png'%(args.variable)
            
        #  Get all the newly created figures
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

    elif args.movie_type == 'show_initial_times':
        # Collect true labels
        root = zarr.open('%s/diagnostic.%04d.zarr/'%(args.data_path, args.year))
        y_true = root[sname][:]
        lat = root['latitude'][:]
        lon = root['longitude'][:]
        times = root['time'][:]
        times = pd.to_datetime(times).to_numpy(dtype = datetime)

        # Select the level for upper air variables
        if ('200 mb' in args.variable) | ('200_mb' in args.variable):
            y_true = y_true[:,0,:,:]
        elif ('500 mb' in args.variable) | ('500_mb' in args.variable):
            y_true = y_true[:,1,:,:]

        # Reverse the lattitude
        lat = lat[::-1, :]

        # Create a list of predictions
        directories = glob('%s/forecasts/%04d*/'%(args.prediction_path, args.year), recursive = True)
        #directories = ['forecasts/%s'%f for f in os.listdir('forecasts/') if re.match(r'%d+/'%2012, f)]

        # Loop through predictions
        pred_fn_base = 'pred_%s_024.nc'
        for path in np.sort(directories)[:-1]: # Exclude last path, which is metrics and not predictions
            _, time_base, _ = path.split('/')
            filename = pred_fn_base%time_base

            # Load nc
            with Dataset(path + filename, 'r') as nc:
                fh = nc.variables['forecast_step'][:]
                time_pred = nc.variables['time'][:]
                time_pred = datetime(1900,1,1) + timedelta(hours = time_pred.item())
                if ('200 mb' in args.variable) | ('200_mb' in args.variable):
                    y_pred = nc.variables[sname][0,0,:,:]
                elif ('500 mb' in args.variable) | ('500_mb' in args.variable):
                    y_pred = nc.variables[sname][0,1,:,:]
                else:
                    y_pred = nc.variables[sname][0,:,:]

            # Find the corresponding true label
            time_ind = np.where(time_pred == times)[0]
            y = y_true[time_ind[0],:,:]

            print('Making figure for ', time_pred.isoformat())
            # Make plots
            make_comparison_map(y, y_pred, lat, lon, time_pred,
                                var_name = plot_sname, globe = True, 
                                path = args.figure_path, savename = 'initial_forecast_%s.png'%time_pred.isoformat())

        # Get all the newly created figures
        images = []
        figures = glob('%s/initial_forecast_*.png'%args.figure_path, recursive = True)
        figures = np.sort(figures)
        for figure in figures:
            images.append(imageio.imread(figure))

    # Make the gif
    sname = '%s_gif.mp4'%args.variable if args.subset is None else '%s_%s_gif.mp4'%(args.subset, args.variable)
    imageio.mimsave('%s/%s'%(args.figure_path, sname), images, format = 'FFMPEG')

    sname = '%s_histogram_gif.mp4'%args.variable if args.subset is None else '%s_%s_histogram_gif.mp4'%(args.subset, args.variable)
    imageio.mimsave('%s/%s'%(args.figure_path, sname), histogram_images, format = 'FFMPEG')




