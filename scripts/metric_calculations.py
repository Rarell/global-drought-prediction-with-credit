import numpy as np
import pandas as pd
import zarr

from datetime import datetime, timedelta
from netCDF4 import Dataset

from utils import subset_data

def calculate_metric(variable, metric, var_name, year, month, day, data_type, 
                     subset = None, Ndays = 30, apply_mask = False, level = None, 
                     prediction_dataset = 'climatology', path = 'test_data'):
    '''
    Calculate a performance metric for climatology/persistence prediction for a given variable

    TODO:
       Add subset
    '''

    # Load the true labels
    root = zarr.open_group('%s/%s.%04d.zarr/'%(path, data_type, year))
    ytrue = root[var_name][:]
    times = root['time'][:]
    times = pd.to_datetime(times).to_numpy(dtype = datetime)

    if level is not None:
        ytrue = ytrue[:,level,:,:]

    if prediction_dataset == 'subset':
        # For the subset predictions, the time stamp starts at the 24 hour prediction, so no + 1 day needed
        # (trying day - 1 resulted in errors)
        start_day = datetime(year, month, day)
    else:
        start_day = datetime(year, month, day) + timedelta(days = 1) # + 1 day as that is what the climatology/persistence are trying to predict
    end_day = start_day + timedelta(days = Ndays)

    # If the prediction time extends into the next year, load and use that
    if (end_day.year > start_day.year) | ((month == 12) & (day == 31)):
        # If the predictions extend into the next year, load the next year of data to use for predictions
        root_new = zarr.open_group('%s/%s.%04d.zarr/'%(path, data_type, end_day.year), mode = 'r')
        ytrue_new = root_new[var_name][:]
        times_new = root_new['time'][:]
        times_new = pd.to_datetime(times_new).to_numpy(dtype = datetime)

        if level is not None:
            ytrue_new = ytrue_new[:,level,:,:]
        
        # Concatenate the new year of data onto the current year
        ytrue = np.concatenate([ytrue, ytrue_new])
        times = np.concatenate([times, times_new])

    # Extract the data for 1 day after initial time (persistence forecast = predicting what happened the day before)
    ind = np.where((times >= start_day) & (times <= end_day))[0]
    ytrue = ytrue[ind,:,:]

    # Load the latitude weights
    with Dataset('%s/lat_and_lons.nc'%path, 'r') as nc:
        #tmp = nc.variables['coslat'][:]
        latitude = nc.variables['latitude'][:]
        longitude = nc.variables['longitude'][:]
        tmp = nc.variables['latitude'][:]
        tmp = np.cos(tmp*np.pi/180)

    # exampe weights to the same shape as the variable data
    latitude_weights = np.ones((ytrue.shape[1], ytrue.shape[2]))
    for j in range(latitude_weights.shape[-1]):
        latitude_weights[:,j] = tmp

    if apply_mask:
        # with Dataset('%s/aridity_mask.nc'%path, 'r') as nc:
        with Dataset('%s/aridity_mask_reduced.nc'%path, 'r') as nc:
            # ai_mask = nc.variables['aim'][:,:-1,:] # Note the aridity mask also masks out the oceans
            ai_mask = nc.variables['aim'][:]
        
        # with Dataset('%s/land.nc'%path, 'r') as nc:
        with Dataset('%s/land_reduced.nc'%path, 'r') as nc:
            # lsm_mask = nc.variables['lsm'][:,:-1,:] # Includes other sea points not in the aridity mask (e.g., Great Lakes)
            lsm_mask = nc.variables['lsm'][:]

        # Apply the masks
        variable = np.where(ai_mask == 1, variable, np.nan)
        variable = np.where(lsm_mask == 1, variable, np.nan)

        ytrue = np.where(ai_mask == 1, ytrue, np.nan)
        ytrue = np.where(lsm_mask == 1, ytrue, np.nan)

    # Subset data
    if subset is not None:
        variable, _, _ = subset_data(variable, latitude, longitude, subset)
        ytrue, _, _ = subset_data(ytrue, latitude, longitude, subset)
        
        # temporarly turn latitudes weights into 3D (which is what the subset_data assumes)
        tmp = np.ones((1, latitude_weights.shape[0], latitude_weights.shape[1]))
        tmp[0,:,:] = latitude_weights
        latitude_weights, _, _ = subset_data(tmp, latitude, longitude, subset)

    # Calculate the metric
    var_metric = []
    for t in range(variable.shape[0]):
        if metric == 'acc':
            var_metric.append(calculate_acc(variable[t,:,:], ytrue[t,:,:], 
                                            latitude_weights = latitude_weights))
        elif metric == 'mae':
            var_metric.append(calculate_mae(variable[t,:,:], ytrue[t,:,:], 
                                            latitude_weights = latitude_weights))
        elif metric == 'mse':
            var_metric.append(calculate_mse(variable[t,:,:], ytrue[t,:,:], 
                                            latitude_weights = latitude_weights))
        elif metric == 'rmse':
            var_metric.append(calculate_rmse(variable[t,:,:], ytrue[t,:,:], 
                                             latitude_weights = latitude_weights))
    
    return var_metric

def calculate_acc(pred, ytrue, latitude_weights = 1, var_weights = 1):
    '''
    Calculate the anomaly correlation coefficient
    '''

    # Avoids division by 0
    epsilon = 1e-7

    # Collect deviations from the mean
    pred_prime = pred - np.nanmean(pred)
    yprime = ytrue - np.nanmean(ytrue)

    denominator = np.sqrt(np.nansum(pred_prime**2 * latitude_weights * var_weights)
                          * np.nansum(yprime**2 * latitude_weights * var_weights)) + epsilon
    
    # Calculate the ACC
    acc = np.nansum(pred_prime * yprime * latitude_weights * var_weights) / denominator

    return acc

def calculate_mae(pred, ytrue, latitude_weights = 1, var_weights = 1):
    '''
    Calculate absolute mean error
    '''

    # Determine the error
    error = pred - ytrue

    # Calculate MAE
    mae = np.nanmean(np.abs(error) * latitude_weights * var_weights)*1.5708359 # Not sure why the factor is needed

    return mae

def calculate_mse(pred, ytrue, latitude_weights = 1, var_weights = 1):
    '''
    Calculate mean square error
    '''

    # Determine the error
    error = pred - ytrue

    # Calculate MSE
    mse = np.nanmean(error**2 * latitude_weights * var_weights)*1.5708359

    return mse

def calculate_rmse(pred, ytrue, latitude_weights = 1, var_weights = 1):
    '''
    Calculate the root mean square error
    '''

    # Determine the error
    error = pred - ytrue

    # Calculate the RMSE
    rmse = np.sqrt(np.nanmean(error**2 * latitude_weights * var_weights)*1.5708359)

    return rmse

def calculate_acc_in_space(pred, ytrue, latitude_weights = 1, var_weights = 1):
    '''
    Calculate the anomaly correlation coefficient in space (assumes pred and ytrue are time x lat x lon format)
    '''

    # Avoids division by 0
    epsilon = 1e-7

    # Collect deviations from the mean
    pred_prime = pred - np.nanmean(pred, axis = 0)
    yprime = ytrue - np.nanmean(ytrue, axis = 0)

    denominator = np.sqrt(np.nansum(pred_prime**2 * latitude_weights * var_weights, axis = 0)
                          * np.nansum(yprime**2 * latitude_weights * var_weights, axis = 0)) + epsilon
    
    # Calculate the ACC
    acc = np.nansum(pred_prime * yprime * latitude_weights * var_weights, axis = 0) / denominator

    return acc

def calculate_mae_in_space(pred, ytrue, latitude_weights = 1, var_weights = 1):
    '''
    Calculate absolute mean error in space (assumes pred and ytrue are time x lat x lon format)
    '''

    # Determine the error
    error = pred - ytrue

    # Calculate MAE
    mae = np.nanmean(np.abs(error) * latitude_weights * var_weights, axis = 0)*1.5708359 # Not sure why the factor is needed

    return mae

def calculate_mse_in_space(pred, ytrue, latitude_weights = 1, var_weights = 1):
    '''
    Calculate mean square error in space (assumes pred and ytrue are time x lat x lon format)
    ''' 

    # Determine the error
    error = pred - ytrue

    # Calculate MSE
    mse = np.nanmean(error**2 * latitude_weights * var_weights, axis = 0)*1.5708359

    return mse

def calculate_rmse_in_space(pred, ytrue, latitude_weights = 1, var_weights = 1):
    '''
    Calculate the root mean square error in space (assumes pred and ytrue are time x lat x lon format)
    '''

    # Determine the error
    error = pred - ytrue

    # Calculate the RMSE
    rmse = np.sqrt(np.nanmean(error**2 * latitude_weights * var_weights, axis = 0)*1.5708359)

    return rmse
