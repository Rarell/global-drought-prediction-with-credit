"""Provides functions for calculation various metrics
(ACC, MAE, MSE, RMSE, and RPC) overall, and in space.
"""

import numpy as np
import pandas as pd
import zarr

from datetime import datetime, timedelta
from netCDF4 import Dataset

from utils import subset_data

def calculate_metric(
        variable, 
        metric, 
        var_name, 
        year, 
        month, 
        day, 
        data_type, 
        subset = None, 
        Ndays = 30, 
        apply_mask = False, 
        level = None, 
        path = 'test_data'
        ) -> list:
    '''
    Calculate a metric performance (RMSE, MAE, MSE, or ACC) of climatology and persistence for a variable

    Inputs:
    :param variable: Array of predictions to determine the performance of (np.ndarray with shape time x lat x lon)
    :param metric: Name of the performance metric to calculate (rmse, mse, mae, or acc)
    :param var_name: Short name of the variable the calculation is performed for (must be a key in true label zarr files)
    :param year: Year value of the start date of variable
    :param month: Month value of the start date of variable
    :param day: Day value of the start date of variable
    :param data_type: Type of data variable is classed under (i.e., what zarr file to find the true labels in; upper_air, surface, or diagnostic)
    :param subset: Name of the subset to use when subsetting data (None to not subset the data)
    :param Ndays: Number of days in the prediction dataset
    :param apply_mask: Boolean; Apply land-sea and aridity masks prior to metric calculations
    :param level: Pressure level index for upper air data (e.g., 0 for 200 mb data, 1 for 500 mb, etc); None if variable is not upper air
    :param path: Directory path to where the true label, masks, and latitude weights are stored

    Outputs:
    :param var_metric: List containing calculated performance metric for each time step in variable (i.e., len(var_metrics) = variable.shape[0])
    '''

    # Load the true labels
    root = zarr.open_group('%s/%s.%04d.zarr/'%(path, data_type, year))
    ytrue = root[var_name][:]
    times = root['time'][:]
    times = pd.to_datetime(times).to_numpy(dtype = datetime)

    # Reduce true labels to 3D for upper air variables
    if level is not None:
        ytrue = ytrue[:,level,:,:]

    # Determine the start date of the prediction dataset
    if subset is not None:
        # For the subset predictions, the time stamp starts at the 24 hour prediction, so no + 1 day needed
        # (trying day - 1 resulted in errors)
        start_day = datetime(year, month, day)
    else:
        start_day = datetime(year, month, day) + timedelta(days = 1) # + 1 day as that is what the climatology/persistence are trying to predict

    # End date of the prediction dataset
    end_day = start_day + timedelta(days = Ndays)

    # If the prediction time extends into the next year, load and use that
    if (end_day.year > start_day.year) | ((month == 12) & (day == 31)):
        # If the predictions extend into the next year, load the next year of data to use for predictions
        root_new = zarr.open_group('%s/%s.%04d.zarr/'%(path, data_type, end_day.year), mode = 'r')
        ytrue_new = root_new[var_name][:]
        times_new = root_new['time'][:]
        times_new = pd.to_datetime(times_new).to_numpy(dtype = datetime)

        # Reduce to 3D for upper air variables
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

    # Expand latitude weights along the longitude line (i.e., make it the same shape as the variable data)
    latitude_weights = np.ones((ytrue.shape[1], ytrue.shape[2]))
    for j in range(latitude_weights.shape[-1]):
        latitude_weights[:,j] = tmp

    # Apply aridity and land-sea masks
    if apply_mask:
        # Load the masks
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
        # Subset predicted data and true labels
        variable, _, _ = subset_data(variable, latitude, longitude, subset)
        ytrue, _, _ = subset_data(ytrue, latitude, longitude, subset)
        
        # Temporarily turn latitudes weights into 3D (which is what the subset_data assumes)
        tmp = np.ones((1, latitude_weights.shape[0], latitude_weights.shape[1]))
        tmp[0,:,:] = latitude_weights

        # Subset the latitude weights to the subset as variable and ytrue
        latitude_weights, _, _ = subset_data(tmp, latitude, longitude, subset)

    # Calculate the metric
    var_metric = []
    # Perform the calculation for each time step in variable
    for t in range(variable.shape[0]):
        # Calcualte acc
        if metric == 'acc':
            var_metric.append(calculate_acc(variable[t,:,:], ytrue[t,:,:], 
                                            latitude_weights = latitude_weights))
        # Calculate MAE
        elif metric == 'mae':
            var_metric.append(calculate_mae(variable[t,:,:], ytrue[t,:,:], 
                                            latitude_weights = latitude_weights))
        # Calculate MSE
        elif metric == 'mse':
            var_metric.append(calculate_mse(variable[t,:,:], ytrue[t,:,:], 
                                            latitude_weights = latitude_weights))
        # Calculate RMSE
        elif metric == 'rmse':
            var_metric.append(calculate_rmse(variable[t,:,:], ytrue[t,:,:], 
                                             latitude_weights = latitude_weights))
    
    return var_metric

def calculate_acc(
        pred, 
        ytrue, 
        latitude_weights = 1, 
        var_weights = 1
        ) -> float:
    '''
    Calculate the anomaly correlation coefficient (ACC)

    Inputs:
    :param pred: Model predictions for a variable (must be the same shape as ytrue)
    :param ytrue: True labels for a variable (must be the same shape as pred)
    :param latitude_weights: Latitude weights to apply during calculation (must be the same shape as pred and ytrue; default to scalar 1 for no modification)
    :param var_weights: Variable or sample weights to apply during calculation (must be the same shape as pred; default to scalar 1 for no modification)

    Outputs:
    :param acc: Calculated ACC value between pred and ytrue
    '''

    # Avoids division by 0
    epsilon = 1e-7

    # Collect deviations from the overall mean
    pred_prime = pred - np.nanmean(pred)
    yprime = ytrue - np.nanmean(ytrue)

    # Determine the denominator of the correlation separately (due to its lengthy nature)
    denominator = np.sqrt(np.nansum(pred_prime**2 * latitude_weights * var_weights)
                          * np.nansum(yprime**2 * latitude_weights * var_weights)) + epsilon
    
    # Calculate the ACC
    acc = np.nansum(pred_prime * yprime * latitude_weights * var_weights) / denominator

    return acc

def calculate_mae(
        pred, 
        ytrue, 
        latitude_weights = 1, 
        var_weights = 1
        ) -> float:
    '''
    Calculate mean absolute error (MAE)

    Inputs:
    :param pred: Model predictions for a variable (must be the same shape as ytrue)
    :param ytrue: True labels for a variable (must be the same shape as pred)
    :param latitude_weights: Latitude weights to apply during calculation (must be the same shape as pred and ytrue; default to scalar 1 for no modification)
    :param var_weights: Variable or sample weights to apply during calculation (must be the same shape as pred; default to scalar 1 for no modification)

    Outputs:
    :param mae: Calculated MAE value between pred and ytrue
    '''

    # Determine the error
    error = pred - ytrue

    # Calculate MAE; The scalar multiple was determined via comparison with CREDIT results of global results, its purpose is unknown
    mae = np.nanmean(np.abs(error) * latitude_weights * var_weights)*1.5708359 # Not sure why the factor is needed

    return mae

def calculate_mse(
        pred, 
        ytrue, 
        latitude_weights = 1, 
        var_weights = 1
        ) -> float:
    '''
    Calculate mean square error (MSE)

    Inputs:
    :param pred: Model predictions for a variable (must be the same shape as ytrue)
    :param ytrue: True labels for a variable (must be the same shape as pred)
    :param latitude_weights: Latitude weights to apply during calculation (must be the same shape as pred and ytrue; default to scalar 1 for no modification)
    :param var_weights: Variable or sample weights to apply during calculation (must be the same shape as pred; default to scalar 1 for no modification)

    Outputs:
    :param mse: Calculated MSE value between pred and ytrue
    '''

    # Determine the error
    error = pred - ytrue

    # Calculate MSE; The scalar multiple was determined via comparison with CREDIT results of global results, its purpose is unknown
    mse = np.nanmean(error**2 * latitude_weights * var_weights)*1.5708359

    return mse

def calculate_rmse(
        pred, 
        ytrue, 
        latitude_weights = 1, 
        var_weights = 1
        ) -> float:
    '''
    Calculate the root mean square error (RMSE)

    Inputs:
    :param pred: Model predictions for a variable (must be the same shape as ytrue)
    :param ytrue: True labels for a variable (must be the same shape as pred)
    :param latitude_weights: Latitude weights to apply during calculation (must be the same shape as pred and ytrue; default to scalar 1 for no modification)
    :param var_weights: Variable or sample weights to apply during calculation (must be the same shape as pred; default to scalar 1 for no modification)

    Outputs:
    :param rmse: Calculated RMSE value between pred and ytrue
    '''

    # Determine the error
    error = pred - ytrue

    # Calculate the RMSE; The scalar multiple was determined via comparison with CREDIT results of global results, its purpose is unknown
    rmse = np.sqrt(np.nanmean(error**2 * latitude_weights * var_weights)*1.5708359)

    return rmse

def calculate_acc_in_space(
        pred, 
        ytrue, 
        latitude_weights = 1, 
        var_weights = 1
        ) -> np.ndarray:
    '''
    Calculate the anomaly correlation coefficient (ACC) in space (assumes pred and ytrue are time x lat x lon format)

    Inputs:
    :param pred: Array of model predictions for a variable (np.ndarray with shape time x lat x lon)
    :param ytrue: Array of true labels for a variable (np.ndarray with shape time x lat x lon)
    :param latitude_weights: Array of latitude weights to apply during calculation (must be the same shape as pred and ytrue; default to scalar 1 for no modification)
    :param var_weights: Array of variable or sample weights to apply during calculation (must be the same shape as pred; default to scalar 1 for no modification)

    Outputs:
    :param acc: Calculated ACC value between pred and ytrue (np.ndarray with shape lat x lon)
    '''

    # Avoids division by 0
    epsilon = 1e-7

    # Collect deviations from the overall mean
    pred_prime = pred - np.nanmean(pred, axis = 0)
    yprime = ytrue - np.nanmean(ytrue, axis = 0)

    # Determine the denominator of the correlation separately (due to its lengthy nature)
    denominator = np.sqrt(np.nansum(pred_prime**2 * latitude_weights * var_weights, axis = 0)
                          * np.nansum(yprime**2 * latitude_weights * var_weights, axis = 0)) + epsilon
    
    # Calculate the ACC
    acc = np.nansum(pred_prime * yprime * latitude_weights * var_weights, axis = 0) / denominator

    return acc

def calculate_mae_in_space(
        pred, 
        ytrue, 
        latitude_weights = 1, 
        var_weights = 1
        ) -> np.ndarray:
    '''
    Calculate mean absolute error (MAE) in space (assumes pred and ytrue are time x lat x lon format)

    Inputs:
    :param pred: Array of model predictions for a variable (np.ndarray with shape time x lat x lon)
    :param ytrue: Array of true labels for a variable (np.ndarray with shape time x lat x lon)
    :param latitude_weights: Array of latitude weights to apply during calculation (must be the same shape as pred and ytrue; default to scalar 1 for no modification)
    :param var_weights: Array of variable or sample weights to apply during calculation (must be the same shape as pred; default to scalar 1 for no modification)

    Outputs:
    :param mae: Calculated MAE value between pred and ytrue (np.ndarray with shape lat x lon)
    '''

    # Determine the error
    error = pred - ytrue

    # Calculate MAE
    mae = np.nanmean(np.abs(error) * latitude_weights * var_weights, axis = 0)*1.5708359 # Not sure why the factor is needed

    return mae

def calculate_mse_in_space(
        pred, 
        ytrue, 
        latitude_weights = 1, 
        var_weights = 1
        ) -> np.ndarray:
    '''
    Calculate mean square error (MSE) in space (assumes pred and ytrue are time x lat x lon format)

    Inputs:
    :param pred: Array of model predictions for a variable (np.ndarray with shape time x lat x lon)
    :param ytrue: Array of true labels for a variable (np.ndarray with shape time x lat x lon)
    :param latitude_weights: Array of latitude weights to apply during calculation (must be the same shape as pred and ytrue; default to scalar 1 for no modification)
    :param var_weights: Array of variable or sample weights to apply during calculation (must be the same shape as pred; default to scalar 1 for no modification)

    Outputs:
    :param mse: Calculated MSE value between pred and ytrue (np.ndarray with shape lat x lon)
    ''' 

    # Determine the error
    error = pred - ytrue

    # Calculate MSE
    mse = np.nanmean(error**2 * latitude_weights * var_weights, axis = 0)*1.5708359

    return mse

def calculate_rmse_in_space(
        pred, 
        ytrue, 
        latitude_weights = 1, 
        var_weights = 1
        ) -> np.ndarray:
    '''
    Calculate the root mean square error (RMSE) in space (assumes pred and ytrue are time x lat x lon format)

    Inputs:
    :param pred: Array of model predictions for a variable (np.ndarray with shape time x lat x lon)
    :param ytrue: Array of true labels for a variable (np.ndarray with shape time x lat x lon)
    :param latitude_weights: Array of latitude weights to apply during calculation (must be the same shape as pred and ytrue; default to scalar 1 for no modification)
    :param var_weights: Array of variable or sample weights to apply during calculation (must be the same shape as pred; default to scalar 1 for no modification)

    Outputs:
    :param rmse: Calculated RMSE value between pred and ytrue (np.ndarray with shape lat x lon)
    '''

    # Determine the error
    error = pred - ytrue

    # Calculate the RMSE
    rmse = np.sqrt(np.nanmean(error**2 * latitude_weights * var_weights, axis = 0)*1.5708359)

    return rmse

def calculate_rpc(
        obs, 
        pred
        ) -> float:
    '''
    Calculate the ratio of predictable components (RPC) comparing variation 
    of observations and a non-ensemble using pearson correlation coefficient 
    and model correlation coefficient using the method outlined in Broker et al. 2023 
    (https://doi.org/10.1002/qj.4440)

    Note the signal-to-noise ratio can also be obtained as std(pred)/std(obs - pred)

    Inputs:
    :param obs: List/array of observation values for a variable
    :param pred: List/array of model predictions for a variable

    Outputs:
    :param rpc: The RPC value comparing obs and pred
    '''

    # Used masked arrays to deal with NaNs
    obs_ma = np.ma.masked_invalid(obs.flatten())
    pred_ma = np.ma.masked_invalid(pred.flatten())

    # In comparing two variables, the covariance matrix is 2 x 2; 
    # diagonals are var(pred) and var(obs) respectively, and off diagonals are covariance of pred and obs
    cov_met = np.ma.cov(pred_ma, obs_ma)

    # .filled returns cov_met to a np.ndarray (replaces any masked values with NaNs)
    cov_met = cov_met.filled(np.nan)

    # Variance in the model error
    var_e = np.nanvar(obs - pred)

    # Determine the correlation between model and observations
    rho = cov_met[0,1]/np.sqrt(cov_met[0,0] * cov_met[1,1])

    # Determine the model correlation
    rho_f = np.sqrt(cov_met[0,0]/(cov_met[0,0] + var_e))

    # Calculate the RPC
    rpc = rho/rho_f
    
    return rpc
