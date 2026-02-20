"""Provides a set of functions for general utility, 
such as subsetting data, sorting, performing least 
squares regression, and parsing large metric names
"""

import numpy as np
from typing import Tuple

# All acceptable subsets and their upper and low lat/lon
subsets = { # Formatted as lower_lat, upper_lat, lower_lon, upper_lon
    'nh': [23.5, 90, 0, 360],
    'sh': [-90, -23.5, 0, 360],
    'tropics': [-23.5, 23.5, 0, 360],
    'africa': [-35, 35, 335, 53],
    'africa_nh': [23.5, 35, 335, 53],
    'africa_sh': [-35, -23.5, 335, 53],
    'africa_tropics': [-23.5, 23.5, 335, 53],
    'conus': [24, 55, 230, 300]
}

# List of different variable times (e.g., upper air variables, whether they have understcores in the name or whether to skip them)
upper_air_variables = ['u', 'v', 'z', 'q']
skip_variables = ['time', 'latitude', 'longitude', 'forecast_step', 'datetime']
underscore_variables = ['tp']

def subset_data(
        data, 
        latitude, 
        longitude, 
        subset
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Subset global dataset to a specific region
    Note this function takes out a box region from a larger set of spatial data

    Inputs:
    :param data: Data to make the subset for (must be 2D or 3D array)
    :param latitude: 1D list/array of laitudes for data
    :param longitude: 1D list/array of longitudes for data
    :param subset: Name of the subset region. Must be a key in subsets

    Outputs:
    :param data_sub: The subsetted dataset
    :param lat_sub: Latitudes for subsetted region
    :param lon_sub: Longitudes for subsetted region
    '''

    # Limits for specific regions
    if subset in subsets.keys():
        lower_lat = subsets[subset][0]
        upper_lat = subsets[subset][1]
        lower_lon = subsets[subset][2]
        upper_lon = subsets[subset][3]
    else:
        sub_list = [sub for sub in subsets.keys()]
        assert "subset must be one of {sub_list}, but got {subset} instead".format(sub_list = sub_list, subset = subset)
        
    # Get longitude indices for the subset
    if lower_lon > upper_lon:
        lon_ind = np.where((longitude >= lower_lon) | (longitude <= upper_lon))[0]
    else:
        lon_ind = np.where((longitude >= lower_lon) & (longitude <= upper_lon))[0]

    # Get the latitude indices for the subset
    lat_ind = np.where((latitude >= lower_lat) & (latitude <= upper_lat))[0]

    # Subset the latitude and longitude
    lat_sub = latitude[lat_ind]
    lon_sub = longitude[lon_ind]

    # Subset the data
    if len(data.shape) < 3:
        data_sub = data[:,lon_ind]
        data_sub = data_sub[lat_ind,:]
    else:
        data_sub = data[:,:,lon_ind]
        data_sub = data_sub[:,lat_ind,:]

    return data_sub, lat_sub, lon_sub


def new_sort(files) -> np.ndarray:
    '''
    Sorts .nc with string formates of 024 .. 984 to 1008 ...
    NOTE: THIS HARD ASSUMES A FILENAME STRUCTURE AND THAT THE 3 DIGIT UNDERSCORE IS ON THE 7TH FROM THE END

    Inputs:
    :param files: List if filenames to be sorted

    Outputs:
    :param files_sorted: Sorted filenames (1D np.array)
    '''

    numbers = '0123456789'

    # Assumes filename is formated as filename_###.nc or filename_####.nc; 
    # so the -7 entry is a number for 4 digit numbers, and underscore for 3 digit
    underscore = []
    non_underscore = []

    # Parse the list of files into two sets 3 digit and 4 digit numbers
    for file in files:
        underscore.append(file) if file[-7] not in numbers else non_underscore.append(file)

    # Sets of files can now be sorted without confusion
    underscore = np.sort(underscore)
    non_underscore = np.sort(non_underscore)

    # Concatenate files together to deliver sorted filenames
    files_sorted = np.concatenate([underscore, non_underscore])

    return files_sorted



def get_metric_information(metric) -> Tuple[str, str, int]:
    '''
    Given a string from CREDIT metrics column, parse the name into separate portions

    Inputs:
    :param metric: Raw name of the metric in the CREDIT csv file

    Outputs:
    :param metric_name: The name of the metric skill score
    :param var_name: The name of the variable evaluated
    :param level: The index for the corresponding pressure level for upper air variables (None if not an upper air variable)
    '''

    # Skip variables do not have compacted names, and they are what the metrics get plotted against
    if metric in skip_variables: 
        return

    # Determine if metric is a performance metric over all variables (in which case metric only contains metric_name)
    if len(metric) < 5:
        var_name = 'Overall' # For overall metrics, only the metric name is in it
        metric_name = metric
        level = None

    # Determine if the metric is for an upper air variable
    elif metric.split('_')[1] in upper_air_variables:
        # Note q_tot gets an extra split and needs special attention to be reconstructed
        if metric.split('_')[1] == upper_air_variables[-1]: 
            metric_name, var1, var2, level = metric.split('_')

            # Special attention for q_tot
            var_name = var1 + '_' + var2 
        else:
            # Split metric into metric_name, var_name, and level
            metric_name, var_name, level = metric.split('_')

        # Make sure level is an int so it can be used as an index
        level = int(level)

    # Underscore variables (e.g., tp_7d, tp_14d, tp_30d) require special attention since they split into an extra entry
    elif (metric.split('_')[1] in underscore_variables) & (len(metric.split('_')) > 2):
        metric_name, var1, var2 = metric.split('_')

        # Special attention for underscore variables (note this is for non upper-air variables; level = None)
        var_name = var1 + '_' + var2
        level = None

    else:
        # Split non upper air variables into metric_name and var_name (level = None for non upper-air)
        metric_name, var_name = metric.split('_')
        level = None

    return metric_name, var_name, level

def least_squares(x, y) -> Tuple[float, float]:
    '''
    Perform a linear least-squares estimate on a time series (i.e., estimate y using x with a linear regression)
    
    Inputs:
    :param x,y: Input time series to be related via regression line
    
    Outputs;
    :param xhat[0]: Linear slope value of the regressed time series
    :param xhat[1]: Linear intercept value of the regressed time series
    '''
    
    # Get the length of the timeseries
    T = x.size
    
    # Initialize some variables
    t = np.arange(T)
    E = np.ones((T, 2))
    
    # Define the model matrix
    E[:,0] = x # x estimates
    E[:,1] = 1 # bias term
    
    # Use least squares (matrix form) to find the linear regression
    invEtE = np.linalg.inv(np.dot(E.T, E))
    xhat = invEtE.dot(E.T).dot(y)
    
    return xhat[0], xhat[1]