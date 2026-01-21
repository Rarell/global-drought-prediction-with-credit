import numpy as np

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
upper_air_variables = ['u', 'v', 'z', 'q']
skip_variables = ['time', 'latitude', 'longitude', 'forecast_step', 'datetime']
underscore_variables = ['tp']

def subset_data(data, latitude, longitude, subset):
    '''
    Subset global dataset to a specific region

    TODO:
    - Add more subset regions
    '''

    # Limits for specific regions
    if subset in subsets.keys():
        lower_lat = subsets[subset][0]
        upper_lat = subsets[subset][1]
        lower_lon = subsets[subset][2]
        upper_lon = subsets[subset][3]
        
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


def new_sort(files):
    '''
    Sorts .nc with string formates of 024 .. 984 to 1008 ...
    NOTE: THIS HARD ASSUMES A FILENAME STRUCTURE AND THAT THE 3 DIGIT UNDERSCORE IS ON THE 7TH FROM THE END ENTRY
    '''

    numbers = '0123456789'

    # Assumes filename is formated as filename_###.nc or filename_####.nc; so the -7 entry is a number for 4 digit numbers, and underscore for 3 digit
    underscore = []
    non_underscore = []

    for file in files:
        underscore.append(file) if file[-7] not in numbers else non_underscore.append(file)

    # Sets of files can be sorted without confusion
    underscore = np.sort(underscore)
    non_underscore = np.sort(non_underscore)

    # Concatenate files together to deliver sorted resorts
    files_sorted = np.concatenate([underscore, non_underscore])

    return files_sorted



def get_metric_information(metric):
    '''
    Given a string from CREDIT metrics column, parse the name into separate portions
    '''

    if metric in skip_variables: # These are what metrics are plotted against
        return

    if len(metric) < 5:
        var_name = 'Overall' # For overall metrics, only the metric name is in it
        metric_name = metric
        level = None

    elif metric.split('_')[1] in upper_air_variables:
        if metric.split('_')[1] == upper_air_variables[-1]: # q_tot gets caught by this and needs special attention to get reconstructed
            metric_name, var1, var2, level = metric.split('_')
            var_name = var1 + '_' + var2 # q_tot gets caught by this and needs special attention to get reconstructed
        else:
            metric_name, var_name, level = metric.split('_') # for u and v wind, there are three separate parts
        #var_name = '500 mb '+var_name if level == '0' else '200 mb '+var_name 
        level = int(level)

    elif (metric.split('_')[1] in underscore_variables) & (len(metric.split('_')) > 2): # Variables that contain an underscore but aren't upper air
        metric_name, var1, var2 = metric.split('_')
        var_name = var1 + '_' + var2 # q_tot gets caught by this and needs special attention to get reconstructed
        level = None

    else:
        metric_name, var_name = metric.split('_')
        level = None

    return metric_name, var_name, level

def least_squares(x, y):
    '''
    Performs a least-squares estimate on a time series
    
    Inputs:
    :param data: Input 1D data
    
    Outputs;
    :param xhat[0]: Linear slope value of the regressed time series
    :param xhat[1]: Linear intercept value of the regressed time series
    '''
    
    # Reshape the data
    T = x.size
    
    # Initialize some variables
    t = np.arange(T)
    E = np.ones((T, 2))
    
    # Define the model matrix
    E[:,0] = x
    #E[:,1] = stats.norm.rvs(loc = 0, scale = 1e-1, size = T) # Add a little white noise to prevent making a singular matrix
    E[:,1] = 1
    
    # Use least squares to find the seasonal trend
    invEtE = np.linalg.inv(np.dot(E.T, E))
    xhat = invEtE.dot(E.T).dot(y)
    
    return xhat[0], xhat[1]