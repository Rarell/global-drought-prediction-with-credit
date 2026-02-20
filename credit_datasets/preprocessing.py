"""General preprocessing and preparation of data for ML training and validation

Provides functions for loading in and processing
data to prepare for training in the CREDIT framework.
data_loading() in particular is the main function for 
preparing data for training. Also includes some general 
calculation functions (such as weed speed).
"""

import numpy as np
from typing import Tuple,  Dict
from netCDF4 import Dataset
from datetime import datetime

from path_to_raw_datasets import path_to_raw_datasets, get_var_shortname, get_fn
from moisture_calculations import calculate_q_total_upper_level
from transform_grid import interpolate_data

# Define the climate index variables
index_variables = [
    'enso',
    'amo',
    'nao',
    'pdo',
    'iod'
]

# Define the variables located in the GLDAS2 dataset
gldas_variables = [
    'temperature',
    'pressure',
    'precipitation',
    'precipitation_7day',
    'precipitation_14day',
    'precipitation_30day',
    'evaporation',
    'potential_evaporation',
    'radiation',
    'wind_speed',
    'soil_moisture_1', # BE AWARE: SM depths BETWEEN GLDAS and ERA5 are DIFFERENT
    'soil_moisture_2',
    'soil_moisture_3',
    'soil_moisture_4',
    'sesr',
    'fdii_1',
    'fdii_2',
    'fdii_3',
    'fdii_4'
]

# Define the variables in the IMERG dataset
imerg_variables = [
    'precipitation',
    'precipitation_7day',
    'precipitation_14day',
    'precipitation_30day'
]

# Define the variables in the MODIS dataset
modis_variables = [
    'ndvi',
    'evi',
    'lai',
    'fpar'
]

# Define the variables not located in ERA5
not_in_era5 = [
    'enso',
    'amo',
    'nao',
    'pdo',
    'iod',
    'ndvi',
    'evi',
    'lai',
    'fpar' # MODIS variables and climate indices
]

# Re-define the function to load .nc files 
# (this cannot be loaded from inputs_outputs.py without causing circular imports downstream)
def load_nc(
        filename, 
        var
        ) -> Dict[str, np.ndarray]:
    '''
    Load a nc file

    Data is stored in a dictionary with keys var, lat, lon, and time

    Inputs:
    :param filename: Filename of the nc file (path to nc file and name of nc file)
    :param var: Short name of the main variable in dictionary (i.e., the key for the data in the nc file)

    Outputs:
    :param x: Dictionary with loaded nc data
    '''

    # Initialize the dataset
    x = {}

    # Load the .nc file
    with Dataset(filename, 'r') as nc:
        x['lat'] = nc.variables['lat'][:]
        x['lon'] = nc.variables['lon'][:]
        x['time'] = nc.variables['date'][:]

        x[var] = nc.variables[var][:]

    return x

def reduce_spatial_scale(
        data, 
        var, 
        print_progress: bool = True
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Reduce the spatial score of a dataset by a quarter (downscales via averaging) 
    General fast method of reducing .25 degree grid to 1 degree

    Inputs:
    :param data: dictionary containing data reduced (np.ndarray with shape time x lat x lon), 
                 and lat and lon data (keys 'lat' and 'lon' and both are np.ndarrays with shape lat x lon)
    :param var: Short name of the variable data in data (i.e., the variable key for dictionary in data)
    :param print_progress: Boolean indicating whether to output progress in scale reduction

    Outputs:
    :param data_reduced: The variable data at the reduced spatial scale (np.ndarray with shape time x lat_small x lon_small)
    :param lat_reduced: Latitude labels at the reduced spatial scale (np.ndarray with shape lat_small x lon_small)
    :param lon_reduced: Longitude labels at the reduced spatial scale (np.ndarray with shape lat_small x lon_small)
    '''

    # reduce scale of data to 1 deg x 1 deg
    lat = data['lat'][::4,::4]
    lon = data['lon'][::4,::4]

    # Remove -90 degree latitude for an even size of 180
    lat = lat[:-1,:]
    lon = lon[:-1,:]

    # Initialize the reduced dataset
    T, I, J = data[var].shape
    I, J = lat.shape

    data_reduced = np.ones((T, I, J)) * np.nan
    
    # Reverse the latitude labels to go from lowest to highest
    lat_tmp = lat[:,0][::-1]

    for j in range(J):
        # Print progress if required
        if print_progress & (np.mod((j*100/J),10) == 0):
            print('%4.2f through variable reduction with %s'%((j*100/J), var))

        # Special case fo the end of the longitude grid
        if j == (J-1):
            # At the end of the grid, collect all longitude points that remain
            ind_lon = np.where(data['lon'][0,:] >= lon[0,j])[0]
        else:
            # Collect all longitude points between the two sets of reduced grids
            ind_lon = np.where((data['lon'][0,:] >= lon[0,j]) & (data['lon'][0,:] < lon[0,j+1]))[0]

        # Perform an average along the longitude line (reduce longitude scale )
        tmp = np.nanmean(data[var][:,:,ind_lon], axis = -1)

        for i in range(I):
            # Special case fo the end of the latitude grid
            if i == (I-1):
                # At the end of the grid, collect all latitude points that remain
                ind_lat = np.where(data['lat'][:,0] >= lat_tmp[i])[0]
            else:
                # Collect all latitude points between the two sets of reduced grids
                ind_lat = np.where((data['lat'][:,0] >= lat_tmp[i]) & (data['lat'][:,0] < lat_tmp[i+1]))[0]
        
            # The average all lat points between the reduced grids makes the average value between the reduced grid
            data_reduced[:,i,j] = np.nanmean(tmp[:,ind_lat], axis = -1)

    # Output any NaNs or 0s if required
    if print_progress:
        print("NaNs found in %s: "%var, np.sum(np.isnan(data_reduced)))
        print("0s found in %s: "%var, np.sum(data_reduced == 0))

    return data_reduced, lat, lon

def running_sum(
        data, 
        N: int = 7
        ) -> np.ndarray:
    '''
    Calculate an N day (end point) running sum 
    Sum is performed along the time axis

    Inputs:
    :param data: Data to perform the running sum over (np.ndarray with shape time x lat x lon)
    :param N: Length of the running sum

    Outputs:
    :param run_sum: The endpoint running sum of data (np.ndarray of shape time x lat x lon)
    '''

    # Initialize the running sum dataset
    T, I, J = data.shape

    run_sum = np.zeros((T, I, J))


    for i in range(I):
        for j in range(J):
            # Convolve the data with arrays of 1s to determine the sum 
            # (end point is determined by the N-1: indexing)
            run_sum[:,i,j] = np.convolve(data[:,i,j], np.ones((N,)))[(N-1):]

    return run_sum


def data_loading(
        year, 
        var, 
        sname, 
        datasets = 'era5', 
        reduce_scale = True, 
        level = None
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Load in and process 1 year of data from a given dataset

    Inputs:
    :param year: The year to load the data for
    :param var: Long name of the variable being loaded
    :param sname: Dictionary key of variable in the .nc files
    :param reduce_scale: Boolean indicating whether to reduces spatial scale of data by quarter
    :param level: Index of the elevation level for upper air variables (e.g., if 200 mb is first, its index/level = 0)

    Outputs:
    :param data: 1 year of processed data (np.ndarray of shape time x lat x lon format)
    :param days: List of date values for each time step in data 
                 (e.g., Jan 1, Feb. 1, etc. have values of 1, Jan. 2, Feb. 2, etc. have values of 2)
    :param months: List of month values for each time step in data 
                   (e.g., Jan 1, Jan 2, etc. all have values of 1)
    '''
    
    if level is None:
        print('Working on %s on %d'%(var, year))
    else:
        print('Working on %s at %d mb on %d'%(var, level, year))

    # Note total specific humidity and surface wind speed to not have datafiles, but are calculated from other variables
    if var == 'total_specific_humidity':
        # Load multiple datasets to determin the total specific humidity
        data = calculate_q_total_upper_level(year, level) # q_tot = q + cloud_liquid_content + cloud_ice_content

    elif var == 'wind_speed':
        # Load in u and v components and calculate the wind speed
        data = calculate_wind_speed(year) # ws = sqrt(surfcae_u**2 + surface_v**2)

    elif var in not_in_era5:
        # Load a placeholder set of data with the grid information for the interpolation step
        path = path_to_raw_datasets('temperature', 'era5')
        filename = get_fn('temperature', year)
        data = load_nc('%s/%s.nc'%(path, filename), 'tair')

    else:
        # Get filename and directory path information
        path = path_to_raw_datasets(var, 'era5', level = level)
        filename = get_fn(var, year, level = level)

        # Note there are not any files for 7 day precip accumulation, 14 day accumulation, etc. Load preciptation instead
        if 'precipitation' in var:
            data = load_nc('%s/%s.nc'%(path, filename), 'tp')
        else:
            # Load the data
            data = load_nc('%s/%s.nc'%(path, filename), sname)

    # Add data from other, non-ERA5 datasets if necessary
    if np.invert(datasets == 'era5'):
        if (datasets == 'few'):
            # For 'few', only add the reanalyses and index data
            if var in gldas_variables:
                dataset = 'gldas'
            elif var in index_variables:
                dataset = 'climate_indices'

        if (datasets == 'all'):
            # Include all reanalyses and satellite data
            # Note, done this way, any variables in reanalyses and satellite data will use satellite (i.e., satellite data has priority)
            if var in modis_variables:
                    dataset = 'modis'
            elif var in imerg_variables:
                dataset = 'imerg'
            elif var in gldas_variables:
                dataset = 'gldas'
            elif var in index_variables:
                dataset = 'climate_indices'
            else:
                dataset = None
                
        # Load in the data from the other dataset and interpolate to the ERA5 grid
        if dataset is not None:
            print('Loading and interpolation data from %s for %s and for year %d'%(dataset, var, year))
            if var in not_in_era5:
                sname_tmp = 'tair'
            elif 'precipitation' in var:
                sname_tmp = get_var_shortname('precipitation')
            else:
                sname_tmp = sname
            # sname_tmp = 'tair' if var in not_in_era5 else sname           
            data = interpolate_data(data, sname_tmp, dataset, var, year, resolution = 0.25)

    # Construct datetime information
    dates_current_year = np.array([datetime.fromisoformat(day) for day in data['time']])

    days = np.array([day.day for day in dates_current_year])
    months = np.array([day.month for day in dates_current_year])

    # Perform a running sum of precipitation if needed
    if var == 'precipitation_7day':
        # 7 day end point accumulation
        sname = get_var_shortname(var)
        data[sname] = running_sum(data['tp'], N = 7)

    elif var == 'precipitation_14day':
        # 14 day end point accumulation
        sname = get_var_shortname(var)
        data[sname] = running_sum(data['tp'], N = 14)

    elif var == 'precipitation_30day':
        # 30 day endpoint accumulation
        sname = get_var_shortname(var)
        data[sname] = running_sum(data['tp'], N = 30)

    # Reduce the spatial scale if desired
    if reduce_scale:
        data, _, _ = reduce_spatial_scale(data, sname, print_progress =  False)
    else:
        # Note at quarter degree resolution, the grid is 721 x 1440, which is difficult to work with since 721 only has two primes (721=7*103)
        # So remove the -90 degree point to make the grid easier (720 = 2^4*3^2*5)
        data = data[sname][:,:-1,:]

    # print some information on the data (useful to check everything is correct)
    print(year, np.nanmin(data), np.nanmax(data), np.nanmean(data))

    return data, days, months

def calculate_wind_speed(year) -> Dict[str, np.ndarray]:
    '''
    Calculate the wind speed from loaded u and v components

    Inputs:
    :param year: The year of data to load u and v wind components for

    Outputs:
    :param ws: Dictionary with calculated surface wind speed data for one year 
               (key is 'ws', np.ndarray with shape time x lat x lon),
               gridded latitude and longitude (keys 'lat' and 'lon' and 
               np.ndarrays with shape lat x lon), and time information
               (key 'time', np.ndarray with shape time)
    '''

    # Get the file information
    sname_ws = get_var_shortname('wind_speed')
    path = path_to_raw_datasets('wind_speed', 'era5')
    variables = ['wind_speed_u', 'wind_speed_v']

    ws = {}
    for variable in variables:
        # Get the wind component file information
        sname = get_var_shortname(variable)
        fn = get_fn(variable, year)

        # Load wind component data
        data = load_nc('%s/%s.nc'%(path, fn), sname)

        # If the WS is not initialized, initialize it
        if sname_ws not in ws.keys():
            ws[sname_ws] = data[sname]**2
        else: 
            # Sum the squared wind components
            ws[sname_ws] = ws[sname_ws] + (data[sname]**2)

        # Add the latitude, longitude, and time information to the full dataset
        ws['lat'] = data['lat']; ws['lon'] = data['lon']; ws['time'] = data['time']

    # Wind speed is the square root of the sume of squared components
    ws[sname_ws] = np.sqrt(ws[sname_ws])

    return ws


def calculate_surface_potential(
        start_year: int = 2000, 
        end_year: int = 2023, 
        reduce_scale: bool = False
        ) -> np.ndarray:
    '''
    Load in and calculate the surface geopotential, averaged in time

    Inputs:
    :param start_year: The starting year of the data to load 
                       (with end_year this makes the climatological time span for mean of Z_surf)
    :param end_year: The ending year of the data to load 
                     (with start_year this makes the climatological time span for mean of Z_surf)
    :param reduce_scale: Boolean indicating whether to reduces spatial scale of data by quarter

    Outputs:
    :param Z_surf: Time averaged surface geopotential (np.ndarray with shape lat x lon)
    '''

    # Load in a test dataset (includes leap year)
    test_path = path_to_raw_datasets('temperature', 'era5')
    test_fn = get_fn('temperature', 2000)
    test_sname = get_var_shortname('temperature')
    test = {}
    with Dataset('%s/%s.nc'%(test_path, test_fn), 'r') as nc:
        test[test_sname] = nc.variables[test_sname][:]
        test['lat'] = nc.variables['lat'][:]
        test['lon'] = nc.variables['lon'][:]
        time = nc.variables['date'][:]

    # Reduce the spatial scale of the test data if required
    if reduce_scale:
        test, lat, lon = reduce_spatial_scale(test, test_sname)
    else:
        # Note at quarter degree resolution, the grid is 721 x 1440, which is difficult to work with since 721 only has two primes (721=7*103)
        # So remove the -90 degree point to make the grid easier (720 = 2^4*3^2*5)
        lat = test['lat'][:,:-1]; lon = test['lon'][:,:-1]
        test = test[test_sname][:,:-1,:]
        
    T, I, J = test.shape

    # Determine all the years to load
    years = np.arange(start_year, end_year+1)

    # Initialize the dataset
    N = 0
    Z_surf = np.zeros((I, J))

    for year in years:
        # Determine file and path information for temperature and pressure
        tair_path = path_to_raw_datasets('temperature', 'era5')
        tair_fn = get_fn('temperature', year)
        tair_sname = get_var_shortname('temperature')

        sp_path = path_to_raw_datasets('pressure', 'era5')
        sp_fn = get_fn('pressure', year)
        sp_sname = get_var_shortname('pressure')

        # Load temperature pressure for the given year
        tair = load_nc('%s/%s.nc'%(tair_path, tair_fn), tair_sname)
        sp = load_nc('%s/%s.nc'%(sp_path, sp_fn), sp_sname)

        # Reduce the spatial scale of the temperature pressure if desired
        if reduce_scale:
            tair, _, _ = reduce_spatial_scale(tair, tair_sname)
            sp, _, _ = reduce_spatial_scale(sp, sp_sname)
        else:
            # Note at quarter degree resolution, the grid is 721 x 1440, which is difficult to work with since 721 only has two primes (721=7*103)
            # So remove the -90 degree point to make the grid easier (720 = 2^4*3^2*5)
            tair = tair[tair_sname][:,:-1,:]; sp = sp[sp_sname][:,:-1,:]

        # Surface geopotential calculations
        for t in range(tair.shape[0]):
            tmp = np.log(100000/sp[t,:,:])*287*tair[t,:,:]/9.8 # ln(p_0/p) * R_d * T /g

            # Add the calculated Z to a sum
            Z_surf = np.nansum([Z_surf, tmp], axis = 0)
            N = N + 1

    # Divide by the number of time steps to get the mean
    Z_surf = Z_surf/N

    return Z_surf
