import numpy as np
from netCDF4 import Dataset
from datetime import datetime

from path_to_raw_datasets import path_to_raw_datasets, get_var_shortname, get_fn
from moisture_calculations import calculate_q_total_upper_level
from transform_grid import interpolate_data

index_variables = [
    'enso',
    'amo',
    'nao',
    'pdo',
    'iod'
]
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
imerg_variables = [
    'precipitation',
    'precipitation_7day',
    'precipitation_14day',
    'precipitation_30day'
]
modis_variables = [
    'ndvi',
    'evi',
    'lai',
    'fpar'
]
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

# Reproduce the load_nc function here to remove circular imports
def load_nc(filename, var):
    '''
    Load a nc file
    '''

    x = {}

    with Dataset(filename, 'r') as nc:
        x['lat'] = nc.variables['lat'][:,:]
        x['lon'] = nc.variables['lon'][:,:]
        x['time'] = nc.variables['date'][:]

        x[var] = nc.variables[var][:,:,:]

    return x

def reduce_spatial_scale(data, var, print_progress = True):
    '''
    Fast and quick (in terms of arguments and writing) method of reducing .25 degree grid to 1 degree

    data: dictionary containing 3D data, and 2D lat and lon
    var: short name of the 3D data in data
    '''

    # reduce scale of data to 1 deg x 1 deg
    lat = data['lat'][::4,::4]
    lon = data['lon'][::4,::4]

    # Remove -90 degree latitude for an even size of 180
    lat = lat[:-1,:]
    lon = lon[:-1,:]

    T, I, J = data[var].shape
    I, J = lat.shape

    data_reduced = np.ones((T, I, J)) * np.nan
    
    lat_tmp = lat[:,0][::-1]
    for j in range(J):
        if print_progress & (np.mod((j*100/J),10) == 0):
            print('%4.2f through variable reduction with %s'%((j*100/J), var))
        if j == (J-1):
            # At the end of the grid, collect all longitude points that remain
            ind_lon = np.where(data['lon'][0,:] >= lon[0,j])[0]
        else:
            # Collect all longitude points between the two sets of reduced grids
            ind_lon = np.where((data['lon'][0,:] >= lon[0,j]) & (data['lon'][0,:] < lon[0,j+1]))[0]

        tmp = np.nanmean(data[var][:,:,ind_lon], axis = -1)

        for i in range(I):

            if i == (I-1):
                # At the end of the grid, collect all latitude points that remain
                ind_lat = np.where(data['lat'][:,0] >= lat_tmp[i])[0]
            else:
                # Collect all latitude points between the two sets of reduced grids
                ind_lat = np.where((data['lat'][:,0] >= lat_tmp[i]) & (data['lat'][:,0] < lat_tmp[i+1]))[0]
        
            # The average all lat and lon points between the reduced grids makes the average value between the reduced grid
            data_reduced[:,i,j] = np.nanmean(tmp[:,ind_lat], axis = -1)

    if print_progress:
        print("NaNs found in %s: "%var, np.sum(np.isnan(data_reduced)))
        print("0s found in %s: "%var, np.sum(data_reduced == 0))
    #lat[lat == 0] = 0.001
    #lon[lon == 0] = 0.001
    #data_reduced[data_reduced == 0] = 0.001
    #data_reduced[np.isnan(data_reduced)] = 0.001
    return data_reduced, lat, lon

def running_sum(data, N = 7):
    '''
    Calculate an N day (end point) running sum 
    '''

    T, I, J = data.shape

    run_sum = np.zeros((T, I, J))
    for i in range(I):
        for j in range(J):
            run_sum[:,i,j] = np.convolve(data[:,i,j], np.ones((N,)))[(N-1):]

    return run_sum


def data_loading(year, var, sname, datasets = 'era5', reduce_scale = True, level = None):
    '''
    Load in and process 1 year of ERA5

    Inputs:
    :param year: int, The year of the variable being loaded
    :param var: str, Long name of the variable being loaded
    :param sname: str, dictionary key of variable in the .nc files
    :param reduce_scale: bool, reduces spatial scale of data by quarter if true
    :param level: int, index of the elevation level for upper air variables (e.g., if 200 mb is first, its index/level = 0)

    Outputs:
    :param data: np.array, 1 year of processed ERA5 data (time x lat x lon format)
    :param days: np.array, list of dates for each time step in data (e.g., Jan 1, Feb. 1, etc. have values of 1, Jan. 2, Feb. 2, etc. have values of 2)
    :param months: np.array, list of months for each time step in data (e.g., Jan 1, Jan 2, etc. all have values of 1)
    '''
    
    if level is None:
        print('Working on %s on %d'%(var, year))
    else:
        print('Working on %s at %d mb on %d'%(var, level, year))

    # Note total specific humidity and surface wind speed to not have datafiles, but are calculated from other variables
    if var == 'total_specific_humidity':
        data = calculate_q_total_upper_level(year, level) # q_tot = q + cloud_liquid_content + cloud_ice_content
    elif var == 'wind_speed':
        data = calculate_wind_speed(year) # ws = sqrt(surfcae_u**2 + surface_v**2)
    elif var in not_in_era5:
        # Load a filler set of data with the grid information for the interpolation step
        path = path_to_raw_datasets('temperature', 'era5')
        filename = get_fn('temperature', year)
        data = load_nc('%s/%s.nc'%(path, filename), 'tair')
    else:
        # Get filename and location information
        path = path_to_raw_datasets(var, 'era5', level = level)
        filename = get_fn(var, year, level = level)

        # Note there are not any files for 7 day precip accumulation, 14 day accumulation, etc. Load preciptation instead
        if 'precipitation' in var:
            data = load_nc('%s/%s.nc'%(path, filename), 'tp')
        else:
            # Load the data
            data = load_nc('%s/%s.nc'%(path, filename), sname)

    # Add data from other, non-ERA5 datasets?
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

    # Perform a running sum of precipitation?
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

def calculate_wind_speed(year):
    '''
    Calculate the wind speed
    '''

    # Get the file information
    sname_ws = get_var_shortname('wind_speed')
    path = path_to_raw_datasets('wind_speed', 'era5')
    variables = ['wind_speed_u', 'wind_speed_v']

    ws = {}
    for variable in variables:
        sname = get_var_shortname(variable)
        fn = get_fn(variable, year)

        # Load data
        data = load_nc('%s/%s.nc'%(path, fn), sname)

        # If the WS is not initialized, initialize it
        if sname_ws not in ws.keys():
            ws[sname_ws] = data[sname]**2
        else: 
            # Sum the squared wind components
            ws[sname_ws] = ws[sname_ws] + (data[sname]**2)

        ws['lat'] = data['lat']; ws['lon'] = data['lon']; ws['time'] = data['time']

    # Wind speed is the square root of the sume of squared components
    ws[sname_ws] = np.sqrt(ws[sname_ws])

    return ws


def calculate_surface_potential(start_year = 2000, end_year = 2023, reduce_scale = False):
    '''
    Load in and calculate the surface geopotential, averaged in time
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

    if reduce_scale:
        test, lat, lon = reduce_spatial_scale(test, test_sname)
    else:
        # Note at quarter degree resolution, the grid is 721 x 1440, which is difficult to work with since 721 only has two primes (721=7*103)
        # So remove the -90 degree point to make the grid easier (720 = 2^4*3^2*5)
        lat = test['lat'][:,:-1]; lon = test['lon'][:,:-1]
        test = test[test_sname][:,:-1,:]
        
    T, I, J = test.shape

    years = np.arange(start_year, end_year+1)

    N = 0
    Z_surf = np.zeros((I, J))
    for year in years:
        tair_path = path_to_raw_datasets('temperature', 'era5')
        tair_fn = get_fn('temperature', year)
        tair_sname = get_var_shortname('temperature')

        sp_path = path_to_raw_datasets('pressure', 'era5')
        sp_fn = get_fn('pressure', year)
        sp_sname = get_var_shortname('pressure')

        tair = load_nc('%s/%s.nc'%(tair_path, tair_fn), tair_sname)
        sp = load_nc('%s/%s.nc'%(sp_path, sp_fn), sp_sname)

        if reduce_scale:
            tair, _, _ = reduce_spatial_scale(tair, tair_sname)
            sp, _, _ = reduce_spatial_scale(sp, sp_sname)
        else:
            # Note at quarter degree resolution, the grid is 721 x 1440, which is difficult to work with since 721 only has two primes (721=7*103)
            # So remove the -90 degree point to make the grid easier (720 = 2^4*3^2*5)
            tair = tair[tair_sname][:,:-1,:]; sp = sp[sp_sname][:,:-1,:]

        # Surface geopotential calculations
        for t in range(tair.shape[0]):
            tmp = np.log(100000/sp[t,:,:])*287*tair[t,:,:]/9.8
            Z_surf = np.nansum([Z_surf, tmp], axis = 0)
            N = N + 1

    # Divide by the number of time steps to get the mean
    Z_surf = Z_surf/N

    return Z_surf
