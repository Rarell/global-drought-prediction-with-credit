"""General grid transformation to move data from one grid to another

Provides a set of functions designed to move from
one grid onto another to resolve inconsistencies
in different resolutions between different datasets

Key function that brings all others together is
interpolate_data()
"""

import gc
import numpy as np
import xarray

import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from typing import Tuple, List, Dict
from glob import glob
from datetime import datetime, timedelta
from netCDF4 import Dataset

from path_to_raw_datasets import path_to_raw_datasets, get_var_shortname, get_fn

# Define a list of variables not present in the ERA5 dataset
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

def interpolate_data(
        data_grid, 
        data_grid_sname, 
        dataset, 
        variable, 
        year, 
        resolution: float = 0.25
        ) -> Dict[str, np.ndarray]:
    '''
    Load in data from one of the non-ERA5 datasets and intopolate it to the same grid and timescale
    as the ERA5 data

    Input:
    :param data_grid: Dictionary containing the ERA5 data for reference, latitude and longitude (keys: lat and lon) 
                      grid to interpolate to, and timestamps (key: time) to interpolate to
    :param data_grid_sname: Key to the ERA5 data in data_grid
    :param dataset: Name of the dataset to be loaded and interpolated (gldas, imerg, or modis)
    :param variable: Name of the variable to be loaded and interpolated.
    :param year: The year of the data being interpolated.
    :param resolution: Resolution scale (in degrees latitude/longitude) to interpolate to

    Outputs:
    :param data_new: Dictionary containing interpolated data from the loaded dataset. Key to data is data_grid_sname variable is in ERA5.
                     Also has keys lat, lon, and time for latitude, longitude, and timestamps respectively.
    '''

    # Get the dictionary key for the data, as well as the path to and filename(s) of the data
    sname_old = 'precip' if 'precipitation' in variable else get_var_shortname(variable, reanalysis = dataset)
    path = path_to_raw_datasets(variable, reanalysis = dataset)
    filenames = get_fn(variable, year, reanalysis = dataset)

    # Convert ERA5 units to the same as data being loaded 
    # Ensures unit consistency with the loaded dataset
    # (in some cases, ERA5 is used to replace missing data)
    if variable not in not_in_era5:
        data_grid[data_grid_sname] = unit_converter(data_grid[data_grid_sname], variable, dataset)

    # Load and interpolate GLDAS2 data
    if dataset == 'gldas':
        if year == 2000: # GLDAS2 does noth ave data for 2000; return the original (ERA5) dataset
            return data_grid
        
        # Load GLDAS2 data and replace bad data with NaNs
        data = load_nc('%s/%s.nc'%(path, filenames), sname_old)
        data[sname_old][data[sname_old] < -900] = np.nan

        if variable == 'potential_evaporation':
            # GLDAS PET is strangely in units that are inconsistent with ET, P, etc. (in W m^-2); Convert them same units as the rest
            data[sname_old] = data[sname_old] / (2.5e6) # Division by latent heat of vaporization yields conversion of W m^-2 = J s^-1 m^-2 -> kg s^-1 m^-2

        elif 'soil_moisture' in variable:
            # Deal with some anomalously high SM values in the arctic
            data[sname_old][data[sname_old] >= 55] = 0

        # Interpolate GLDAS2 data onto the ERA5 grid
        grid_new = interpolate_to_new_grid(data['lat'], 
                                           data['lon'], 
                                           data[sname_old], 
                                           data_grid['lat'], 
                                           data_grid['lon'], 
                                           dataset = dataset, 
                                           resolution = resolution)

        # Process GLDAS data (this replaces NaNs with ERA5 values, particularly for locations south of 60 deg S)
        grid_new = gldas_postprocess(grid_new, data_grid[data_grid_sname])

        # Convert moisture units to mm to be consistent with IMERG precipitation
        if ('precipitation' in variable) | (variable == 'evaporation') | (variable == 'potential_evaporation'):
            grid_new = grid_new * (60*60*24)/1000 * 1000 # Multiplties 60*60*24 seconds day^-1 (daily accumulation),
                                                         # 1000 mm m^-1, and divides by the density of water (1000 kg m^-3)
                                                         # Converts units kg s^-1 m^-2 -> mm/day

    # Load and interpolate IMERG data
    elif dataset == 'imerg':
        # Load IMERG precipitation data and replace any labeled bad data
        data = load_nc('%s/%s.nc'%(path, filenames), sname_old)
        data[sname_old][data[sname_old] < -900] = np.nan

        # Interpolate IMERG data to ERA5 grid
        grid_new = interpolate_to_new_grid(data['lat'], 
                                           data['lon'], 
                                           data[sname_old], 
                                           data_grid['lat'], 
                                           data_grid['lon'], 
                                           dataset = dataset, 
                                           resolution = resolution)

        # Process IMERG data (replaces any NaNs with ERA5 data)
        grid_new = imerg_postprocess(grid_new, data_grid[data_grid_sname])

       # grid_new = grid_new / 1000 # Convert to m; CREDIT seems to have a problem with mm precipitation predictions (resolved)

    # Load and interpolate MODIS data
    elif dataset == 'modis':
        # Note MODIS data is stored in multiple .nc files per year; collect all those files
        files = glob('%s/%s*.nc'%(path, filenames), recursive = True)

        # Initialize the dataset
        data = {}
        data[sname_old] = []
        data['time'] = []

        # Load in the data for each time stamp than add them to the dataset
        for file in np.sort(files):
            # Load data
            tmp = load_nc(file, sname_old)
            # Add data to list
            data[sname_old].append(tmp[sname_old])
            data['lat'] = tmp['lat']; data['lon'] = tmp['lon']
            data['time'].append(datetime.fromisoformat(tmp['time'][0])) # Convert time to datetimes

        # Also insert the last time step from the previous year for time interpolations that don't start on Jan. 1
        filenames = get_fn(variable, year-1, reanalysis = dataset) # Get the files for the previous year
        files = glob('%s/%s*.nc'%(path, filenames), recursive = True)

        # Collect only last timestamp in the previous year
        file = np.sort(files)[-1]
        
        # Load the file and insert (adds to the start of the list)
        tmp = load_nc(file, sname_old)
        data[sname_old].insert(0, tmp[sname_old])
        data['time'].insert(0, datetime.fromisoformat(tmp['time'][0]))

        # Convert data into an array and remove bad data 
        # Note the MODIS data collected only collected data for land, so all sea values are NaN
        data[sname_old] = np.array(data[sname_old]).astype(np.float32)
        data[sname_old][data[sname_old] < -900] = np.nan
        data[sname_old][data[sname_old] > 1e9] = np.nan # Also removes fill values
        data['time'] = np.array(data['time'])
        
        # Make the date time arrays to interpolate to
        new_time = np.array([datetime.fromisoformat(date) for date in data_grid['time']])

        # Interpolate MODIS data onto the ERA5 grid
        grid_new = interpolate_to_new_grid(data['lat'], 
                                           data['lon'], 
                                           data[sname_old], 
                                           data_grid['lat'], 
                                           data_grid['lon'], 
                                           dataset = dataset, 
                                           resolution = resolution)

        # Interpolate MODIS data onto the daily time scale (using the most recent entry for each day)
        grid_new = interpolate_timeseries(grid_new, 
                                          data_grid[data_grid_sname], 
                                          data['time'], 
                                          new_time)

        # Process the MODIS data (replaces NaNs with 0s)
        grid_new = modis_postprocess(grid_new, data_grid['lat'])

        # Modify the sname, if necessary
        if variable in not_in_era5:
            data_grid_sname = get_var_shortname(variable, reanalysis = 'modis')

    # Load and interpolate climate index data
    elif dataset == 'climate_indices':
        # Determine if the timestamps are already prepared for the given index
        if (variable == 'pdo') | (variable == 'iod') | (variable == 'nao'):
            timestamps_prepared = False
        else:
            timestamps_prepared = True

        # Load climate index time seris
        data = {}
        # Special case for ENSO since the .csv file has multiple time series 
        # (this collects ENSO3.4 SST anomalies)
        if variable == 'enso':
            data[sname_old], data['time'] = load_index_data('%s.csv'%filenames, 
                                                            timestamps_prepared = timestamps_prepared, 
                                                            enso = True, 
                                                            path = path)

            # Select out the ENSO3.4 SSTa's
            # NOTE, this does leave the option to select other SSTs or SSTa's for other ENSO zones
            data[sname_old] = data[sname_old][:,5]
        
        # Collect the index data for non-ENSO indices
        else:
            data[sname_old], data['time'] = load_index_data('%s.csv'%filenames, 
                                                            timestamps_prepared = timestamps_prepared, 
                                                            enso = False, 
                                                            path = path)
        
        # Select data for the singular year
        years = np.array([date.year for date in data['time']])
        ind = np.where(years == year)[0]
        ind = np.insert(ind, 0, ind[0]-1) # Add the last timestamp of the previous year

        data[sname_old] = data[sname_old][ind]; data['time'] = np.array(data['time'])[ind]

        # Make the reference time axis into datetimes for comparison
        new_time = np.array([datetime.fromisoformat(date) for date in data_grid['time']])

        # Interpolate climate index to onto a map (each point will have the same entry)
        grid_new = point_to_grid(data[sname_old], data_grid['lat'])

        # Interpolate index data onto a daily set
        grid_new = interpolate_timeseries(grid_new, 
                                          data_grid[data_grid_sname], 
                                          data['time'], 
                                          new_time)

        grid_new[grid_new <= -90] = 0 # Zero out bad data points to prevent notable errors in the ML training

        # Modify the variable sname since climate indices don't have ERA5 entries
        data_grid_sname = get_var_shortname(variable) 

    # Create dictionary for the new data, add the interpolated data and latitude, longitude, and time
    data_new = {}
    data_new[data_grid_sname] = grid_new; data_new['lat'] = data_grid['lat']; data_new['lon'] = data_grid['lon']; data_new['time'] = data_grid['time']

    return data_new

def interpolate_to_new_grid(
        lat_old, 
        lon_old, 
        data, 
        lat_new, 
        lon_new, 
        dataset: str = 'gldas', 
        resolution: float = 0.1
        ) -> np.ndarray:
    '''
    Interpolate from an old geospatial grid system onto a new grid geospatial system 
    via averaging (assumed the new grid system is coarser) 

    Inputs:
    :param lat_old: Latitudes of the old grid system (np.ndarray with shape lat_old or lat_old x lon_old)
    :param lon_old: Longitudes of the old grid system (np.ndarray with shape lon_old or lat_old x lon_old)
    :param data: Data to be interpolated (np.ndarray with shape time x lat_old x lon_old)
    :param lat_new: Latitudes of the new grid system (np.ndarray with shape lat_new or lat_new x lon_new)
    :param lon_new: Longitudes of the new grid system (np.ndarray with shape lon_new or lat_new x lon_new)
    :param dataset: Dataset that data comes from. Must be gldas, imerg, or modis (default = gldas)
    :param resolution: Spatial resolution (in degrees lat/lon) of the new grid (default = 0.1)

    Outputs:
    :param data_new_grid: data interpolated unto the new spatial grid (np.ndarray with shape time x lat_new x lon_new)
    '''
    
    # Make the latitude and longitudes 2D if they are not already
    if (len(lat_old.shape) < 2) & (len(lon_old.shape) < 2):
        lon_old, lat_old = np.meshgrid(lon_old, lat_old)

    if (len(lat_new.shape) < 2) & (len(lon_new.shape) < 2):
        lon_new, lat_new = np.meshgrid(lon_new, lat_new)

    # Make sure in the old longitude system is consistent with the new grid 
    # (i.e., they both go from -180 to 180 or both go from 0 to 360)
    if (lon_old < 0).any() & (lon_new >= 0).all():
        lon_old = np.where(lon_old < 0, lon_old + 360, lon_old)

    elif (lon_new < 0).any() & (lon_old >= 0).all():
        lon_old = np.where(lon_old > 180, lon_old - 360, lon_old)
    
    # Initialize new grid
    T, I_old, J_old = data.shape

    I, J = lat_new.shape

    data_new_grid = np.ones((T, I, J)) * np.nan

    # Perform the interpolation for each time period in the old grid 
    # (interpolating time scale is in a different function)
    for t in range(T):
        # print(t/T*100)
        for n, ind in enumerate(range(lat_new.shape[0])):
            # Get all latitudes in the old system between this and the next latitudes in the new grid
            lat_ind = np.where( (lat_old[:,0] >= lat_new[n,0]) & (lat_old[:,0] < (lat_new[n,0] + resolution)) )[0]

            # If no latitudes are found in the old system, skip (e.g., GLDAS does not have data south of 60 deg S)
            if len(lat_ind) < 1:
                continue

            # IMERG data is transposed compared to the other datasets
            if dataset == 'imerg':
                tmp = data[t,:,:].T
            else:
                tmp = data[t,:,:]

            # Average the data and longitude along the latitudes found
            data_tmp = np.nanmean(tmp[lat_ind,:], axis = 0)
            lon_tmp = np.nanmean(lon_old[lat_ind,:], axis = 0)
            
            # Convert to an xarray Dataset to use interpolation later on 
            # (faster and more effective than averaging)
            ds = xarray.Dataset(data_vars = dict(data = (['x'], data_tmp)), 
                                coords = dict(x = (['x'], lon_tmp)) )
            
            # Use linear interpolation to go from the old longitudes to the new one
            ds_interp = ds.interp(coords = dict(x = (['lon'], lon_new[n,:])) )
            
            # Add the interpolated data onto the new dataset
            data_new_grid[t,ind,:] = ds_interp.data.values[:] #np.nanmean(tmp, axis = 0)
    
    # Remove xarray Datasets to free up memory
    del ds, ds_interp
    gc.collect()

    return data_new_grid

def point_to_grid(point, grid) -> np.ndarray:
    '''
    Interpolate data from a time series (i.e., from a single point) onto a geospatial grid
    This is performed by repeating the time series on each grid point

    Inputs:
    :param point: Time series data to put on a map (np.ndarray with shape time)
    :param grid: Gridded data to put the time series onto (np.ndarray with shape lat x lon or tmp x lat x lon).

    Outputs:
    :param grid_new: Interpolated timeseries data (np.ndattay with shape time x lat x lon)
    '''

    # Get the grid size
    if len(grid.shape) == 2:
        I, J = grid.shape
    else:
        _, I, J = grid.shape

    # Initialize the new grid
    grid_new = np.ones((point.size, I, J))

    # For each point on the new grid, add the time series data
    for i in range(I):
        for j in range(J):
            grid_new[:,i,j] = point

    return grid_new

def interpolate_timeseries(
        grid_old, 
        grid_new, 
        dates_old, 
        dates_new
        ) -> np.ndarray:
    '''
    Interpolate from a coarser time series onto a finer one by using the most recent entry in the data.

    Inputs:
    :param grid_old: Data with smaller time axis to be interpolated (np.ndarray with shape time_old x lat x lon)
                     grid_old.shape[1,2] must equal grid_new.shape[1,2]
    :param grid_new: Data with the new time dimension to be interpolated to (np.ndarray with shape time_new x lat x lon) 
                     grid_new.shape[1,2] must equal grid_old.shape[1,2]
    :param dates_old: Array of datetimes labels for the grid_old.shape[0] axis (np.ndarray with shape time_old)
    :param dates_new: Array of datetimes labels for the grid_new.shape[0] axis (np.ndarray with shape time_new)

    Outputs:
    :param new_ts: Interpolated grid_old interpolated to the grid_new time axis (np.ndarray with shape time_new x lat x lon)
    '''

    # Get the new grid size and intitialize it.
    T, I, J = grid_new.shape

    new_ts = np.ones((T, I, J))


    for t, date in enumerate(dates_new):
        # Find the most recent time stamp from grid_old
        ind = np.where(date >= dates_old)[0][-1]
        
        # Use the most recent time stamp for the current, finer time resolution
        new_ts[t,:,:] = grid_old[ind,:,:]

    return new_ts

def modis_postprocess(data, lat) -> np.ndarray:
    '''
    Postprocess and fix up the MODIS data

    This removes an issue at ~71 degrees N by replacing them with the average value of the nearest grids
    This replaces missing values with 0

    Inputs:
    :param data: Input MODIS data to process (np.ndarray with shape time x lat x lon)
    :param lat: Latitude labels for the MODIS data (np.ndarray with shape lat x lon)

    Outputs:
    :param data: Processed MODIS data (np.ndarray with shape time x lat x lon)
    '''

    # MODIS data exhibits strange and erroroneous behavior at certain latitudes; 
    # replace them with the average of the surrounding latitudes
    ind = np.where( (lat[:,0] >= 70.75) & (lat[:,0] <= 71.50) )[0]
    tmp = np.array([data[:,ind[0]-1,:], data[:,ind[-1]+1,:]])

    # Replace the erroroneous data with the average of the surrounding latitudes
    for i in ind:
        data[:,i,:] = np.nanmean(tmp, axis = 0)

    # Fill in NaNs (sea values) with 0 (note NaN values must be removed here as they will cause errors later on in ML portion).
    data = np.where(np.isnan(data), 0, data)

    return data

def gldas_postprocess(data, data_replace) -> np.ndarray:
    '''
    Postprocesses GLDAS2 data.

    This replaces missing values with data from another dataset

    Inputs:
    :param data: Input GLDAS data to process (np.ndarray with shape time x lat x lon)
    :param data_replace: Dataset to use to replace missing values (np.ndarray with shape time x lat x lon)

    Outputs:
    :param data: Processed GLDAS data (np.ndarray with shape time x lat x lon)
    '''

    # Note GLDAS2 data only goes down 60 deg S. Replace remaining data to 90 deg S, and any other NaNs with data_replace (ERA5 data).
    # (note NaN values must be removed here as they will cause errors later on in ML training)
    data = np.where(np.isnan(data), data_replace, data)

    return data

def imerg_postprocess(data, data_replace) -> np.ndarray:
    '''
    Postprocesses IMERG data.

    This replaces missing values with data from another dataset

    Inputs:
    :param data: Input IMERG data to process (np.ndarray with shape time x lat x lon)
    :param data_replace: Dataset to use to replace missing values (np.ndarray with shape time x lat x lon)

    Outputs:
    :param data: Processed IMERG data (np.ndarray with shape time x lat x lon)
    '''

    # Replace any missing (NaN) data with data_replace (ERA5 data).
    # (note NaN values must be removed here as they will cause errors later on in ML training)
    data = np.where(np.isnan(data), data_replace, data)

    return data

def unit_converter(
        data, 
        variable, 
        dataset
        ) -> np.ndarray:
    '''
    Convert the units of ERA5 data to be consistent with units of another dataset.

    Inputs:
    :param data: Data to be converted (np.ndarray with shape time x lat x lon)
    :param variable: Name of the variable to convert units for
    :param dataset: Name of the dataset whose units data is being converted to

    Outputs:
    :param data_converted: Data with converted units (np.ndarray with shape time x lat x lon)
    '''

    if dataset == 'gldas':
        # All GLDAS2 variables overlap with ERA5
        if (variable == 'temperature') | (variable == 'wind_speed') | (variable == 'pressure') | ('fdii' in variable) | (variable == 'sesr'):
            # Temperature in GLDAS and ERA5 is in K, wind speed is in m s^-1 for both, and pressure is in Pa for both
            # No unit conversion necessary
            data_converted = data

        elif variable == 'radiation':
            # Convert from J m^-2 (over a 24 hour accumulation) to W m^-2
            data_converted = data / (60 * 60 * 24) # converts J m^-2 day^-1 -> J s^-1 m^-2 = W

        elif ('precipitation' in variable) | (variable == 'evaporation') | (variable == 'potential_evaporation'):
            # Precipitation and evaporation have the same unit conversions (PET will have the same conversion): m (1 day accumulation) to kg m^-2 s^-1
            # Multiply water variable by density of water to get kg m^-2, and then convert day^-1 to s^-1
            data_converted = data * 1000 / (60 * 60 * 24)

        elif variable == 'soil_moisture_1':
            # Convert m^3 m^-3 to kg m^-2 
            # (for a water variable, multiply by the density of water, then multiply by the soil depth in m [0.07])
            data_converted = data * 1000 * 0.07

        elif variable == 'soil_moisture_2':
            # Convert m^3 m^-3 to kg m^-2 
            # (for a water variable, multiply by the density of water, then multiply by the soil depth in m [0.21])
            data_converted = data * 1000 * 0.21

        elif variable == 'soil_moisture_3':
            # Convert m^3 m^-3 to kg m^-2 
            # (for a water variable, multiply by the density of water, then multiply by the soil depth in m [0.72])
            data_converted = data * 1000 * 0.72

        elif variable == 'soil_moisture_4':
            # Convert m^3 m^-3 to kg m^-2 
            # (for a water variable, multiply by the density of water, then multiply by the soil depth in m [1.89])
            data_converted = data * 1000 * 1.89

    elif dataset == 'modis':
        # Only MODIS variables that overlap with other datasets are ET and PET, which have the same unit conversions
        # Convert m to kg m^-2 (multiply a water-based variable multiplying by the density of water)
        data_converted = data * 1000

    elif dataset == 'imerg':
        # IMERG only has precipitation; convert m to mm
        data_converted = data * 1000

    return data_converted

def load_index_data(
        filename, 
        timestamps_prepared: bool = False, 
        enso: bool = False, 
        path: str = './'
        ) -> Tuple[np.ndarray, List[datetime]]:
    '''
    Load the climate index data into a dictionary

    Inputs:
    :param filename: Filename of the .csv containing the climate index
    :param timestamps_prepared: Boolean indicating whether the .csv file already has timestamps in it
    :param ense: Boolean indicating whether the climate index is ENSO or not
                 Note ENSO needs to be treated separately since it has SSTs and SSTa's for multiple 
                 regions instead of just one
    :param path: Directory path pointing to the climate index file

    Outputs:
    :param data: Array climate index time series (np.ndarray with shape time)
    :param dates: Datetime labels for data as a list (length of time)
    '''

    # Load the datasets
    timestamps = np.loadtxt('%s/%s'%(path, filename), delimiter = ',', dtype = str, skiprows = 1, usecols = 0)   
    if enso: # The ENSO timeseries has multiple indices; SSTs and SSTas, for all 4 ENSO zones
        data = np.loadtxt('%s/%s'%(path, filename), delimiter = ',', skiprows = 1, usecols = (1, 2, 3, 4, 5, 6, 7, 8))
    else:
        data = np.loadtxt('%s/%s'%(path, filename), delimiter = ',', skiprows = 1, usecols = 1)

    # Turn the timestamps into datetimes
    if timestamps_prepared:
        dates = [datetime.fromisoformat(date) for date in timestamps]
    else:
        dates = [datetime.strptime(date, '%Y-%m-%d') for date in timestamps]

    return data, dates

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

def test_map(
        data, 
        lat, 
        lon, 
        date, 
        data_name
        ) -> None:
    '''
    Create a simple plot of the data to examine and test it.
    
    Inputs:
    :param data: Data to be plotted (np.ndarray with shape lat x lon)
    :param lat: Latitude grid of the data (np.ndarray with shape lat x lon)
    :param lon: Longitude grid of the data (np.ndarray with shape lat x lon)
    :param dates: Array of datetimes corresponding to the timestamps in data (np.ndarray with shape time)
    :param data_name: Full name of the variable being plotted
    '''
    
    # Lonitude and latitude tick information
    lat_int = 15
    lon_int = 30
    
    lat_label = np.arange(-90, 90, lat_int)
    lon_label = np.arange(-180, 180, lon_int)


    # Projection information
    data_proj = ccrs.PlateCarree()
    fig_proj  = ccrs.PlateCarree()

    # Colorbar information
    #### NOTE: Might need to adjust cmin, cmax, and cint for other variables
    # cmin = np.ceil(np.nanmin(data[:,:])); cmax = np.floor(np.nanmax(data[:,:])); cint = (cmax - cmin)/100
    cmin = np.nanmin(data[:,:]); cmax = np.nanmax(data[:,:]); cint = (cmax - cmin)/100
    
    clevs = np.arange(cmin, cmax+cint, cint)
    nlevs = len(clevs) - 1
    cmap  = plt.get_cmap(name = 'BrBG', lut = nlevs)

    # Create the figure
    fig = plt.figure(figsize = [12, 16])
    ax = fig.add_subplot(1, 1, 1, projection = fig_proj)

    # Set the title
    ax.set_title('%s for %s'%(data_name, date.strftime('%Y-%m-%d')), fontsize = 16)    

    # Add coastlines
    ax.coastlines()

    # Set tick information
    ax.set_xticks(lon_label, crs = ccrs.PlateCarree())
    ax.set_yticks(lat_label, crs = ccrs.PlateCarree())
    ax.set_xticklabels(lon_label, fontsize = 14)
    ax.set_yticklabels(lat_label, fontsize = 14)

    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()

    # Plot the data
    cs = ax.contourf(lon, 
                     lat, 
                     data[:,:], 
                     levels = clevs, 
                     cmap = cmap, 
                     transform = data_proj, 
                     extend = 'both', 
                     zorder = 1)

    # Add a colorbar
    cbax = fig.add_axes([0.125, 0.30, 0.80, 0.02])
    cbar = fig.colorbar(cs, 
                        cax = cbax, 
                        orientation = 'horizontal')

    ax.set_extent([np.nanmin(lon), np.nanmax(lon), np.nanmin(lat), np.nanmax(lat)], 
                    crs = fig_proj)
    
    # Save the figure
    plt.savefig('%s_%s_test_map.png'%(data_name, date.strftime('%Y-%m-%d')))

    #plt.show(block = False)
    plt.close('all')

if __name__ == '__main__':
    # Main function used to test run functions within this script

    # Parameters of data to load and dataset to use
    variable = 'potential_evaporation'
    dataset = 'modis' # options are gldas or modis or imerg
    year = 2012

    # Get the path, filename, and data key for the ERA5 data.
    path = path_to_raw_datasets(variable)
    new_grid_fn = get_fn(variable, year)
    new_sname = get_var_shortname(variable)

    # Load in ERA5 data, and convert time to datetimes
    data_grid = load_nc('%s/%s.nc'%(path, new_grid_fn), new_sname)
    data_grid['time'] = np.array([datetime.fromisoformat(date) for date in data_grid['time']])
    print(data_grid[new_sname].shape, data_grid['lat'].shape)

    # Load in data from other dataset and interpolate
    grid_new = interpolate_data(data_grid, 
                                new_sname, 
                                dataset, 
                                variable, 
                                year, 
                                resolution = 0.25)
    print(grid_new[new_sname].shape)

    # Make test plot to examine the interpolation
    rand_int = np.random.randint(grid_new[new_sname].shape[0])
    plot_data = grid_new[new_sname]
    plot_data[plot_data == 0] = np.nan
    test_map(plot_data[rand_int,:,:], 
             data_grid['lat'], 
             data_grid['lon'], 
             data_grid['time'][rand_int], 
             data_name = variable)
