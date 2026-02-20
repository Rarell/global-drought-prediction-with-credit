'''General data loading and outputing (saving/storing)

Provides several functions for loading nc datasets
and saving datasets to zarr files
'''

import numpy as np
import zarr
from typing import Dict
from netCDF4 import Dataset
from datetime import datetime, timedelta

from path_to_raw_datasets import path_to_raw_datasets, get_var_shortname, get_fn
from preprocessing import reduce_spatial_scale

# Format of zarr files to load and make
zarr.config.set({'default_zarr_format': 2})

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

def make_zarr_group(
        data, 
        lat, 
        lon, 
        dates, 
        variables, 
        year, 
        var_type = 'surface', 
        levels = None
        ) -> None:
    '''
    Make a zarr group and store data in. Data is chunked along the time axis

    Inputs:
    :param data: Data to be stored (should be dict with each entry being an ndarray with shape 
                 time x level x lat x lon for upper air variables,
                 lat x lon for static variables, or time x lat x lon for others)
    :param lat: Latitude labels for data (must be ndarray with shape lat x lon)
    :param lon: Longitude labels for data (must be ndarray with shape lat x lon)
    :param dates: list/array of date labels (as strings)
    :param variables: List of variables in data to store
    :param year: Year of data being saved (used to determine the zarr filename)
    :param var_type: Type of data being save (upper_air, surface, dynamic, forcing, diagnostic, static, or climatology)
    :param levels: List of pressure levels being stored if upper air variables is being stored
    '''

    # Determine the array dimensions based on type of variables 
    if var_type == 'upper_air':
        array_dims = ['time', 'level', 'latitude', 'longitude']
    elif var_type == 'surface':
        array_dims = ['time', 'latitude', 'longitude']
    elif var_type == 'dynamic':
        array_dims = ['time', 'latitude', 'longitude']
    elif var_type == 'diagnostic':
        data_dir = 'diagnostic_variables'
        array_dims = ['time', 'latitude', 'longitude']
    elif var_type == 'forcing':
        array_dims = ['time', 'latitude', 'longitude']
    elif var_type == 'static':
        array_dims = ['latitude', 'longitude']
    elif var_type == 'climatology':
        array_dims1 = ['time', 'level', 'latitude', 'longitude']
        array_dims2 = ['time', 'latitude', 'longitude']

    print('data_dir determined')

    # Make the root zarr file
    store = zarr.storage.LocalStore('./%s.%04d.zarr'%(var_type, year))
    root = zarr.create_group(store = store, overwrite = True)

    # Convert times to 
    time = dates

    print('root branch created')

    # Store each variable in data
    for var in variables:
        # Get the variable short name
        sname = get_var_shortname(var)
        if var_type == 'forcing':
            sname = sname.upper()

        # Determine the shape of data chunks
        chunk_shape = []
        if var_type == 'static':
            # Special case since static variables don't have a time dimension
            chunk_shape = data[sname].shape
        else:
            # Chunk along the time axis (so chuck shapes are 1 x level x lat x lon/1 x lat x lon)
            chunk_shape.append(1) # Chunk along the first (time) axis

            # Get the rest of the chuck shape based on the input variable (minus the time axis)
            for sh in data[sname].shape[1:]:
                chunk_shape.append(sh)
        
        print('chunk shape made: ', chunk_shape)

        # Special case for climatology dataset, since that has both upper air and surface variables
        # Determine which data shape to use based on current variable in the loop
        if (var_type == 'climatology'):
            if sname in ['u', 'v', 'z', 'q_tot']:
                array_dims = array_dims1
            else:
                array_dims = array_dims2

        # create variable entry in zarr
        data_zarr = root.create_array(name = sname, 
                                      shape = data[sname].shape, 
                                      chunks = chunk_shape, 
                                      dtype = data[sname].dtype, 
                                      fill_value = 9.999e+20, # Note this should be specified as fill_value = 0 by default (bad for precipitation, land-sea, etc.)
                                      overwrite = True)

        # Add data to zarr
        data_zarr[:] = data[sname]

        # Assign array dimensions attributes (this is necessary for xarray to load the files)
        data_zarr.attrs['_ARRAY_DIMENSIONS'] = array_dims

    # For upper air and static variables, add the levels used
    # This is needed in both upper air and static for the CREDIT postblock to run smoothly
    if (var_type == 'upper_air') | (var_type == 'static'):
        # Create the entry
        level_zarr = root.create_array(name = 'level', 
                                       shape = (len(levels), ), 
                                       chunks = (len(levels), ), 
                                       dtype = type(levels[0]), 
                                       fill_value = 0,
                                       overwrite = True)

         # Add the levels to zarr
        level_zarr[:] = levels

        # Assign array dimensions attributes (this is necessary for xarray to load the files)
        level_zarr.attrs['_ARRAY_DIMENSIONS'] = ['level']

    # Add lat and lon to the zarr file
    lat = lat.astype(np.float32)
    lon = lon.astype(np.float32)

    # Create the entries
    lat_zarr = root.create_array(name = 'latitude', 
                                 shape = lat.shape, 
                                 chunks = lat.shape, 
                                 dtype = lat.dtype, 
                                 fill_value = 9.999e+20, 
                                 overwrite = True)
    
    lon_zarr = root.create_array(name = 'longitude', 
                                 shape = lon.shape, 
                                 chunks = lon.shape, 
                                 dtype = lon.dtype, 
                                 fill_value = 9.999e+20, 
                                 overwrite = True)

    lat_zarr[:] = lat
    lon_zarr[:] = lon

    # Assign array dimensions attributes (this is necessary for xarray to load the files)
    lat_zarr.attrs['_ARRAY_DIMENSIONS'] = ['latitude', 'longitude']
    lon_zarr.attrs['_ARRAY_DIMENSIONS'] = ['latitude', 'longitude']

    # Add timestamps to zarr file
    if dates is None:
        pass
    else:
        # Create the entry
        time_zarr = root.create_array(name = 'time', 
                                      shape = time.size, 
                                      chunks = chunk_shape[0], 
                                      dtype = type(time[0]),
                                      overwrite = True)
        
        # Add time stamps to the variable
        time_zarr[:] = time

        # Assign array dimensions attributes (this is necessary for xarray to load the files)
        time_zarr.attrs['_ARRAY_DIMENSIONS'] = ['time']

    print('data entered into zarr files')

    # Consolidate the metadata into a .zmetadata file (this is used when loading the zarr file)
    zarr.consolidate_metadata(store) 

    # display information on the zarr group
    print(root.info)
    print(root.tree())

def load_static_type(
        variable, 
        start_year, 
        end_year, 
        reduce_scale
        ) -> np.ndarray:
    '''
    Load static data (e.g., soil type, vegetation type) as an average in time

    Inputs:
    :param variable: Name of the variable being loaded
    :param start_year: Starting year of the dataset to load
    :param end_year: Ending year of the dataset to load
    :param reduce_scale: Boolean indicating whether to reduce gridded 
                         resolution by 1 quarter (i.e., quarter degree to one degree resolution)

    Outputs:
    :param data_type: Time averaged variable (np.ndarray with shape lat x lon)
    '''

    # Load in a test dataset to get the spatial size

    # Get the path and filename information for the test set
    test_path = path_to_raw_datasets(variable, 'era5')
    test_fn = get_fn(variable, 2000)
    test_sname = get_var_shortname(variable)

    # Load the test set
    test = {}
    with Dataset('%s/%s.nc'%(test_path, test_fn), 'r') as nc:
        test[test_sname] = nc.variables[test_sname][:]
        test['lat'] = nc.variables['lat'][:]
        test['lon'] = nc.variables['lon'][:]
        time = nc.variables['date'][:]

    # Reduce spatial scale if desired
    if reduce_scale:
        test, lat, lon = reduce_spatial_scale(test, test_sname)
    else:
        # Note at quarter degree resolution, the grid is 721 x 1440, 
        # which is difficult to work with since 721 only has two primes (721=7*103)
        # So remove the -90 degree point to make the grid easier (720 = 2^4*3^2*5)
        lat = test['lat'][:,:-1]; lon = test['lon'][:,:-1]
        test = test[test_sname][:,:-1,:]

    T, I, J = test.shape

    # Construct the full set of years in the dataset
    years = np.arange(start_year, end_year+1)

    # Initialize the dataset
    data_type = np.zeros((I, J))
    N = 0

    # Perform calculation year by year 
    # (so only one year is loaded to RAM at a time)
    for year in years:

        # Determine the path and filename for the variable
        path = path_to_raw_datasets(variable, 'era5')
        fn = get_fn(variable, year)
        sname = get_var_shortname(variable)

        # Load in the data for a given year
        data = load_nc('%s/%s.nc'%(path, fn), sname)

        # Reduce the spatial scale of data if necessary
        if reduce_scale:
            data, _, _ = reduce_spatial_scale(data, sname)
        else:
            # For quarter degree resolution remove the -90 degree lat data point
            data = data[sname][:,:-1,:]

        # Sum along the time axis (divide later to obtain the average)
        tmp = np.nansum(data, axis = 0)
        data_type = np.nansum([data_type, tmp], axis = 0) # nansum account for any possible NaNs

        # Add the number of time points summed over
        N = N + data.shape[0] 

    # Divide by the total number of time points to obtain the time average
    data_type = data_type/N

    # Round the data to an integer
    # data_type = np.round(data_type)

    return data_type