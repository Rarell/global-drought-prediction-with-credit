import numpy as np
import zarr
from netCDF4 import Dataset
from datetime import datetime, timedelta

from path_to_raw_datasets import path_to_raw_datasets, get_var_shortname, get_fn
from preprocessing import reduce_spatial_scale

zarr.config.set({'default_zarr_format': 2})

def load_nc(filename, var):
    '''
    Load a nc file
    '''

    x = {}

    with Dataset(filename, 'r') as nc:
        x['lat'] = nc.variables['lat'][:]
        x['lon'] = nc.variables['lon'][:]
        x['time'] = nc.variables['date'][:]

        x[var] = nc.variables[var][:]

    return x

def make_zarr_group(data, lat, lon, dates, variables, year, var_type = 'surface', levels = None, Z_surf = False):
    '''
    Make a zarr group to store data in. Data is chunked along the time axis
    '''

    if var_type == 'upper_air':
        data_dir = 'upper_air_variables'
        array_dims = ['time', 'level', 'latitude', 'longitude']
    elif var_type == 'surface':
        data_dir = 'surface_variables'
        array_dims = ['time', 'latitude', 'longitude']
    elif var_type == 'dynamic':
        data_dir = 'dynamic_variables'
        array_dims = ['time', 'latitude', 'longitude']
    elif var_type == 'diagnostic':
        data_dir = 'diagnostic_variables'
        array_dims = ['time', 'latitude', 'longitude']
    elif var_type == 'forcing':
        data_dir = 'forcing_variables'
        array_dims = ['time', 'latitude', 'longitude']
    elif var_type == 'static':
        data_dir = 'static_variables'
        array_dims = ['latitude', 'longitude']
    elif var_type == 'climatology':
        array_dims1 = ['time', 'level', 'latitude', 'longitude']
        array_dims2 = ['time', 'latitude', 'longitude']

    print('data_dir determined')

    # make root zarr file
    store = zarr.storage.LocalStore('./%s.%04d.zarr'%(var_type, year))
    root = zarr.create_group(store = store, overwrite = True)
    #store = zarr.DirectoryStore('%s.%04d.zarr'%(var_type, year))
    #root = zarr.group(store, overwrite = True)

    # Convert times to 
    if dates is None:
        pass
    else:
        #days = [datetime.fromisoformat(day) for day in dates]
        #time = np.array([(day - datetime(1970,1,1)).total_seconds()/3600 for day in days])
        #print(time)
        time = dates

    print('root branch created')

    for var in variables:
        sname = get_var_shortname(var)
        if var_type == 'forcing':
            sname = sname.upper()

        # Shape of data chunks
        chunk_shape = []
        if var_type == 'static':
            chunk_shape = data[sname].shape
        else:
            chunk_shape.append(1) # Chunk along the first (time) axis
            for sh in data[sname].shape[1:]:
                chunk_shape.append(sh) # Remaining chunk shapes are 1 to prevent chunking along those axes
        
        print('chunk shape made: ', chunk_shape)
        if(var_type == 'climatology'):
            if sname in ['u', 'v', 'z', 'q_tot']:
                array_dims = array_dims1
            else:
                array_dims = array_dims2

        # create variable entry in zarr
        data_zarr = root.create_array(name = sname, 
                                      shape = data[sname].shape, 
                                      chunks = chunk_shape, 
                                      dtype = data[sname].dtype, 
                                      fill_value = 9.999e+20, 
                                      #zarr_format = 2, 
                                      overwrite = True)

        # Add data to zarr
        data_zarr[:] = data[sname]

        # Assign array dimensions attributes (this is necessary for xarray to load the files)
        data_zarr.attrs['_ARRAY_DIMENSIONS'] = array_dims

    # if Z_surf:
    #     data_zarr = root.create_array(name = 'surface_geopotential_var', 
    #                                   shape = data['surface_geopotential_var'].shape, 
    #                                   chunks = chunk_shape, dtype = data['surface_geopotential_var'].dtype, 
    #                                   fill_value = 9.999e+20,
    #                                   overwrite = True)
        
    #     # Add data to zarr
    #     data_zarr[:] = data['surface_geopotential_var']

    #     # Assign array dimensions attributes (this is necessary for xarray to load the files)
    #     data_zarr.attrs['_ARRAY_DIMENSIONS'] = array_dims

    if (var_type == 'upper_air') | (var_type == 'static'):
        level_zarr = root.create_array(name = 'level', 
                                       shape = (len(levels), ), 
                                       chunks = (len(levels), ), dtype = type(levels[0]), 
                                       fill_value = 0,
                                       overwrite = True)

         # Add data to zarr
        level_zarr[:] = levels

        # Assign array dimensions attributes (this is necessary for xarray to load the files)
        level_zarr.attrs['_ARRAY_DIMENSIONS'] = ['level']

    # Lat and lon entries
    lat = lat.astype(np.float32)
    lon = lon.astype(np.float32)
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

    lat_zarr.attrs['_ARRAY_DIMENSIONS'] = ['latitude', 'longitude']
    lon_zarr.attrs['_ARRAY_DIMENSIONS'] = ['latitude', 'longitude']

    # Time entries
    if dates is None:
        pass
    else:
        time_zarr = root.create_array(name = 'time', 
                                      shape = time.size, 
                                      chunks = chunk_shape[0], 
                                      dtype = type(time[0]),
                                      overwrite = True)
        time_zarr[:] = time

        time_zarr.attrs['_ARRAY_DIMENSIONS'] = ['time']

    

    print('data entered into zarr files')

    zarr.consolidate_metadata(store) # Consolidate the metadata into a .zmetadata file
    # display information on the zarr group
    print(root.info)
    print(root.tree())

def load_static_type(variable, start_year, end_year, reduce_scale):
    '''
    Load the 'type' static data (e.g., soil type, vegetation type) as an average in time
    '''

    # Load in a test dataset to get the spatial size
    test_path = path_to_raw_datasets(variable, 'era5')
    test_fn = get_fn(variable, 2000)
    test_sname = get_var_shortname(variable)
    test = {}
    with Dataset('%s/%s.nc'%(test_path, test_fn), 'r') as nc:
        test[test_sname] = nc.variables[test_sname][:]
        test['lat'] = nc.variables['lat'][:]
        test['lon'] = nc.variables['lon'][:]
        time = nc.variables['date'][:]

    # Reduce spatial scale?
    if reduce_scale:
        test, lat, lon = reduce_spatial_scale(test, test_sname)
    else:
        # Note at quarter degree resolution, the grid is 721 x 1440, which is difficult to work with since 721 only has two primes (721=7*103)
        # So remove the -90 degree point to make the grid easier (720 = 2^4*3^2*5)
        lat = test['lat'][:,:-1]; lon = test['lon'][:,:-1]
        test = test[test_sname][:,:-1,:]

    T, I, J = test.shape

    years = np.arange(start_year, end_year+1)

    # Initialize the dataset
    data_type = np.zeros((I, J))
    N = 0

    for year in years:
        # Load in the data for a given year
        path = path_to_raw_datasets(variable, 'era5')
        fn = get_fn(variable, year)
        sname = get_var_shortname(variable)

        data = load_nc('%s/%s.nc'%(path, fn), sname)

        # Reduce the spatial scale of data?
        if reduce_scale:
            data, _, _ = reduce_spatial_scale(data, sname)
        else:
            data = data[sname][:,:-1,:]

        # Sum on the time axis, (divide later to obtain the average)
        tmp = np.nansum(data, axis = 0)
        data_type = np.nansum([data_type, tmp], axis = 0) # nansum account for any possible NaNs
        N = N + data.shape[0] # Add the number of time points summed over

    # Divide by the total number of time points to obtain the average
    data_type = data_type/N

    # Round the data and return it an integer
    # data_type = np.round(data_type)

    return data_type