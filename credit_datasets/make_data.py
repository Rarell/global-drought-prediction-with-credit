# Script to takes netcdf ERA5 files and turn them into zarrs of the desired format for CREDIT test runs
import os
import numpy as np
import gc
import multiprocessing as mp
import zarr
import argparse
from datetime import datetime, timedelta
from netCDF4 import Dataset
from functools import partial

from inputs_outputs import load_nc, make_zarr_group, load_static_type
from preprocessing import reduce_spatial_scale, running_sum, data_loading, calculate_surface_potential, calculate_wind_speed
from path_to_raw_datasets import path_to_raw_datasets, get_var_shortname, get_fn
from moisture_calculations import calculate_q_total_upper_level
from transform_grid import interpolate_data, interpolate_to_new_grid

# TODO:
#    Add to parser
#    update metadata.yaml file
#    Add multiprocessing to static function
#    Might try converting to zarr files, then doing preprocessing calculations
#    Investigate Nvidia CUDA docs for anything useful
#    Add calculations for surface moisture (currently stuck at convertin kg m^-2 to kg kg^-1)


#    Add regridding to account for different grids between IMERG, MODIS, GLDAS, and ERA5
#    Calculate SESR and FDII for GLDAS2
#    Test regridding
#    Collect Climate indices


# Set default zarr format to 2; this speeds up creation of zarr files, 
# and creates the .zarray, .zattrs, and .zmetadata files that xarray 
# looks for when loading zarr
zarr.config.set({'default_zarr_format': 2})
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

def make_climatology_dataset(args, make_climatology = True, make_statistics = False):
    '''
    Calculate the climatological mean for each day of the year for each of the predicted variables
    (that is, upper air, surface, and diagnostic variables)
    '''

    # Initialize dataset
    climatology = {}
    years = np.arange(args.years[0], args.years[1]+1)
    N_levels = len(args.pressure_levels)

    upper_air_variables = args.upper_air_variables
    # Surface and diagnost variables should have the same shape and can then be considered together
    variables = np.concatenate([args.surface_variables, args.dynamic_variables, args.diagnostic_variables])
    all_variables = np.concatenate([upper_air_variables, variables])

    # Load in a test dataset (includes leap year)
    print('Loading test data')
    test_path = path_to_raw_datasets(upper_air_variables[0], 'era5', level = args.pressure_levels[0])
    test_fn = get_fn(upper_air_variables[0], 2000, level = args.pressure_levels[0])
    test_sname = get_var_shortname(upper_air_variables[0])
    test = {}
    with Dataset('%s/%s.nc'%(test_path, test_fn), 'r') as nc:
        test[test_sname] = nc.variables[test_sname][:]
        test['lat'] = nc.variables['lat'][:]
        test['lon'] = nc.variables['lon'][:]
        time = nc.variables['date'][:]

    # Reduce data scale to 1 degree x 1 degree?
    if args.reduce_scale:
        print('Reducing spatial scale of test data')
        test, lat, lon = reduce_spatial_scale(test, test_sname)
    else:
        # Note at quarter degree resolution, the grid is 721 x 1440, which is difficult to work with since 721 only has two primes (721=7*103)
        # So remove the -90 degree point to make the grid easier (720 = 2^4*3^2*5)
        lat = test['lat'][:-1,:]; lon = test['lon'][:-1,:]
        test = test[test_sname][:,:-1,:]

    # Get information on the data
    T, I, J = test.shape
    dates = np.array([datetime.fromisoformat(day) for day in time])

    if os.path.exists('climatology.2012.zarr') & (make_climatology | make_statistics):
        print('Climatology file made. Loading file instead')
        climatology = zarr.open_group('climatology.2012.zarr')
    else:

        # Iterate through upper-air (4D) variables and get climatologies
        for var in upper_air_variables:
            # Initialize current dataset
            sname = get_var_shortname(var)
            climatology[sname] = np.zeros((T, N_levels, I, J), dtype = np.float32)

            # Loop through all elevation levels in upper air variables
            for l, level in enumerate(args.pressure_levels):

                # Collect the arguments needed for to load and process 1 year of data
                param_args = [(year, var, sname, args.datasets, args.reduce_scale, level) for year in years]
                
                # Use multiprocessing to load and process data; when joined, sum all the pieces
                with mp.Pool(args.nthreads) as pool:
                    data = pool.starmap(data_loading, param_args) # Load and process data

                    # Sum all the values according to the day of year
                    for datum in data:
                        days = datum[1]; months = datum[2]
                        for t, date in enumerate(dates):
                            # Get the days in the current corresponding to the day in the loop (may not be the same as t as current year may not be a leap year)
                            ind = np.where( (date.day == days) & (date.month == months) )[0]

                            # For non-leap years, ind can be empty; skip this day
                            if len(ind) < 1:
                                continue
                            else:
                                climatology[sname][t,l,:,:] = np.nansum([climatology[sname][t,l,:,:], datum[0][ind[0],:,:]], axis = 0) # np.nansum to account for any NaNs
                
                del data; gc.collect() # Free up memory

            # Reduce the variable save to help with memory
            climatology[sname] = climatology[sname].astype(np.float32)
    

        # Repeat the process with 3D variables
        N = np.zeros((T)) 
        for var in variables:
            # Initialize current dataset
            sname = get_var_shortname(var)
            climatology[sname] = np.zeros((T, I, J), dtype = np.float32)
            
            # Collect the arguments needed for to load and process 1 year of data
            param_args = [(year, var, sname, args.datasets, args.reduce_scale, None) for year in years]

            # Use multiprocessing to load and process data; when joined, sum all the pieces
            with mp.Pool(args.nthreads) as pool:
                data = pool.starmap(data_loading, param_args) # Load and process the data
    
                # Sum all the values according to the day of year
                for datum in data:
                    days = datum[1]; months = datum[2]
                    for t, date in enumerate(dates):
                        # Get the days in the current corresponding to the day in the loop (may not be the same as t as current year may not be a leap year)
                        ind = np.where( (date.day == days) & (date.month == months) )[0]
                        # For non-leap years, ind can be empty; skip this day
                        if len(ind) < 1:
                            continue
                        else:
                            climatology[sname][t,:,:] = np.nansum([climatology[sname][t,:,:], datum[0][ind[0],:,:]], axis = 0) # np.nansum to account for any NaNs
                            if var == variables[0]:
                                N[t] = N[t] + 1
            
            del data; gc.collect() # Free up memory

            # Decrease the variable size to help with memory
            climatology[sname] = climatology[sname].astype(np.float32)

        print(N)

        # At the end, loop again to get to divide the sums by N to get the means
        print('Calculating averages')
        for t, date in enumerate(dates):
            for var in upper_air_variables:
                sname = get_var_shortname(var)
                climatology[sname][t,:,:,:] = climatology[sname][t,:,:,:]/N[t]

            for var in variables:
                sname = get_var_shortname(var)
                climatology[sname][t,:,:] = climatology[sname][t,:,:]/N[t]

        # Compress dataset to float32 datatype
        for var in all_variables:
            sname = get_var_shortname(var)
            climatology[sname] = climatology[sname].astype(np.float32)
            print(sname, np.nanmin(climatology[sname]), np.nanmax(climatology[sname]), np.nanmean(climatology[sname]))

        # Write the climatology (climatological mean) dataset
        if make_climatology:
            print('Saving results')
            make_zarr_group(climatology, lat, lon, time, all_variables, 2012, var_type = 'climatology', levels = args.pressure_levels)

    if make_statistics:
        stds = {}
        # Do standard deviation calculations, then average it all together to get the "average mean" and the "average standard deviation"
        print('Calculating standard deviations')
        # Iterate through upper-air (4D) variables and get climatologies
        for var in upper_air_variables:
            # Initialize current dataset
            sname = get_var_shortname(var)
            # stds[sname] = np.zeros((T, N_levels, I, J), dtype = np.float32)
            stds[sname] = np.zeros((N_levels))
            for l, level in enumerate(args.pressure_levels):
                # Collect the arguments needed for to load and process 1 year of data
                param_args = [(year, var, sname, args.datasets, args.reduce_scale, level) for year in years]

                # Use multiprocessing to load and process data; when joined, calculate and sum the error
                with mp.Pool(args.nthreads) as pool:
                    data = pool.starmap(data_loading, param_args) # Load and process 1 year of data 

                    # Calculate and sum the error according to the day of year
                    for datum in data:
                        days = datum[1]; months = datum[2]
                        # This method should keep the spatial variation in time and space
                        stds[sname][l] = np.nansum([stds[sname][l], np.nanstd(datum[0][:,:,:])])
                        
                        # The method below will calculate the average standard deviation in space, but will not contain temporal variation as well
                        # for t, date in enumerate(dates):
                        #     # Get the days in the current corresponding to the day in the loop (may not be the same as t as current year may not be a leap year)
                        #     ind = np.where( (date.day == days) & (date.month == months) )[0]
                        #     # For non-leap years, ind can be empty; skip this day
                        #     if len(ind) < 1:
                        #         continue
                        #     else:
                        #         error = (datum[0][ind[0],:,:] - climatology[sname][t,l,:,:])**2
                        #         stds[sname][t,l,:,:] = np.nansum([stds[sname][t,l,:,:], error], axis = 0)
                
                del data; gc.collect() # Free up memory

            # Decrease variable size to help with memory
            stds[sname] = stds[sname].astype(np.float32)


        # Repeat the process with 3D variables
        # N = np.zeros((T)) 
        N = 0
        for var in variables:
            # Initialize current dataset
            sname = get_var_shortname(var)
            # stds[sname] = np.zeros((T, I, J), dtype = np.float32)
            stds[sname] = 0

            # Collect the arguments needed for to load and process 1 year of data
            param_args = [(year, var, sname, args.datasets, args.reduce_scale, None) for year in years]

            # Use multiprocessing to load and process data; when joined, calculate and sum the error
            with mp.Pool(args.nthreads) as pool:
                data = pool.starmap(data_loading, param_args) # Load and proceess 1 year of data
    
                # Calculate and sum the error according to the day of year
                for datum in data:
                    days = datum[1]; months = datum[2]

                    # This method should keep the spatial variation in time (in a given year) and space
                    stds[sname] = np.nansum([stds[sname], np.nanstd(datum[0])])
                    if var == variables[0]:
                        N = N+1

                    # The method below will calculate the average standard deviation in space, but will not contain temporal variation as well
                    # for t, date in enumerate(dates):
                    #     # Get the days in the current corresponding to the day in the loop (may not be the same as t as current year may not be a leap year)
                    #     ind = np.where( (date.day == days) & (date.month == months) )[0]
                    #     # For non-leap years, ind can be empty; skip this day
                    #     if len(ind) < 1:
                    #         continue
                    #     else:
                    #         error = (datum[0][ind[0],:,:] - climatology[sname][t,:,:])**2
                    #         stds[sname][t,:,:] = np.nansum([stds[sname][t,:,:], error], axis = 0)
                    #         if var == variables[0]:
                    #             N[t] = N[t] + 1

            del data; gc.collect() # Free up memory

            # Decrease variable size to help with memory
            stds[sname] = stds[sname].astype(np.float32)

        # Finish standard deviation calculations
        # for t, date in enumerate(dates):
        #     for var in upper_air_variables:
        #         sname = get_var_shortname(var)
        #         stds[sname][t,:,:,:] = np.sqrt(stds[sname][t,:,:,:]/(N[t]-1))

        #     for var in variables:
        #         sname = get_var_shortname(var)
        #         stds[sname][t,:,:] = np.sqrt(stds[sname][t,:,:]/(N[t]-1))
        print(N)
        for var in all_variables:
            sname = get_var_shortname(var)
            stds[sname] = stds[sname]/N
            print(sname, np.nanmin(stds[sname]), np.nanmax(stds[sname]), np.nanmean(stds[sname]))

        # Finally, average the means and standard deviations to a single variable
        means = {} # Creating a new dictionary for teh averaged means gets around the fact climatology may be a loaded in zarr array
        for var in upper_air_variables:
            sname = get_var_shortname(var)
            means[sname] = np.array([np.nanmean(climatology[sname][:,l,:,:]) for l in range(len(args.pressure_levels))]).astype(np.float32)
            # stds[sname] = np.array([np.nanmean(stds[sname][:,l,:,:]) for l in range(len(args.pressure_levels))]).astype(np.float32)

        for var in variables:
            sname = get_var_shortname(var)
            means[sname] = np.nanmean(climatology[sname][:]).astype(np.float32)
            # stds[sname] = np.nanmean(stds[sname][:]).astype(np.float32)

        # Write the means and standard deviations to .nc files
        print('Writing means and standard deviations')
        with Dataset('means.nc', 'w', format = 'NETCDF4') as nc:
            nc.createDimension('level', size = len(args.pressure_levels))
            #nc.createDimension('surface', size = 1)
            for var in upper_air_variables:
                sname = get_var_shortname(var)
                nc.createVariable(sname, means[sname].dtype, ('level', ))
                nc.variables[sname][:] = means[sname][:]

            for var in variables:
                sname = get_var_shortname(var)
                print('Means: ', means[sname].item(), means[sname].dtype)
                nc.createVariable(sname, means[sname].dtype)#, ('surface', ))
                nc.variables[sname][:] = means[sname]

        with Dataset('stds.nc', 'w', format = 'NETCDF4') as nc:
            nc.createDimension('level', size = len(args.pressure_levels))
            #nc.createDimension('surface', size = 1)
            for var in upper_air_variables:
                sname = get_var_shortname(var)
                nc.createVariable(sname, stds[sname].dtype, ('level', ))
                nc.variables[sname][:] = stds[sname][:]

            for var in variables:
                sname = get_var_shortname(var)
                nc.createVariable(sname, stds[sname].dtype)#, ('surface', ))
                nc.variables[sname][:] = stds[sname]

        # Calculate 2D latitude weights
        weights = np.ones((lat.shape))
        weights = np.cos(lat*np.pi/180)
        lat_tmp = lat.copy()

        # Get the region
        lat_tmp[np.invert((lat >= -35) & (lat <= 35))] = -999
        lat_tmp[np.invert((lon >= 335) | (lon <= 53))] = -999
        weights = np.where(lat_tmp == -999, weights/5, weights) # Alternatively to 0, weights/5 or weights/10

        # Normalize weights by mean(weights)
        lat_ind = np.where((lat[:,0] >= -35) & (lat[:,0] <= 35))[0]
        lon_ind = np.where((lon[0,:] >= 335) | (lon[0,:] <= 53))[0]
        tmp = weights[lat_ind,:]
        region_weights = tmp[:,lon_ind]
        weights_mean = np.nanmean(region_weights)
        weights = weights/weights_mean

        # Collect latitudes and longitudes
        lat = lat[:,0]
        lon = lon[0,:]
        coslat = np.cos(lat*np.pi/180)

        # Write lat and lons
        with Dataset('lat_and_lons.nc', 'w') as nc:
            nc.createDimension('lat', size = lat.size)
            nc.createDimension('lon', size = lon.size)

            nc.createVariable('latitude', lat.dtype, ('lat', ))
            nc.createVariable('longitude', lon.dtype, ('lon', ))
            nc.createVariable('coslat', coslat.dtype, ('lat', ))
            nc.createVariable('latitude_weights', weights.dtype, ('lat', 'lon'))

            nc.variables['latitude'][:] = lat[:]
            nc.variables['longitude'][:] = lon[:]
            nc.variables['coslat'][:] = coslat[:]
            nc.variables['latitude_weights'][:] = weights[:]

def make_upper_air_dataset(args):
    '''
    Make the upper air zarr dataset
    '''

    data_end = {}
    for var in args.upper_air_variables:
        data_total = []
        sname = get_var_shortname(var)
        # Collect data for each level
        for l, level in enumerate(args.pressure_levels):
            # Total specific humidity has to be calculated separately, and does not have a file
            if var == 'total_specific_humidity':
                data = calculate_q_total_upper_level(args.year, level)
            else:
                # Get filename
                path = path_to_raw_datasets(var, 'era5', level = level)
                filename = get_fn(var, args.year, level = level)
                print(filename)

                # Load variable
                data = load_nc('%s/%s.nc'%(path, filename), sname)

            # reduce scale of data to 1 deg x 1 deg
            if args.reduce_scale:
                data_new, lat, lon = reduce_spatial_scale(data, sname)
            else:
                # Note at quarter degree resolution, the grid is 721 x 1440, which is difficult to work with since 721 only has two primes (721=7*103)
                # So remove the -90 degree point to make the grid easier (720 = 2^4*3^2*5)
                data_new = data[sname][:,:-1,:]; lat = data['lat'][:-1,:]; lon = data['lon'][:-1,:] # Latitude may need to be reversed
    
            # reduce datasize to float16
            data_new = data_new.astype(np.float32)
            #data_new = data_new.astype(np.float16)
            print(np.nanmin(data_new), np.nanmax(data_new), np.nanmean(data_new))

            data_total.append(data_new)
        
        data_total = np.stack(data_total, axis = 1)

        # reduce datasize to float16
        data_total = data_total.astype(np.float32)
        data_end[sname] = data_total

        print('data size reduced')

    # create variable entry in zarr
    make_zarr_group(data_end, lat, lon, data['time'], args.upper_air_variables, args.year, var_type = 'upper_air', levels = args.pressure_levels)

def make_dataset(args, var_type = 'surface'):
    '''
    Make the zarr dataset for surface, dynamic, or diagnostic variables
    '''

    if var_type == 'surface':
        variables = args.surface_variables
    elif var_type == 'dynamic':
        variables = args.dynamic_variables
    elif var_type == 'diagnostic':
        variables = args.diagnostic_variables

    data_end = {}
    for var in variables:
        print(var)
        if var not in not_in_era5:
            if 'precipitation' in var:
                sname = get_var_shortname('precipitation')
            else:
                sname = get_var_shortname(var)

            if var == 'wind_speed':
                data = calculate_wind_speed(args.year)
            else:
                # Get filename
                path = path_to_raw_datasets(var, 'era5')
                filename = get_fn(var, args.year)
                print(filename)

                # Load variable
                data = load_nc('%s/%s.nc'%(path, filename), sname)
        else:
            # Load a filler set of data with the grid information for the interpolation step
            path = path_to_raw_datasets('temperature', 'era5')
            filename = get_fn('temperature', args.year)
            data = load_nc('%s/%s.nc'%(path, filename), 'tair')

            # Get the sname for the dataset
            sname = get_var_shortname(var)

        # Add data from other, non-ERA5 datasets?
        if np.invert(args.datasets == 'era5'):
            if (args.datasets == 'few'):
                # For 'few', only add the reanalyses and index data
                if var in gldas_variables:
                    dataset = 'gldas'
                elif var in index_variables:
                    dataset = 'climate_indices'
                else:
                    dataset = None

            if (args.datasets == 'all'):
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
                print('Loading and interpolation data from %s for %s and for year %d'%(dataset, var, args.year))
                sname_tmp = 'tair' if var in not_in_era5 else sname       
                data = interpolate_data(data, sname_tmp, dataset, var, args.year, resolution = 0.25)
                
        # Perform a running sum of precipitation?
        if var == 'precipitation_7day':
            sname = get_var_shortname(var)
            data['tp'] = data['tp'].astype(np.float64)
            data[sname] = running_sum(data['tp'], N = 7)
        elif var == 'precipitation_14day':
            data['tp'] = data['tp'].astype(np.float64)
            sname = get_var_shortname(var)
            data[sname] = running_sum(data['tp'], N = 14)
        elif var == 'precipitation_30day':
            data['tp'] = data['tp'].astype(np.float64)
            sname = get_var_shortname(var)
            data[sname] = running_sum(data['tp'], N = 30)

        # reduce scale of data to 1 deg x 1 deg
        if args.reduce_scale:
            data_new, lat, lon = reduce_spatial_scale(data, sname)
        else:
            data_new = data[sname][:,:-1,:]; lat = data['lat'][:-1,:]; lon = data['lon'][:-1,:] # Latitude may need to be reversed
    
        # reduce datasize to float16
        data_new = data_new.astype(np.float32)
        #data_new = data_new.astype(np.float16)
        print(np.nanmin(data_new), np.nanmax(data_new), np.nanmean(data_new))

        data_end[sname] = data_new

    # create variable entry in zarr
    make_zarr_group(data_end, lat, lon, data['time'], variables, args.year, var_type = var_type)

def make_forcing_dataset(args):
    '''
    Make the forcing dataset (annual mean of a given variable, giving the average value for a given day in the year)
    '''

    # Initialize dataset
    forcings = {}
    years = np.arange(args.years[0], args.years[1]+1)

    # Load in a test dataset (includes leap year)
    print('Loading test data')
    test_path = path_to_raw_datasets(args.forcing_variables[0], 'era5')
    test_fn = get_fn(args.forcing_variables[0], 2012)
    test_sname = get_var_shortname(args.forcing_variables[0])
    test = {}
    with Dataset('%s/%s.nc'%(test_path, test_fn), 'r') as nc:
        test[test_sname] = nc.variables[test_sname][:]
        test['lat'] = nc.variables['lat'][:]
        test['lon'] = nc.variables['lon'][:]
        time = nc.variables['date'][:]
        time_final = time

    if args.reduce_scale:
        print('Reducing spatial scale of test data')
        test, lat, lon = reduce_spatial_scale(test, test_sname)
    else:
        # Note at quarter degree resolution, the grid is 721 x 1440, which is difficult to work with since 721 only has two primes (721=7*103)
        # So remove the -90 degree point to make the grid easier (720 = 2^4*3^2*5)
        lat = test['lat'][:-1,:]; lon = test['lon'][:-1,:]
        test = test[test_sname][:,:-1,:]
        
    # Get information on the data
    T, I, J = test.shape
    dates = np.array([datetime.fromisoformat(day) for day in time])

    # Iterate through upper-air (4D) variables and get climatologies
    N = np.zeros((T))
    for var in args.forcing_variables:
        # Initialize current dataset
        sname = get_var_shortname(var)
        forcings[sname.upper()] = np.zeros((T, I, J), dtype = np.float32)

        # Collect the arguments needed for to load and process 1 year of data
        param_args = [(year, var, sname, args.datasets, args.reduce_scale) for year in years]
        
        # Use multiprocessing to load and process data; when joined, sum all the pieces
        with mp.Pool(args.nthreads) as pool:
            data = pool.starmap(data_loading, param_args) # Load and process data

            # Sum all the values according to the day of year
            for datum in data:
                days = datum[1]; months = datum[2]
                for t, date in enumerate(dates):
                    # Get the days in the current corresponding to the day in the loop (may not be the same as t as current year may not be a leap year)
                    ind = np.where( (date.day == days) & (date.month == months) )[0]

                    # For non-leap years, ind can be empty; skip this day
                    if len(ind) < 1:
                        continue
                    else:
                        forcings[sname.upper()][t,:,:] = np.nansum([forcings[sname.upper()][t,:,:], datum[0][ind[0],:,:]], axis = 0) # np.nansum to account for any NaNs
                        if var == args.forcing_variables[0]:
                            N[t] = N[t]+1

        forcings[sname.upper()] = forcings[sname.upper()].astype(np.float32)
        
        del data; gc.collect() # Free up memory
        # for year in years:
        #     # Load in the data
        #     print('Working on %s on %d'%(var, year))
        #     path = path_to_raw_datasets(var, 'era5')
        #     filename = get_fn(var, year)

        #     data = load_nc('%s/%s.nc'%(path, filename), sname)

        #     # Construct datetime information
        #     time = data['time']
        #     dates_current_year = np.array([datetime.fromisoformat(day) for day in time])

        #     days = np.array([day.day for day in dates_current_year])
        #     months = np.array([day.month for day in dates_current_year])

        #     if year == 2012:
        #         time_final = data['time'] # The timestamps used for the zarr file

        #     # Reduce the spatial scale if desired
        #     if args.reduce_scale:
        #         data, _, _ = reduce_spatial_scale(data, sname)
        #     else:
        #         data = data[sname]

        #     for t, date in enumerate(dates):
        #         # Get the days in the current corresponding to the day in the loop (may not be the same as t as current year may not be a leap year)
        #         ind = np.where( (date.day == days) & (date.month == months) )[0]

        #         # For non-leap years, ind can be empty; skip this day
        #         if len(ind) < 1:
        #             continue
        #         else:
        #             forcings[sname.upper()][t,:,:] = np.nansum([forcings[sname.upper()][t,:,:], data[ind[0],:,:]], axis = 0) # np.nansum to account for any NaNs
        #             if var == args.forcing_variables[0]:
        #                 N[t] = N[t] + 1
            

    # At the end, loop again to get to divide the sums by N to get the annual averages
    print('Calculating averages')
    for t, date in enumerate(dates):
        for var in args.forcing_variables:
            sname = get_var_shortname(var)
            forcings[sname.upper()][t,:,:] = forcings[sname.upper()][t,:,:]/N[t]

            forcings[sname.upper()] = forcings[sname.upper()].astype(np.float32)

    # Normalize the forcing data
    for var in args.forcing_variables:
        sname = get_var_shortname(var)
        forcings[sname.upper()] = forcings[sname.upper()]/np.nanmax(forcings[sname.upper()])

    print(np.nanmin(forcings[sname.upper()]), np.nanmax(forcings[sname.upper()]), np.nanmean(forcings[sname.upper()]))

    # Fix for final version
    make_zarr_group(forcings, lat, lon, time_final, args.forcing_variables, 2012, var_type = 'forcing')

def make_static_dataset(args):
    '''
    Make the static dataset; has no time dimension (for static variables like land-sea mask, vegetation and soil types, etc.)
    '''
    year = 2012
    path = path_to_raw_datasets('temperature', 'era5')
    filename = get_fn('temperature', year)
    # Load test set
    with Dataset('%s/%s.nc'%(path, filename), 'r') as nc:
        lat = nc.variables['lat'][:]
        lon = nc.variables['lon'][:]

    data_end = {}
    for var in args.static_variables:
        # Load the data
        if var == 'land_cover':
            # Get information for land cover file
            sname = get_var_shortname(var)
            path = path_to_raw_datasets(var, 'modis')
            filename = get_fn(var, 2000)

            # Load the MODIS land cover data (Note, this set does not have a date variable, and so cannot be loaded via load_nc)
            data_old = {}
            with Dataset('%s/%s'%(path, filename), 'r') as nc:
                data_old['lat'] = nc.variables['lat'][:,:].T/1e6 # Note the MODIS lat and lon here were multiplied by 1e6, presumable to keep them in integer form
                data_old['lon'] = nc.variables['lon'][:,:].T/1e6

                # The interpolation functions assumes a 3D array; initialize the land cover as 3D (axis 0 has length 1)
                I, J = data_old['lat'].shape
                # J = data_old['lat'].size; I = data_old['lon'].size
                data_old[sname] = np.ones((1, I, J))

                # Load land cover; note there is room here to use other land cover types, percentage coverage, or assessments for use
                data_old[sname][0,:,:] = nc.variables['Majority_Land_Cover_Type_1'][:,:].T

            # Interpolate land cover data to ERA5 .25 degree resolution
            data = {}
            data[sname] = interpolate_to_new_grid(data_old['lat'], data_old['lon'], data_old[sname], lat, lon, dataset = 'modis', resolution = 0.25)
            data['lat'] = lat; data['lon'] = lon
            print(data[sname].shape)

            # reduce scale of data to 1 deg x 1 deg
            if args.reduce_scale:
                data_new, _, _ = reduce_spatial_scale(data, sname)
            else:
                # Note at quarter degree resolution, the grid is 721 x 1440, which is difficult to work with since 721 only has two primes (721=7*103)
                # So remove the -90 degree point to make the grid easier (720 = 2^4*3^2*5)
                lat = data['lat'][:-1,:]; lon = data['lon'][:-1,:]
                data_new = data[sname][:,:-1,:] # Latitude may need to be reversed

            # Remove the unnecessary time axis
            data_new = data_new.squeeze()

            # Restore the land cover to integer labels
            data_new = np.round(data_new)

        elif ('type' in var) | (('cover' in var) & np.invert(var == 'land_cover')): # For 'types' (vegetation type, soil type, etc.), load an average of the classification in time
            sname = get_var_shortname(var)
            data_new = load_static_type(var, args.years[0], args.years[1], reduce_scale = args.reduce_scale)
        else:
            sname = get_var_shortname(var)
        
            # Get filename
            path = path_to_raw_datasets(var, 'era5')
            filename = get_fn(var, year)
            #filename = 'land'
            print(filename)
        
            # Load variable
            data = {}
            with Dataset('%s/%s'%(path, filename), 'r') as nc:
                data[sname] = nc.variables[sname][:,:,:]
                data['lat'] = lat; data['lon'] = lon

            # reduce scale of data to 1 deg x 1 deg
            if args.reduce_scale:
                data_new, lat, lon = reduce_spatial_scale(data, sname)
            else:
                # Note at quarter degree resolution, the grid is 721 x 1440, which is difficult to work with since 721 only has two primes (721=7*103)
                # So remove the -90 degree point to make the grid easier (720 = 2^4*3^2*5)
                lat = data['lat'][:-1,:]; lon = data['lon'][:-1,:]
                data_new = data[sname][:,:-1,:] # Latitude may need to be reversed

            data_new = data_new[0,:,:]

        if 'cover' not in var:
            data_new = np.round(data_new)
        #data_new[np.isnan(data_new) | (data_new == 0)] = 0.001

        print(np.nanmin(data_new), np.nanmax(data_new), np.nanmean(data_new))
        data_end[sname] = data_new

    # Calculate surface geopotential
    data_end['surface_geopotential_var'] = calculate_surface_potential(args.years[0], args.years[1], reduce_scale = args.reduce_scale)
    print(data_end['surface_geopotential_var'].shape, data_end['surface_geopotential_var'])
    print(np.nanmin(data_end['surface_geopotential_var']), np.nanmax(data_end['surface_geopotential_var']), np.nanmean(data_end['surface_geopotential_var']))

    static_variables = np.concatenate([args.static_variables, ['surface_geopotential_var']])

    # create variable entry in zarr
    make_zarr_group(data_end, lat, lon, None, static_variables, 2012, var_type = 'static', levels = args.pressure_levels)
    
    

def create_parser():
    '''
    Create an argument parser
    '''
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='Data preprocessing', fromfile_prefix_chars='@')

    parser.add_argument('--reduce_scale', action='store_true', help='Reduce spatial resolution by a quarter (useful for training test models)')
    parser.add_argument('--additional-reanalysis', action='store_true', help='Replace certain variables with other datasets (e.g., replace precipitation with IMERG data), otherwise use only ERA5')
    parser.add_argument('--nthreads', type=int, default=4, help='Number of working threads for multiprocesses tasks (in make_statistics, make_forcing, and make_static)')
    parser.add_argument('--datasets', type=str, default='era5', help='Include data from non-ERA5 sources. "few" = include other reanalyeses and climate indices, "all" = include other reanalyses, climate indices, and satellite data')

    parser.add_argument('--years', type=int, nargs=2, default=[2010,2020], help='Start and end years (inclusive) of climatology and statistics data')
    parser.add_argument('--year', type=int, default=0, help='Year of the data being processed (+ 2000)')

    parser.add_argument('--pressure_levels', type=int, nargs='+', default=[200,500], help='Pressure levels to use when processing upper air variables')
    parser.add_argument('--upper_air_variables', type=str, nargs='+', default=['u','v'], help='Upper air variables used in training (also predicted by the model)')
    parser.add_argument('--surface_variables', type=str, nargs='+', default=['temperature', 'precipitation', 'pressure'], help='Surface variables used in the model (also they output by the model)')
    parser.add_argument('--dynamic_variables', type=str, nargs='+', default=['radiation'], help='Variables used to force the model (input only variables)')
    parser.add_argument('--forcing_variables', type=str, nargs='+', default=['radiation'], help='Variables used to force the model (cyclic variables that vary on the annual scale, such as radiation; input only)')
    parser.add_argument('--static_variables', type=str, nargs='+', default=['land_sea'], help='Static variables that do no change in time (e.g., land-sea mask, land-cover, elevation, etc.)')
    parser.add_argument('--diagnostic_variables', type=str, nargs='+', default=['soil_moisture'], help='Variables the model will try to predict (output only)')

    parser.add_argument('--make_upper_air', action='store_true', help='Make upper air variable calculations and data creation (improves computation time; useful if the dataset is already present)')
    parser.add_argument('--make_surface', action='store_true', help='Make surface variable calculations and data creation (improves computation time; useful if the dataset is already present)')
    parser.add_argument('--make_dynamic', action='store_true', help='Make dynamic variable calculations and data creation (improves computation time; useful if the dataset is already present)')
    parser.add_argument('--make_forcing', action='store_true', help='Make forcing variable calculations and data creation (improves computation time; useful if the dataset is already present)')
    parser.add_argument('--make_static', action='store_true', help='Make static variable calculations and data creation (improves computation time; useful if the dataset is already present)')
    parser.add_argument('--make_diagnostic', action='store_true', help='Make diagnostic calculations and data creation (improves computation time; useful if the dataset is already present)')
    parser.add_argument('--make_climatology', action='store_true', help='Make a climatology dataset that contains the climatological mean (for each day of the year) for predicted variables')
    parser.add_argument('--make_statistics', action='store_true', help='Make the statics (means and standard deviation) nc files')


    return parser

    return


if __name__ == '__main__':
    # Parse and check incoming arguments
    parser = create_parser()
    args = parser.parse_args()

    args.year = args.year + 2000

    upper_air_variables = args.upper_air_variables # time x level x lat x lon
    surface_variables = args.surface_variables # time x lat x lon
    dynamic_variables = args.dynamic_variables # time x lat x lon
    diagnostic_variables = args.diagnostic_variables # time x lat x lon
    forcing_variables = args.forcing_variables # time (366) x lat x lon
    static_variables = args.static_variables # lat x lon

    levels = args.pressure_levels # hPa/mb

    years = np.arange(args.years[0], args.years[1]+1)
    # upper_air_variables = ['u', 'v'] # time x level x lat x lon
    # surface_variables = ['tair', 'sp', 'tp'] # time x lat x lon
    # dynamic_variables = ['ssr'] # time x lat x lon
    # diagnostic_variables = ['swvl1'] # time x lat x lon
    # forcing_variables = ['SSR'] # time (366) x lat x lon
    # static_variables = ['lsm'] # lat x lon
    # levels = [500, 200] # hPa/mb

    # years = [2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]
    N_years = len(years)

    # Make the climatology dataset
    if args.make_climatology and not args.make_statistics:
        make_climatology_dataset(args, make_climatology = True, make_statistics = False)
    elif args.make_statistics and not args.make_climatology:
        make_climatology_dataset(args, make_climatology = False, make_statistics = True)
    elif args.make_climatology and args.make_statistics:
        make_climatology_dataset(args, make_climatology = True, make_statistics = True)

    # Upper air variables
    if args.make_upper_air:
        make_upper_air_dataset(args)

    # Surface Variables
    if args.make_surface:
        make_dataset(args, var_type = 'surface')

    # Dynamic variables
    if args.make_dynamic:
        make_dataset(args, var_type = 'dynamic')

    # Diagnostic variables
    if args.make_diagnostic:
        make_dataset(args, var_type = 'diagnostic')


    # Forcing variables
    if args.make_forcing:
        make_forcing_dataset(args)

    # Static variables
    if args.make_static:
        make_static_dataset(args)

        
