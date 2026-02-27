'''
In the event MODIS .hdf data is downloaded and not processed, 
this script processes the .hdf data without the downloading step
'''

import os, warnings
import argparse
import requests
import time
import numpy as np
import re
import gc
import pyproj
import xarray
import multiprocessing as mp
from pyhdf.SD import SD, SDC
from lxml import etree
from io import StringIO
from netCDF4 import Dataset
from datetime import datetime, timedelta
from glob import glob
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker

# Most of the function incorporation will be the same as the modis_downloader, so just read in them instead of writing them anew
from modis_downloader import create_parser, load_global_grid, tile_to_grid, date_range, write_nc, test_map

def downscale_data(
        base_fname, 
        variable, 
        sname, 
        modis_sname, 
        timestamp, 
        start_time, 
        day_interval
        ) -> None:
    '''
    Function to load in MODIS data and downscale it to a global .05 deg x .05 deg grid

    Inputs:
    :param base_fname: Base filename of the MODIS .hdf file
    :param variable: Name of the variable in the MODIS file to downscale
    :param sname: Short name of the variable (i.e., the variable key to use in the written .nc file)
    :param modis_sname: Short name of the variable in the .hdf file (i.e., the variable key in the .hdf file)
    :param timestamp: Timestamp of the MODIS .hdf file being loaded (must be datetime)
    :param start_time: Starting time the year in the dataset (i.e., datetime of the year in the 
                       MODIS data is for, with month and day of 1/1; used to get number of days since start of year)
    :param day_interval: The day interval of the MODIS data (e.g., 8)
    '''

    # Determine the number of days since the start of year
    day_of_year = (timestamp - start_time).days + 1

    # Collect all the .hdf filenames for the global mosaic
    filenames = glob('%s%04d%03d*.hdf'%(base_fname, start_time.year, day_of_year), recursive = True)
    filenames = np.sort(filenames)

    # Check that .hdf files were downloaded
    if len(filenames) < 1:
        print('%04d-%02d-%02d is has no data downloaded!'%(timestamp.year, timestamp.month, timestamp.day))
        return

    # Construct the filename for the processed MODIS data, and check if it was already made
    filename = 'modis.%s.%d-day.%04d.%02d.%02d.nc'%(variable, day_interval, timestamp.year, timestamp.month, timestamp.day)
    if os.path.exists(filename):
        print('%s already exists. Skipping to the next timestamp'%filename)
        return 

    # Create the lat and lon for 0.05 degree x 0.05 degree grid
    resolution = 0.05
    lat = np.arange(-90, 90, resolution)
    lon = np.arange(-180, 180, resolution)
    
    # Initialize the grid
    I = lat.size
    J = lon.size

    # For reflectance, there are multiple bands to initialize
    if variable == 'reflectance':
        data = {}
        for sn in sname:
            data[sn] = np.ones((I, J), dtype = np.float32) * np.nan
    else:
        data = np.ones((I, J), dtype = np.float32) * np.nan

    data_full = []
    lat_full = []
    lon_full = []

    # Load the data and convert from the tile to a grid
    for filename in tqdm(filenames, desc = 'Interpolating tiles to grid for %04d-%02d-%02d'%(timestamp.year, timestamp.month, timestamp.day)):
        # Note the reflectance has multiple variables to save (multiple reflectance channels) rather than just one
        if variable == 'reflectance': 
            data_modis = {}
            for n, sn in enumerate(sname):
                # Load and convert each tile/.hdf file to a grid for each band
                # Note each grid only encompasses that tile's their respective domain
                data_modis[sn], lat_modis, lon_modis, long_name, units = tile_to_grid(filename, 
                                                                                      modis_sname[n], 
                                                                                      path = './')

                I_modis, J_modis = data_modis[sn].shape
        else:
            # Load and convert each tile/.hdf file to a grid for each band
            # Note each grid only encompasses that tile's their respective domain
            data_modis, lat_modis, lon_modis, long_name, units = tile_to_grid(filename, 
                                                                              modis_sname[0], 
                                                                              path = './')
                
            I_modis, J_modis = data_modis.shape

        # Select the latitudes of the tile from the global grid
        lat_ind = np.where((lat >= np.nanmin(lat_modis)) & (lat <= np.nanmax(lat_modis)))[0]
        lon_ind = np.where((lon >= np.nanmin(lon_modis)) & (lon <= np.nanmax(lon_modis)))[0]

        # Select out the longitudes
        if (lon_modis < 0).any() & (lon_modis > 0).any():
        #if len(lon_ind) == lon.size:
            lon_neg = np.where(lon_modis < 0, lon_modis, np.nan)
            
            lon_pos = np.where(lon_modis > 0, lon_modis, np.nan)
            
            lon_ind = np.where((lon <= np.nanmax(lon_neg)) | (lon >= np.nanmin(lon_pos)))[0]
        
        # Select out the latitude and longitude values for the loaded tile
        lat_sub = lat[lat_ind]
        lon_sub = lon[lon_ind]

        # Mesh the latitude and longitudes
        I = lat_sub.size
        J = lon_sub.size
        lon_sub, lat_sub = np.meshgrid(lon_sub, lat_sub)

        # Interpolate to 0.05 degree x 0.05 degree grid
        for n, ind in enumerate(lat_ind):
            #print(n)
            # Select all the MODIS latitudes between the current two latitude lines to interpolate to
            la_ind = np.where( (lat_modis[:,0] >= lat_sub[n,0]) & (lat_modis[:,0] < (lat_sub[n,0] + resolution)) )[0]

            lon_tmp = np.nanmean(lon_modis[la_ind,:], axis = 0)

            # Perform a mean along the latitude line (reducing MODIS data to 0.05 degree along latitude)
            if variable == 'reflectance':
                data_tmp = {}
                for sn in sname:
                    data_tmp[sn] = np.nanmean(data_modis[sn][la_ind,:], axis = 0)
            else:
                data_tmp = np.nanmean(data_modis[la_ind,:], axis = 0)
            
            if np.invert(lon_tmp.size == np.unique(lon_tmp).size):
                ### NOTE: This seems to only occur at at specific tiles, and this method with the 
                ### interpolation later on seems to cause strange distortions near the 70˚ latitude line.
                print('Not all lon are unique - collecting unique longitudes only')
                res_ind = [m for m, lo in enumerate(lon_tmp) if lo not in lon_tmp[:m]] # Collects indices for non-repeating longitudes
                
                # Select out the non-repeating indices
                lon_tmp = lon_tmp[res_ind[:]]
                if variable == 'reflectance':
                    for sn in sname:
                        data_tmp[sn] = data_tmp[sn][res_ind[:]]
                else:
                    data_tmp = data_tmp[res_ind[:]]
            
            # Interpolate along the longitude line
            if variable == 'reflectance':
                for sn in sname:
                    # Create the data as an xarray dataset
                    ds = xarray.Dataset(data_vars = dict(modis_data = (['x'], data_tmp[sn])), 
                                        coords = dict(x = (['x'], lon_tmp)) )
                    
                    # Interpolate along longitude (faster and more efficient 
                    # than looping over longitudes to average)
                    ds_interp = ds.interp(coords = dict(x = (['lon'], lon_sub[n,:])) )
                    
                    # Return the interpolated data to np.ndarray and add to the 0.05 deg x 0.05 grid
                    tmp = np.array([data[sn][ind,lon_ind], ds_interp.modis_data.values[:]])
                    data[sn][ind,lon_ind] = np.nanmean(tmp, axis = 0)
            else:
                # Create the data as an xarray dataset
                ds = xarray.Dataset(data_vars = dict(modis_data = (['x'], data_tmp)), 
                                    coords = dict(x = (['x'], lon_tmp)) )
                #print(np.nanmin(ds.modis_data.values), np.nanmax(ds.modis_data.values))
                #print(ds)
                        
                #if np.invert(np.sum(np.isnan(data_tmp)) == data_tmp.size):
                #    print('In if block')
                #    ds = ds.dropna(dim = 'x')
                #    print(ds)
                #for m, lo_ind in enumerate(lon_ind):
                #    lon_index = np.where( (lon_tmp >= lon_sub[0,m]) & (lon_tmp < (lon_sub[0,m] + resolution)) )[0]
                #    print(lon_index)
                #    print(lon_index.size)
                
                # Interpolate along longitude (faster and more efficient 
                # than looping over longitudes to average)
                ds_interp = ds.interp(coords = dict(x = (['lon'], lon_sub[n,:])) )
                
                # Return the interpolated data to np.ndarray and add to the 0.05 deg x 0.05 grid
                tmp = np.array([data[ind,lon_ind], ds_interp.modis_data.values[:]])
                data[ind,lon_ind] = np.nanmean(tmp, axis = 0)
            
        del ds, ds_interp
        gc.collect()

    # Mesh the lat and lon for the 0.05 x 0.05 degree grid
    lon, lat = np.meshgrid(lon, lat)
        
    print('Writing data...')

    # Write the data 
    filename = 'modis.%s.%d-day.%04d.%02d.%02d.nc'%(variable, day_interval, timestamp.year, timestamp.month, timestamp.day)
    description = 'Global MODIS %s (%s) for an %d-day period starting at %s, and averaged down to 0.05 degree x 0.05 degree resolution.'%(long_name, units, args.day_interval, timestamp.strftime('%b %d, %Y'))
    write_nc(data, 
             lat, 
             lon, 
             [timestamp], 
             filename = filename, 
             var_sname = sname, 
             description = description)

    # Remove potentially large and un-needed data to free up space
    del data, lat, lon#, data_full, lat_full, lon_full
    gc.collect()

    # Take a short rest
    time.sleep(10)


if __name__ == '__main__':
    # Parse and check incoming arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # Turn off warnings
    warnings.simplefilter('ignore')

    print('Initializing variables...')
    # Determine the base filename of the .hdf file based on variable (and shortname for extracting data)
    if (args.variables[0] == 'evaporation') | (args.variables[0] == 'potential_evaporation'):
        base_fname = 'MOD16A2GF.061'
        sname = 'evap' if args.variables[0] == 'evap' else 'pevap'
    elif (args.variables[0] == 'fpar') | (args.variables[0] == 'lai'):
        base_fname = 'MOD15A2H.A'
        # base_fname = 'MCD15A2H.061'
        sname = 'fpar' if args.variables[0] == 'fpar' else 'lai'
    elif (args.variables[0] == 'ndvi') | (args.variables[0] == 'evi'):
        base_fname = 'MCD19A3CMG.061'
        sname = 'ndvi' if args.variables[0] == 'ndvi' else 'evi'
    elif (args.variables[0] == 'dsr'):
        base_fname = 'MCD18C1.062'
        sname = 'dsr'
    elif (args.variables[0] == 'reflectance'):
        base_fname = 'MOD09A1.A' # Used to find the filename in the string
        sname = ['sur_refl_b01', 'sur_refl_b02', 'sur_relf_b03', 'sur_relf_b04', 
                 'sur_relf_b05', 'sur_relf_b06', 'sur_relf_b07']

    # Determine all years to perform the interpolation for
    years = np.arange(args.start_year+2000, args.end_year+1+2000)

    print('Processing MODIS data...')
    for year in years: # Need to loop through each year as the timesteps reset to Jan. 1 at the start of each year

        # Construct the time stamp information for each .hdf file
        start_time = datetime(year, 1, 1)
        end_time   = datetime(year, 12, 31)
        timestamps = date_range(start_time, end_time)
    
        timestamps = np.array([timestamp for timestamp in timestamps])

        # Make the parameters for multiprocessing
        param_args = [(base_fname, args.variables[0], 
                       sname, args.var_snames_modis, 
                       timestamp, 
                       start_time, 
                       args.day_interval) for timestamp in timestamps[::args.day_interval]]

        # Use multiprocessing to load tiles and processes them onto a single, downscaled grid
        with mp.Pool(args.nprocesses) as pool:
            data = pool.starmap(downscale_data, param_args) # Load and process data
        
        
        # for n, timestamp in enumerate(timestamps[::args.day_interval]):
        #     day_of_year = (timestamp - start_time).days + 1

        #     filenames = glob('%s%04d%03d*.hdf'%(base_fname, start_time.year, day_of_year), recursive = True)
        #     filenames = np.sort(filenames)

        #     if len(filenames) < 1:
        #         print('%04d-%02d-%02d is has no data downloaded!'%(timestamp.year, timestamp.month, timestamp.day))
        #         continue

        #     # Create the lat and lon for 0.05 degree x 0.05 degree grid
        #     resolution = 0.05
        #     lat = np.arange(-90, 90, resolution)
        #     lon = np.arange(-180, 180, resolution)
            
        #     # Initialize the grid
        #     I = lat.size
        #     J = lon.size
        #     if args.variables[0] == 'reflectance':
        #         data = {}
        #         for sn in sname:
        #             data[sn] = np.ones((I, J), dtype = np.float32) * np.nan
        #     else:
        #         data = np.ones((I, J), dtype = np.float32) * np.nan
    
        #     data_full = []
        #     lat_full = []
        #     lon_full = []

        #     # Load the data and convert from the tile to a grid
        #     for filename in tqdm(filenames, desc = 'Interpolating tiles to grid for %04d-%02d-%02d'%(timestamp.year, timestamp.month, timestamp.day)):
        #         if args.variables[0] == 'reflectance':
        #             data_modis = {}
        #             for n, sn in enumerate(sname):
        #                 data_modis[sn], lat_modis, lon_modis, long_name, units = tile_to_grid(filename, args.var_snames_modis[n], path = './')

        #                 I_modis, J_modis = data_modis[sn].shape
        #         else:
        #             data_modis, lat_modis, lon_modis, long_name, units = tile_to_grid(filename, args.var_snames_modis[0], path = './')
                        
        #             I_modis, J_modis = data_modis.shape

        #         # Select the latitudes from the global grid
        #         lat_ind = np.where((lat >= np.nanmin(lat_modis)) & (lat <= np.nanmax(lat_modis)))[0]
        #         lon_ind = np.where((lon >= np.nanmin(lon_modis)) & (lon <= np.nanmax(lon_modis)))[0]

        #         # Select out the longitudes
        #         if (lon_modis < 0).any() & (lon_modis > 0).any():
        #         #if len(lon_ind) == lon.size:
        #             lon_neg = np.where(lon_modis < 0, lon_modis, np.nan)
                    
        #             lon_pos = np.where(lon_modis > 0, lon_modis, np.nan)
                    
        #             lon_ind = np.where((lon <= np.nanmax(lon_neg)) | (lon >= np.nanmin(lon_pos)))[0]
                
        #         lat_sub = lat[lat_ind]
        #         lon_sub = lon[lon_ind]

        #         # Interpolate to 0.05 degree x 0.05 degree grid
        #         I = lat_sub.size
        #         J = lon_sub.size
        #         lon_sub, lat_sub = np.meshgrid(lon_sub, lat_sub)

        #         # Might be more efficient: convert data to an xarray, select out lat_ind, lon_ind, and set data.select(coords = [lat_ind, lon_ind]) = ds.interp.modis_data.values
        #         for n, ind in enumerate(lat_ind):
        #             #print(n)
        #             la_ind = np.where( (lat_modis[:,0] >= lat_sub[n,0]) & (lat_modis[:,0] < (lat_sub[n,0] + resolution)) )[0]

        #             lon_tmp = np.nanmean(lon_modis[la_ind,:], axis = 0)

        #             if args.variables[0] == 'reflectance':
        #                 data_tmp = {}
        #                 for sn in sname:
        #                     data_tmp[sn] = np.nanmean(data_modis[sn][la_ind,:], axis = 0)
        #             else:
        #                 data_tmp = np.nanmean(data_modis[la_ind,:], axis = 0)
                    
        #             if np.invert(lon_tmp.size == np.unique(lon_tmp).size):
        #                 ### NOTE: This seems to only occur at at specific tiles, and this method with the 
        #                 ### interpolation later on seems to cause strange distortions along the 70˚ latitude line.
        #                 print('Not all lon are unique - collecting unique longitudes only')
        #                 res_ind = [m for m, lo in enumerate(lon_tmp) if lo not in lon_tmp[:m]] # Collects indices for non-repeating longitudes
                        
        #                 # Select out the non-repeating indices
        #                 lon_tmp = lon_tmp[res_ind[:]]
        #                 if args.variables[0] == 'reflectance':
        #                     for sn in sname:
        #                         data_tmp[sn] = data_tmp[sn][res_ind[:]]
        #                 else:
        #                     data_tmp = data_tmp[res_ind[:]]
                    
        #             if args.variables[0] == 'reflectance':
        #                 for sn in sname:
        #                     ds = xarray.Dataset(data_vars = dict(modis_data = (['x'], data_tmp[sn])), 
        #                                     coords = dict(x = (['x'], lon_tmp)) )
                            
        #                     ds_interp = ds.interp(coords = dict(x = (['lon'], lon_sub[n,:])) )
                            
        #                     tmp = np.array([data[sn][ind,lon_ind], ds_interp.modis_data.values[:]])
        #                     data[sn][ind,lon_ind] = np.nanmean(tmp, axis = 0)
        #             else:
        #                 ds = xarray.Dataset(data_vars = dict(modis_data = (['x'], data_tmp)), 
        #                                     coords = dict(x = (['x'], lon_tmp)) )
        #                 #print(np.nanmin(ds.modis_data.values), np.nanmax(ds.modis_data.values))
        #                 #print(ds)
                                
        #                 #if np.invert(np.sum(np.isnan(data_tmp)) == data_tmp.size):
        #                 #    print('In if block')
        #                 #    ds = ds.dropna(dim = 'x')
        #                 #    print(ds)
        #                 #for m, lo_ind in enumerate(lon_ind):
        #                 #    lon_index = np.where( (lon_tmp >= lon_sub[0,m]) & (lon_tmp < (lon_sub[0,m] + resolution)) )[0]
        #                 #    print(lon_index)
        #                 #    print(lon_index.size)
                        
        #                 ds_interp = ds.interp(coords = dict(x = (['lon'], lon_sub[n,:])) )
        #                 #print(np.nanmin(ds_interp.modis_data.values), np.nanmax(ds_interp.modis_data.values))
        #                 #print(ds_interp)
                        
        #                 tmp = np.array([data[ind,lon_ind], ds_interp.modis_data.values[:]])
        #                 data[ind,lon_ind] = np.nanmean(tmp, axis = 0)
                    
        #         del ds, ds_interp
        #         gc.collect()

        #     lon, lat = np.meshgrid(lon, lat)
                
        #     print('Writing data...')

        #     # Write the data 
        #     filename = 'modis.%s.%d-day.%04d.%02d.%02d.nc'%(args.variables[0], args.day_interval, timestamp.year, timestamp.month, timestamp.day)
        #     description = 'Global MODIS %s (%s) for an %d-day period starting at %s, and averaged down to 0.05 degree x 0.05 degree resolution.'%(long_name, units, args.day_interval, timestamp.strftime('%b %d, %Y'))
        #     write_nc(data, lat, lon, [timestamp], filename = filename, var_sname = sname, description = description)

        #     del data, lat, lon#, data_full, lat_full, lon_full
        #     gc.collect()
        
        #     # Take a short rest
        #     time.sleep(10)