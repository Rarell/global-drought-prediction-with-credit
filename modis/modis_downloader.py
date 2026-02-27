'''
Script to download MODIS data from Earth Data HTTPs,
save it on a standard grid in a netcdf format 
(1 timestamp of data over the globe is ~ # GB per variable)

NOTE: Earth Data requires authentication to access data. 
      Earth Data login credentials in the .netrc file MUST be up to date for the script to run
NOTE: For MODIS data has been moved since Lines 439 - 450 were last run, and those lines need 
      to be updated and and base_fname needs to be added to them before they will work

Acceptable names for --variables and --var_snames_gldas (i.e., variable names in the downloaded GLDAS dataset):
    --variables:
    - Evaporation: evaporation
    - Potential Evaporation: potential_evaporation
    - NDVI: ndvi
    - EVI: evi
    - FPAR: fpar
    - LAI: lai
    - Downward shortwave radiation: dsr
    - Land cover: land_cover
    
    --var_snames_modis:
    - Evaporation: ET_500m
    - Potential Evaporation: PET_500m
    - NDVI: NDVI
    - EVI: EVI
    - FPAR: Fpar_500m
    - LAI: Lai_500m
    - Downward shortwave radiation: DSR # For DSR, this would be modified in the code to 
                                          'GMT_HHHH_%s'%args.variables format to collect DSR at all hours
    - Land cover: Majority_Land_Cover_Type_1
    
Variable units in MODIS dataset:
    - Evaporation: kg m^-2 8-day^-1
    - Potential Evaporation: kg m^-2 8-day^-1
    - NDVI: Dimensionless
    - EVI: Dimensionless
    - FPAR: Dimensionless
    - LAI: Dimensionless
    - Downward shortwave radiation: W m^-2
    - Land cover: Unitless
    
TODO:
 - Loop through multiple variables per set of MODIS files
 - Look into making grid_to_tile
 - Create lat/lon in load_global_grid()
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
from typing import Tuple
from pyhdf.SD import SD, SDC
from lxml import etree
from io import StringIO
from netCDF4 import Dataset
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker


#### Example of loading and looking at .hdf4 data

# Load the data
# file = SD('MCD19A3CMG.A2000080.061.2022315223048.hdf', SDC.READ)

# Examine the variables
# print(file.datasets())
# {'NDVI': (('YDim:CMGgrid', 'XDim:CMGgrid'), (3600, 7200), 22, 0), 'NDVI_n': (('YDim:CMGgrid', 'XDim:CMGgrid'), (3600, 7200), 22, 1), 'EVI': (('YDim:CMGgrid', 'XDim:CMGgrid'), (3600, 7200), 22, 2), 'EVI_n': (('YDim:CMGgrid', 'XDim:CMGgrid'), (3600, 7200), 22, 3), 'NDVI_gapfill': (('YDim:CMGgrid', 'XDim:CMGgrid'), (3600, 7200), 22, 4)}

# Select the data
# One set: 'NDVI', 'EVI', another: 'GMT_0000_DSR', 'GMT_0300_DSR', 'GMT_0600_DSR', 'GMT_0900_DSR', 'GMT_1200_DSR', 'GMT_1500_DSR', 'GMT_1800_DSR', 'GMT_2100_DSR'
# data = file.select('NDVI')

# Get the data dimensions
# print(data.dimensions())
# {'YDim:CMGgrid': 3600, 'XDim:CMGgrid': 7200}

# Get the attributes of the data
# print(data.attributes())
# {'longname': 'Daily Normalized difference vegetation index at surface', '_FillValue': -28672, 'scale_factor': 0.0001, 'valid_range': [0, 10000]}

# print(data.info())
# ('NDVI', 2, [3600, 7200], 22, 4)

# Get the data
# data.get()
# data = np.array(data)

# Close the data files
# data.endaccess()
# file.end()


# Function to create a parser using the terminal
def create_parser():
    '''
    Create argument parser
    '''
    
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='MODIS Downloader', fromfile_prefix_chars='@')

    parser.add_argument('--variables', type=str, nargs='+', default=['evapotranspiration'], help='Variables to download')
    parser.add_argument('--var_snames_modis', type=str, nargs='+', default=['ET_500m'], help='The short name for --variables used in raw datafile')
    parser.add_argument('--day_interval', type=int, default=8, help='Number of days between each MODIS scan')
    parser.add_argument('--start_year', type=int, default=2000, help='Beginning year dat to be downloaded.')
    parser.add_argument('--end_year', type=int, default=2023, help='Ending year to be downloaded.')
    parser.add_argument('--save_raw', action='store_true', help='Save the raw data at the higher resolution (can be up to 20 GB) as well as the lower resolution .nc file.')
    parser.add_argument('--nprocesses', type=int, default=1, help='Number of processes for using multiprocessing in the data processing step')
    
    return parser
    
def load_global_grid(
        filename, 
        var, 
        resolution = 0.05, 
        path = './'
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, str]:
    '''
    Load a global .hdf file in which the data is over the full globe.
    Also load some of the metadata for the file.
    
    Code is heavily based off of the work done by Dr. John Evans 
    in https://hdfeos.org/zoo/MORE/LPDAAC/MCD/MCD19A2.A2010010.h25v06.006.2018047103710.hdf.py
    
    Inputs:
    :param filename: Filename of the hdf file to be loaded
    :param var: Name of the data (in the hdf file) being loaded from the hdf file
                E.g., in the MODIS dataset, IGBP land cover is labeled as Majority_Land_Cover_Type_1, 
                so var = 'Majority_Land_Cover_Type_1' in that file
    :param resolution: Spatial resolution of the global grid 
                       (e.g., if the grid has 0.05 degree x 0.05 degree, then resolution = 0.05)
    :param path: Directory path to the directory with the hdf file
    
    Outputs:
    :param data: hdf data in a numpy array, with attributes (scale factor, offset) 
                 applied and fill values converted to NaNs (np.ndarray with shape lat x lon)
    :param lat: Latitudes coordinates for each gridded entry in data (np.ndarray with shape lat x lon)
    :param lon: Longitude coordinates for each gridded entry in data (np.ndarray with shape lat x lon)
    :param long_name: Long name of the variable that data represents
    :param units: Units of data
    '''
    
    # Load the data
    file = SD('%s/%s'%(path, filename), SDC.READ)
        
    # Select the data
    data_holder = file.select(var) # shortname, or attributes.keys()[...]
        
    # Get the data
    data_attrs = data_holder.attributes(full = 1)
    data = data_holder.get()
    data = np.array(data)
    data = data.astype(np.float32) # compress the data to take less size
            
    # Collect data attributes
    long_name = data_attrs['long_name'][0]
    fill_value = data_attrs['_FillValue'][0]
    scale_factor = data_attrs['scale_factor'][0]
    units = data_attrs['unit'][0]
    offset = data_attrs['add_offset'][0]
    valid_range = data_attrs['valid_range'][0]
            
    # Apply attributes
    invalid = np.where(((data < valid_range[0]) | (data > valid_range[1])) | (data == fill_value))[0]
    data[invalid] = np.nan
    data = (data - offset) * scale_factor

    # Create the lat/lon grid using the given resolution
    lat = np.arange(-90, 90, resolution)
    lon = np.arange(-180, 180, resolution)
    lon, lat = np.meshgrid(lon, lat)
        
    # Close files
    data_holder.endaccess()
    file.end()
    
    return data, lat, lon, long_name, units
    
def tile_to_grid(
        filename, 
        var,
        path = './'
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, str]:
    '''
    Load a .hdf file in a sinusoidal grid and convert it onto a 
    standard grid in which x and y represent lon and lat respectively.

    Primary purpose of the function: convert MODIS data from the tile schema onto a standard grid.
    Also loads some of the metadata for the file.
    
    Code is heavily based off of the work done by Dr. John Evans 
    in https://hdfeos.org/zoo/MORE/LPDAAC/MCD/MCD19A2.A2010010.h25v06.006.2018047103710.hdf.py
    
    Inputs:
    :param filename: Filename of the hdf file to be loaded
    :param var: Name of the variable (in the hdf file) being loaded
                E.g., in the MODIS dataset, 500m resolution ET is 
                labeled as ET_500m, so var = 'ET_500m' in that file
    :param path: Directory path to the directory with the hdf file
    
    Outputs:
    :param data: hdf data in a numpy array, with attributes (scale factor, offset) 
                 applied and fill values converted to NaNs (np.ndarray with shape lat x lon)
    :param lat: Latitudes coordinates for each gridded entry in data (np.ndarray with shape lat x lon)
    :param lon: Longitude coordinates for each gridded entry in data (np.ndarray with shpe lat x lon)
    :param long_name: Long name of the variable that data represents
    :param units: Units of data
    '''
    
    # Load the data
    file = SD('%s/%s'%(path, filename), SDC.READ)
            
    # Select the data
    data_holder = file.select(var) # shortname, or attributes.keys()[...]
            
    # Get the data
    data_attrs = data_holder.attributes(full = 1)
    data = data_holder.get()
    data = np.array(data)
    data = data.astype(np.float32) # compress the data to take less size
            
    # Collect data attributes
    long_name = data_attrs['long_name'][0]
    fill_value = data_attrs['_FillValue'][0]
    scale_factor = data_attrs['scale_factor'][0]
    units = data_attrs['units'][0]
    offset = data_attrs['add_offset'][0]
    valid_range = data_attrs['valid_range'][0]
    
    # Prevent NaN offsets from turning the whole dataset into NaNs
    if np.isnan(offset):
        offset = 0
            
    # Apply attributes
    #invalid = np.where(((data < valid_range[0]) | (data > valid_range[1])) | (data == fill_value))[0]
    invalid = np.where((data == fill_value))[0]
    data = np.where( ((data < valid_range[0]) | (data > valid_range[1])) | (data == fill_value), 
                    np.nan, data)
    data = (data - offset) * scale_factor
            
    # Get the MODIS grid...
    fattrs = file.attributes(full = 1)
    ga = fattrs['StructMetadata.0'] # Collect metadata
    grid_meta = ga[0]
    ul_regex = re.compile(r'''UpperLeftPointMtrs=\(
                              (?P<upper_left_x>[+-]?\d+\.\d+)
                              ,
                              (?P<upper_left_y>[+-]?\d+\.\d+)
                              \)''', re.VERBOSE) # String corresponding to upper left coordinates in metadata
    match = ul_regex.search(grid_meta) # Get and create the upper left coordinates
    x0 = np.float32(match.group('upper_left_x'))
    y0 = np.float32(match.group('upper_left_y'))
    lr_regex = re.compile(r'''LowerRightMtrs=\(
                              (?P<lower_right_x>[+-]?\d+\.\d+)
                              ,
                              (?P<lower_right_y>[+-]?\d+\.\d+)
                              \)''', re.VERBOSE) # String corresponding to lower right coordinates in metadata
    match = lr_regex.search(grid_meta) # Get and create the lower right coordinates
    x1 = np.float32(match.group('lower_right_x'))
    y1 = np.float32(match.group('lower_right_y'))
                
    # Create XDim and YDim with the upper left and lower right coordinates
    nx, ny = data.shape
                
    x = np.linspace(x0, x1, nx, endpoint=False)
    y = np.linspace(y0, y1, ny, endpoint=False)

    xv, yv = np.meshgrid(x, y)
    
    # Create sinusoidal and standard (WGS84) projections
    sinu = pyproj.Proj("+proj=sinu +R=6371007.181 +nadgrids=@null +wktext")
    wgs84 = pyproj.Proj("EPSG:4326")
    
    # Convert the coordinates from sinusoidal to standard coordinates 
    # (where x and y now represent lon and lat respectively)

    ### NOTE: This method of grid transformation is out of date (delivers warnings from pyproj) 
    #         and may be truncated/need to be changed in future
    ### Newer method might be sinu_to_wgs = pyproj.Transformer.from_proj(old_proj, new_proj) -> sinu_to_wgs.transform(xv, yv)
    lat, lon= pyproj.transform(sinu, wgs84, xv, yv)
    
    # Ensure data size is compressed (full float length should not be needed)
    data = data.astype(np.float32)
    lon = lon.astype(np.float32)
    lat = lat.astype(np.float32)
            
    # Close files
    data_holder.endaccess()
    file.end()
    
    return data, lat, lon, long_name, units
    
def date_range(start_date, end_date):
    '''
    This function takes in two dates and outputs all the dates inbetween
    those two dates.
    
    Inputs:
    :param start_Date: Starting date of the interval (must be a datetime)
    :param end_date: Ending date of the interval (must be a datetime)
        
    Outputs:
    - A generator of all dates between StartDate and EndDate (inclusive)
    '''
    for n in range(int((end_date - start_date).days) + 1):
        yield start_date + timedelta(n)
         
def write_nc(
        data, 
        lat, 
        lon, 
        dates, 
        mask = None, 
        filename = 'tmp.nc', 
        var_sname = 'tmp', 
        description = 'Description', 
        path = './'
        ) -> None:
    '''
    Write data, and additional information such as latitude and longitude and timestamps, to a .nc file.
    
    Inputs:
    :param data: Variable being written (np.ndarray with shape time x lat x lon)
    :param lat: Latitude labels (np.ndarray with shape lat x lon)
    :param lon: Longitude labels (np.ndarray with shape lat x lon)
    :param dates: Timestamps for data in a %Y-%m-%d format (np.ndarray with shape time)
    :param mask: Land-sea mask to be added to the dataset
    :param filename: Filename of the .nc file being written
    :param var_sname: Short name of the variable being written (i.e., variable key in the .nc file)
    :param description: A string describing the data
    :param path: Directory path to the directory the data will be written in
    '''
    
    # Determine the spatial and temporal lengths
    if type(var_sname) is list:
        I, J = data[var_sname[0]].shape
    else:
        I, J = data.shape

    T = len(dates)
    
    with Dataset(path + filename, 'w', format = 'NETCDF4') as nc:
        # Write a description for the .nc file
        nc.description = description

        
        # Create the spatial and temporal dimensions
        nc.createDimension('lat', size = I)
        nc.createDimension('lon', size = J)
        nc.createDimension('time', size = T)
        
        # Create the lat and lon variables       
        nc.createVariable('lat', lat.dtype, ('lat', 'lon'))
        nc.createVariable('lon', lon.dtype, ('lat', 'lon'))
        
        nc.variables['lat'][:,:] = lat[:,:]
        nc.variables['lon'][:,:] = lon[:,:]
        
        # Create the date variable
        nc.createVariable('date', str, ('time', ))
        for n in range(len(dates)):
            nc.variables['date'][n] = str(dates[n])
            
        # Create the main variable
        if type(var_sname) is list:
            for sn in var_sname:
                nc.createVariable(sn, data[sn].dtype, ('lat', 'lon'))
                nc.variables[str(sn)][:,:] = data[sn][:,:]
        else:
            nc.createVariable(var_sname, data.dtype, ('lat', 'lon'))
            nc.variables[str(var_sname)][:,:] = data[:,:]
        
        if np.invert(mask is None):
             nc.createVariable('landmask', mask.dtype, ('lat', 'lon'))
             nc.variables['landmask'][:,:] = mask[:,:]

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
    :param dates: Datetime corresponding to the present timestamp of data
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
    if (np.ceil(np.nanmin(data[:,:])) == 1) & (np.floor(np.nanmax(data[:,:])) == 0): # Special case if the variable varies from 0 to 1
        cmin = np.round(np.nanmin(data[:,:]), 2); cmax = np.round(np.nanmax(data[:,:]), 2); cint = (cmax - cmin)/100
    else:
        cmin = np.ceil(np.nanmin(data[:,:])); cmax = np.floor(np.nanmax(data[:,:])); cint = (cmax - cmin)/100
    
    clevs = np.arange(cmin, cmax+cint, cint)
    nlevs = len(clevs) - 1
    cmap  = plt.get_cmap(name = 'BrBG', lut = nlevs)
    
    data_proj = ccrs.PlateCarree()
    fig_proj  = ccrs.PlateCarree()

    # Create the figure
    fig = plt.figure(figsize = [12, 16])
    ax = fig.add_subplot(1, 1, 1, projection = fig_proj)

    # Make the title
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
    cbar = fig.colorbar(cs, cax = cbax, orientation = 'horizontal')

    # Set the extent of the map
    ax.set_extent([np.nanmin(lon), np.nanmax(lon), np.nanmin(lat), np.nanmax(lat)], 
                    crs = fig_proj)
    
    # Save the figure
    plt.savefig('%s_%s_test_map.png'%(data_name, date.strftime('%Y-%m-%d')))

    # Close the figure
    #plt.show(block = False)
    plt.close('all')




#### Begin main program
if __name__ == '__main__':
    # Parse and check incoming arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # Turn off warnings
    warnings.simplefilter('ignore')

    print('Initializing variables...')
    # Determine the base filename based on variable (and shortname for extracting data)
    if (args.variables[0] == 'evaporation') | (args.variables[0] == 'potential_evaporation'):
        url_modifier = 'MOD16A2GF.061'
        sname = 'evap' if args.variables[0] == 'evap' else 'pevap'
    elif (args.variables[0] == 'fpar') | (args.variables[0] == 'lai'):
        url_modifier = 'MCD15A2H.061'
        sname = 'fpar' if args.variables[0] == 'fpar' else 'lai'
    elif (args.variables[0] == 'ndvi') | (args.variables[0] == 'evi'):
        url_modifier = 'MCD19A3CMG.061'
        sname = 'ndvi' if args.variables[0] == 'ndvi' else 'evi'
    elif (args.variables[0] == 'dsr'):
        url_modifier = 'MCD18C1.062'
        sname = 'dsr'
    elif (args.variables[0] == 'reflectance'):
        url_modifier = 'hyrax/DP131/MOLT/MOD09A1.061'
        base_fname = 'MOD09A1.A2' # Used to find the filename in the string
        sname = ['sur_refl_b01', 'sur_refl_b02', 'sur_relf_b03', 'sur_relf_b04', 
                 'sur_relf_b05', 'sur_relf_b06', 'sur_relf_b07']

    # Determine the years to download and process data for
    years = np.arange(args.start_year+2000, args.end_year+1+2000)
    print('Starting data download...')
    for year in years: # Need to loop through each year as the timesteps reset to Jan. 1 at the start of each year
        # Construct the timestamps for the MODIS files to download and process
        start_time = datetime(year, 1, 1)
        end_time   = datetime(year, 12, 31)
        timestamps = date_range(start_time, end_time)
    
        timestamps = np.array([timestamp for timestamp in timestamps])
        
        # Download data for each timestamp
        for n, timestamp in enumerate(timestamps[::args.day_interval]):
            print('On day %s'%timestamp.strftime('%b %d, %Y'))
        
            # Make the base url to find the data
            # url_base = 'https://e4ftl01.cr.usgs.gov/MOLT/%s/%s/'%(url_modifier, timestamp.strftime('%Y.%m.%d'))
            # New url location for MODIS data
            url_base = 'https://opendap.cr.usgs.gov/opendap/%s/%s/'%(url_modifier, timestamp.strftime('%Y.%m.%d')) 
            print(url_base)

            # Collect all the files with the MODIS data
            ## Heavily based on the code developed by Dr. Maas in https://www.matecdev.com/posts/login-download-files-python.html
            session = requests.Session()
            page = session.get(url_base) # Get and decode all filenames on the url page
            html = page.content.decode('utf-8')

            # Encoding string needs to be removed for lxml to work
            encoding_string = '<?xml version="1.0" encoding="UTF-8"?>' 
            html = html.replace(encoding_string, "")

            # Parse filenames
            tree = etree.parse(StringIO(html), parser = etree.HTMLParser(encoding = 'utf-8'))
            html = html.replace(encoding_string, "")

            # Split the string into multiple strings, one for each file (set each string as a list entry)
            refs = tree.xpath('//a') 
            link_list = list(set([link.get('href', '') for link in refs]))

            # Filter out all files to only include .hdf files
            extension = '.hdf' 
            filenames = [link for link in link_list if link.endswith(extension)]
            print(filenames[:10])
            ##
            
            # Create the lat and lon for 0.05 degree x 0.05 degree grid
            resolution = 0.05
            lat = np.arange(-90, 90, resolution)
            lon = np.arange(-180, 180, resolution)
            
            # Initialize the grid
            I = lat.size
            J = lon.size

            # For reflectance, multiple bands need to be initialized
            if args.variables[0] == 'reflectance':
                data = {}
                for sn in sname:
                    data[sn] = np.ones((I, J), dtype = np.float32) * np.nan
            else:
                data = np.ones((I, J), dtype = np.float32) * np.nan
    
            data_full = []
            lat_full = []
            lon_full = []
    
            # Download the global data
            if len(filenames) < 2:
                # New MODIS site attaches a lot of extra to the fname string, this gets rid of it
                ind = filenames[0].find(base_fname) 

                # Check if the file has already been downloaded (load it if so)
                if os.path.exists('./raw/%s.dap.nc4'%filenames[0][ind:]):
                    # Processed file does exist: skip the download step
                    print("%s is already downloaded."%filenames[0])
                else:
        
                    # If there is only 1 hdf file, the dataset has already been merged into a single, global set
                    # Collect that global set
                    response = session.get('%s/%s.dap.nc4'%(url_base, filenames[0][ind:]))
                    open('%s/%s.dap.nc4'%('./raw', filenames[0][ind:]), 'wb').write(response.content)
        
                # Load the data
                data_modis, lat_modis, lon_modis, long_name, units = load_global_grid(filenames[0][ind:], 
                                                                                      args.var_snames_modis[0], 
                                                                                      resolution = resolution, 
                                                                                      path = './raw')
        
                # Get the keys
                #attributes = file.datasets()
        
                # Place the data in the full set
                data = data_modis
                lat = lat_modis
                lon = lon_modis
    
            else:
                # Multiple hdf files means that the satellite data will be loaded piece-meal 
                # and need to be merged into a single global array
                
                # Determine the MODIS filename
                filename = 'modis.%s.%d-day.%04d.%02d.%02d.nc'%(args.variables[0], args.day_interval, timestamp.year, timestamp.month, timestamp.day)

                # Check if the file has already been downloaded (load it if so)
                if os.path.exists('./%s'%filename):
                    # Processed file does exist: next iteration
                    print("Data for %s has already been downloaded and merged to a global scale."%timestamp.strftime('%b %d, %Y'))
                    continue
    
                filenames = np.sort(filenames)
                print('%d files to collect'%len(filenames))
                
                # Download each tile separately
                for filename in filenames:
                    # New MODIS site attaches a lot of extra to the fname string, this gets rid of it
                    ind = ind = filename.find(base_fname) 

                    # Check if the file has already been downloaded
                    if os.path.exists('./raw/%s.dap.nc4'%filename[ind:]):
                        # Processed file does exist: skip the download step
                        print("%s is already downloaded."%filename[ind:])
                    else:
                        # Download and collect the data
                        response = session.get('%s/%s.dap.nc4'%(url_base, filename[ind:]))
                        open('%s/%s.dap.nc4'%('./raw', filename[ind:]), 'wb').write(response.content)
            
                    # Get the keys
                    #attributes = file.datasets()
            
                    # Select the data
                    if args.variables[0] == 'reflectance':
                        # Reflectance data load is slightly different since there are multiple bands
                        data_modis = {}
                        with Dataset('%s/%s.dap.nc4'%('./raw', filename[ind:]), 'r') as nc:
                            for sn in sname:
                                data_modis[sn] = nc.variables[sn][:]
                            lat_modis = nc.variables['Latitude'][:,0]
                            lon_modis = nc.variables['Longitude'][0,:]

                    else: 
                        # Load the data for the tile and convert it to a grid
                        data_modis, lat_modis, lon_modis, long_name, units = tile_to_grid(filename[ind:], 
                                                                                          args.var_snames_modis[0], 
                                                                                          path = './raw')
                    
                    I_modis, J_modis = data_modis.shape
                    
                    #print(lat_modis[:,0], lon_modis[0,:])
                    
                    #data_modis = data_modis.reshape(I_modis*J_modis)
                    #lat_modis = lat_modis.reshape(I_modis*J_modis)
                    #lon_modis = lon_modis.reshape(I_modis*J_modis)
                
                    ### New idea to interpolate data
                    print('Finding lat/lon subsets')

                    # Collect all lat and lon indices in the tile
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
                    #print(lat_sub)
                    #print(lon_sub)
                    
                    # New method with xarray
                    #ds = xarray.Dataset(data_vars = dict(modis_data = (['x', 'y'], data_modis)),
                    #                    coords = dict(x = (['x', 'y'], lon_modis), y = (['x', 'y'], lat_modis)))
                    
                    # For interpolate to work, both old and new lat and lon need to be non-repeating vectors.
                    # Look into resample, interp_like, and interpolating along lon only
                    
                    #print(np.nanmin(data_modis), np.nanmax(data_modis))
                    # print(np.nanmin(ds.modis_data.values), np.nanmax(ds.modis_data.values))
                    # print(ds)
                    
                    # Mesh the latitude and longitudes
                    I = lat_sub.size
                    J = lon_sub.size
                    lon_sub, lat_sub = np.meshgrid(lon_sub, lat_sub)
    
                    #lon_sub = lon_sub.reshape(I*J)
                    #lat_sub = lat_sub.reshape(I*J)
                    
                    #ds_interp = ds.interp(coords = dict(x = (['z'], lon_sub), y = (['z'], lat_sub)))
                    #ds_interp = ds.interp(coords = dict(x = lon_sub, y = lat_sub))
                    #print(np.nanmin(ds_interp.modis_data.values), np.nanmax(ds_interp.modis_data.values))
                    # Add interpolated values onto the global grid
                    #print(ds_interp)
                    
                    #ds_interp_values = ds_interp.modis_data.values
                    #ds_interp_values = np.reshape(I, J)
                    
                    # Interpolate to 0.05 degree x 0.05 degree grid
                    for n, ind in enumerate(lat_ind):
                        #print(n)
                        # Select all the MODIS latitudes between the current two latitude lines to interpolate to
                        la_ind = np.where( (lat_modis[:,0] >= lat_sub[n,0]) & (lat_modis[:,0] < (lat_sub[n,0] + resolution)) )[0]

                        lon_tmp = np.nanmean(lon_modis[la_ind,:], axis = 0)

                        # Perform a mean along the latitude line (reducing MODIS data to 0.05 degree along latitude)
                        if args.variables[0] == 'reflectance':
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
                            if args.variables[0] == 'reflectance':
                                for sn in sname:
                                    data_tmp[sn] = data_tmp[sn][res_ind[:]]
                            else:
                                data_tmp = data_tmp[res_ind[:]]
                        
                        # Interpolate along the longitude line
                        if args.variables[0] == 'reflectance':
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
                            #print(np.nanmin(ds_interp.modis_data.values), np.nanmax(ds_interp.modis_data.values))
                            #print(ds_interp)
                            
                            # Return the interpolated data to np.ndarray and add to the 0.05 deg x 0.05 grid
                            tmp = np.array([data[ind,lon_ind], ds_interp.modis_data.values[:]])
                            data[ind,lon_ind] = np.nanmean(tmp, axis = 0)
                        
                    del ds, ds_interp
                    gc.collect()

                    #### OLD Interpolation strategies  
                    # Recommendations: Break the process up (aggregate, then merge grid in parts, then merge into whole)
                    # Looks into os.path
                    # Possibly np.extract
                    #n = 1
                    #print('Beginning loop')
                    #for la in lat_sub:
                    #    lat_ind = np.where(la == lat)[0]
                    #    for lo in lon_sub:
                    #        # print('On %d of %d'%(n, lat_sub.size*lon_sub.size)) # Printing takes up a lot of time
                    #        lon_ind = np.where(lo == lon)[0]
                    #        ind = np.where((lat_modis >= la) & (lat_modis < (la + resolution)) & (lon_modis >= lo) & (lon_modis < (lo + resolution)))[0]
                                
                    #        # store these instead?
                    #        tmp = data[lat_ind,lon_ind] * np.ones(data[ind].shape)
                    #        if np.isnan(tmp).all():
                    #            n = n+1
                    #            continue
                    #            
                    #        data[lat_ind,lon_ind] = np.nanmean([data[ind].flatten(), tmp.flatten()])
                    #        n = n+1
                            
                	# Average all at once?
                    # data[lat_ind, lon_ind] = np.nanmean(data_list, axis = 0)
                
                    # data_full.append(data_modis.ravel())
                    # lat_full.append(lat_modis.ravel())
                    # lon_full.append(lon_modis.ravel())
            
                #print('Finished downloading, sorting data...')
            
                # Data global data to full set
                # data_full = np.array(data_full)
                # lon_full = np.array(lon_full)
                # lat_full = np.array(lat_full)
        
                # Sort the lon and lat along each tile
                #ind = np.argsort(lon_full, axis = -1) # Collect the indices for sorting lon

                #data_full = np.take_along_axis(data_full, ind, axis = -1) # Apply the sorting indices
                #lon_full = np.take_along_axis(lon_full, ind, axis = -1)
                #lat_full = np.take_along_axis(lat_full, ind, axis = -1)

                #ind = np.argsort(lat_full, axis = 0) # Collect the indices for sorting lat

                #data_full = np.take_along_axis(data_full, ind, axis = 0) # Apply the sorting indices
                #lon_full = np.take_along_axis(lon_full, ind, axis = 0)
                #lat_full = np.take_along_axis(lat_full, ind, axis = 0)
        
                #print('Calculating lower resolution scale...')
        
                # Create the lat and lon for 0.05 degree x 0.05 degree grid
                # resolution = 0.05
                # lat = np.arange(-90, 90, resolution)
                # lon = np.arange(-180, 180, resolution)
            
                # Initialize the grid
                # I = lat.size
                # J = lon.size
                # data = np.ones((I, J), dtype = np.float32)
                # for i in range(I):
                #     for j in range(J):
                #         ind = np.where( ((lat_full >= lat[i]) & (lat_full < (lat[i]+resolution))) & ((lon_full >= lon[j]) & (lon_full <= (lon[j]+resolution))) )[0]
                #         data[i,j] = np.nanmean(data_full[ind])
                    
                # Mesh the lat and lon into a grid
                lon, lat = np.meshgrid(lon, lat)
                
                print('Writing data...')
                   
            # Write the data 
            filename = 'modis.%s.%d-day.%04d.%02d.%02d.nc'%(args.variables[0], args.day_interval, timestamp.year, timestamp.month, timestamp.day)
            description = 'Global MODIS %s (%s) for an %d-day period starting at %s, and averaged down to 0.05 degree x 0.05 degree resolution.'%(long_name, units, args.day_interval, timestamp.strftime('%b %d, %Y'))
            write_nc(data, 
                     lat, 
                     lon, 
                     [timestamp], 
                     filename = filename, 
                     var_sname = sname, 
                     description = description)
            
            # Create a test map
            #test_map(data, lat, lon, timestamp, '%s'%args.variables[0])
            #plt.close('all')
        
            # Write the raw 500m data?
            #if args.save_raw:
                # filename = 'modis.%s.%d-day.raw.%02d.%02d.%04d.nc'%(args.variables[0], args.day_interval, timestamp.day, timestamp.month, timestamp.year)
                # if os.path.exists('./%s'%filename):
                    # Processed file does exist: next iteration
                #     print("Raw data for %s has already been saved."%timestamp.strftime('%b %d, %Y'))
                # else:
                #    description = 'Global MODIS %s (%s) for an %d-day period starting at %s.'%(long_name, units, args.day_interval, timestamp.strftime('%b %d, %Y'))
                #    write_nc(data_full, lat_full, lon_full, [timestamp], filename = filename, var_sname = sname, description = description)
            
                    # Create a test map
                #    test_map(data_full, lat_full, lon_full, timestamp, '%s_raw'%args.variables[0])        
            
            # Remove potentially large datasets to free up space
            del data, lat, lon#, data_full, lat_full, lon_full
            gc.collect()
        
            # Take a short rest
            time.sleep(10)
