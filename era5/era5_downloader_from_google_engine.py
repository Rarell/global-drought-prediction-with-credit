"""
Script to download raw ERA5 data from the Google Earth engine and reduce it to a more managable daily timescale (1 year of data at the daily times scale is ~ 3 GB per variable)

Acceptable names for --variable and --var_sname_era (i.e., variable names in the downloaded ERA5 dataset):
    --variable:
    
    (Variables for single levels dataset)
    - Temperature: 2m_temperature
    - Precipitation: total_precipitation
    - Dewpoint: 2m_dewpoint_temperature
    - Pressure: surface_pressure
    - Net Radiation: surface_net_solar_radiation
    - Evaporation: evaporation
    - Potential Evaporation: potential_evaporation
    - Runoff (total): runoff
    - Soil Moisture (0 - 7 cm): volumetric_soil_water_layer_1
    - Soil Moisture (7 - 28 cm): volumetric_soil_water_layer_2
    - Soil Moisture (28 - 100 cm): volumetric_soil_water_layer_3
    - Soil Moisture (100 - 289 cm): volumetric_soil_water_layer_4
    - Type of High Vegetation: type_of_high_vegetation
      (3 = Evergreen needleleaf trees, 4 = Deciduous needleleaf trees, 5 = Deciduous broadleaf trees, 6 = Evergreen broadleaf trees, 18 = Mixed forest/woodland, 19 = Interrupted forest)
    - Type of Low Vegetation: type_of_low_vegetation
      (1 = Crops/Mixed farming, 2 = grass, 7 = Tall grass, 9 = Tundra, 10 = Irrigated crops, 11 = Semidesert, 13 = Bogs and marshes, 16 = Evergreen shrubs, 17 = Deciduous shrubs, 20 = water and land mixtures)
    - Percent Coverage of High Vegetation: high_vegetation_cover 
    - Percent Coverage of Low Vegetation: low_vegetation_cover
    - Total Column Rain Water: total_column_rain_water
    - Total Column Snow Water: total_column_snow_water
      
    (Variables for daily statistics on pressure levels dataset)
    - Geopotential: geopotential
      (Geopotential height, Phi, can be obtained from geopotential divided by 9.80665 m s^-2)
    - u-component of Wind: u_component_of_wind
    - v-component of Wind: v_component_of_wind
    - Specific Humidity: specific_humidity 
    - Specific Cloud Liquid Water Content: specific_cloud_liquid_water_content
    - Specific Cloud Ice Water Content: specific_cloud_ice_water_content
    
    --var_sname_era:
    - Temperature: t2m
    - Precipitation: tp
    - Dewpoint: d2m
    - Pressure: sp
    - Net Radiation: ssr
    - Evaporation: e
    - Potential Evaporation: pev
    - Runoff (total): ro
    - Soil Moisture (0 - 7 cm): swvl1
    - Soil Moisture (7 - 28 cm): swvl2
    - Soil Moisture (28 - 100 cm): swvl3
    - Soil Moisture (100 - 289 cm): swvl4
    - Type of High Vegetation: tvh
    - Type of Low Vegetation: tvl
    - Percent Coverage of High Vegetation: cvh
    - Percent Coverage of Low Vegetation: cvl
    - Total Column Rain Water: tcrw
    - Total Column Snow Water: tcsw
    
    - Geopotential: z
    - u-component of Wind: u
    - v-component of Wind: v
    - Specific Humidity: q
    - Specific Cloud Liquid Water Content: clwc
    - Specific Cloud Ice Water Content: ciwc
    
Variable units in ERA5 dataset:
    - Temperature: K
    - Precipitation: m (accumulation over 6 hours)
    - Dewpoint: K
    - Pressure: Pa
    - Net Radiation: J m^-2 (over 6 hours)
    - Evaporation: m (accumulation over 6 hours)
    - Potential Evaporation: m (accumulation over 6 hours)
    - Runoff: m
    - Soil Moisture: Unitless (m^3 m^-3)
    - Total Column Rain Water: kg m^-2
    - Total Column Snow Water: kg m^-2
    - Vegetation Variables: Dimensionless
    
    - Geopotential: m^2 s^-2
    - Wind components: m s^-1
    - Specific Humidity: kg kg^-1
    - Specific Cloud Liquid Water Content: kg kg^-1
    - Specific Cloud Ice Water Content: kg kg^-1
"""

import os, warnings
import re
import numpy as np
import argparse
import requests
import time
from tqdm import tqdm
from netCDF4 import Dataset
from datetime import datetime, timedelta

# Function to create a parser using the terminal
def create_parser():
    '''
    Create argument parser
    '''
    
    # To add: args.time_series, args.climatology_plot, args.case_studies, args.case_study_years
              
    
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='ERA Downloader', fromfile_prefix_chars='@')

    parser.add_argument('--variables', type=str, nargs='+', default=['2m_temperature'], help='Variable to download (Note only 1 variable should be downloaded at a time)')
    parser.add_argument('--var_snames_era5', type=str, nargs='+', default=['t2m'], help='The short name for --variable used in raw datafile')
    parser.add_argument('--test', action='store_true', help='Perform a test download (only retrieves 1 year of data)?')
    parser.add_argument('--years', type=int, nargs=2, default=[1979,2021], help='Beginning and ending years to download data for.')
    parser.add_argument('--era5_dataset', type=str, default='date-variable-single_level', help='ERA5 dataset to collect from (e.g., date-variable-single_level or date-variable-pressure_level)')
    parser.add_argument('--pressure_level', type=int, default=500, help='Pressure level (in hPa) of the data to be downloaded (only used for datasets that use pressure levels)')
    parser.add_argument('--process', action='store_true', help='Process the data (downscales data to daily scale, which reduces the size by 1/6th)')
    
    return parser

# Create a function to generate a range of datetimes
def date_range(StartDate, EndDate):
    '''
    This function takes in two dates and outputs all the dates inbetween
    those two dates.
    
    Inputs:
    :param StartDate: A datetime. The starting date of the interval.
    :param EndDate: A datetime. The ending date of the interval.
        
    Outputs:
    - A generator of all dates between StartDate and EndDate (inclusive)
    '''
    for n in range(int((EndDate - StartDate).days) + 1):
        yield StartDate + timedelta(n) 
    
# Create a function to load .nc files
def load_nc(variable, filename, path = './'):
    '''
    Load a .nc file.
    
    Inputs:
    :param variables: List of short names of the variables being loaded. I.e., the name used
                  to call the variable in the .nc file.
    :param filename: The name of the .nc file.
    :param path: The path from the current directory to the directory the .nc file is in.
    
    Outputs:
    :param X: A dictionary containing the data loaded from the .nc file. The 
              entry 'lat' contains latitude (space dimensions), 'lon' contains longitude
              (space dimensions), and 'variable' contains a variable (lat x lon).
    '''
    
    # Initialize the directory to contain the data
    X = {}
    DateFormat = '%Y-%m-%d %H:%M:%S'
    
    with Dataset(path + filename, 'r') as nc:
        # Load the grid
        lat = nc.variables['latitude'][:]
        lon = nc.variables['longitude'][:]

        X['lat'] = lat
        X['lon'] = lon

        # Collect the data itself
        # Assumes 3D data, in the shape of time by x by y
        X[str(variable)] = nc.variables[str(variable)][:,:,:]
            
        # Load the file description
        desc = ''
        for attr in nc.ncattrs():
            desc = desc + attr + ': ' + getattr(nc, attr) + '\n'
        X['desc'] = desc
        
    return X

# Function to write netcdf files  
def write_nc(var, lat, lon, dates, filename = 'tmp.nc', var_sname = 'tmp', description = 'Description', path = './'):
    '''
    Write data, and additional information such as latitude and longitude and timestamps, to a .nc file.
    
    Inputs:
    :param var: The variable being written (time x lat x lon format).
    :param lat: The latitude data with the same spatial grid as var.
    :param lon: The longitude data with the same spatial grid as var.
    :param dates: The timestamp for each pentad in var in a %Y-%m-%d format, same time grid as var.
    :param filename: The filename of the .nc file being written.
    :param sm: A boolean value to determine if soil moisture is being written. If true, an additional variable containing
               the soil depth information is provided.
    :param VarName: The full name of the variable being written (for the nc description).
    :param VarSName: The short name of the variable being written. I.e., the name used
                     to call the variable in the .nc file.
    :param description: A string descriping the data.
    :param path: The path to the directory the data will be written in.

    '''
    
    # Determine the spatial and temporal lengths
    T, I, J = var.shape
    T = len(dates)
    
    # Make the lat and lon 2D
    lon, lat = np.meshgrid(lon, lat)
    
    with Dataset(path + filename, 'w', format = 'NETCDF4') as nc:
        # Write a description for the .nc file
        nc.description = description

        
        # Create the spatial and temporal dimensions
        nc.createDimension('x', size = I)
        nc.createDimension('y', size = J)
        nc.createDimension('time', size = T)
        
        # Create the lat and lon variables       
        nc.createVariable('lat', lat.dtype, ('x', 'y'))
        nc.createVariable('lon', lon.dtype, ('x', 'y'))
        
        nc.variables['lat'][:,:] = lat[:,:]
        nc.variables['lon'][:,:] = lon[:,:]
        
        # Create the date variable
        nc.createVariable('date', str, ('time', ))
        for n in range(len(dates)):
            nc.variables['date'][n] = str(dates[n])
            
        # Create the main variable
        nc.createVariable(var_sname, var.dtype, ('time', 'x', 'y'))
        nc.variables[str(var_sname)][:,:,:] = var[:,:,:]
    

if __name__ == '__main__':
    # Parse and check incoming arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # Turn off warnings
    warnings.simplefilter('ignore')

    print("Initializing variables...")
    
    # Construct the base of the HTTPs ERA5 is located in, and base structure of ERA5 filenames
    if args.era5_dataset == 'date-variable-single_level':
        level = 'surface'
    else:
        level = str(args.pressure_level)
    url_base = 'https://storage.googleapis.com/gcp-public-data-arco-era5/raw'
    fn_base = 'era5_%s_%04d%02d%02d_%s.nc'
    
    # Create a list of timestamps
    ### Note the +2000 is so the start and end times works with SLURM task IDs, otherwise normal year values can be used
    start_time = datetime(args.years[0]+2000, 1, 1)
    end_time   = datetime(args.years[1]+2000, 12, 31)
    timestamps = date_range(start_time, end_time)
    
    timestamps = np.array([timestamp for timestamp in timestamps])
    
    # Initialize a dataset 
    data_year = {}
    for variable in args.variables:
        data_year[variable] = []

    # Begin iterating for each timestamps
    print('Downloading ERA5 data for each date...')
    for v, variable in enumerate(args.variables):
        t = 0
        n = 0
        if args.era5_dataset == 'date-variable-single_level':
            filename = '%s_%d.nc'%(variable, timestamps[n].year)
        elif args.era5_dataset == 'date-variable-pressure_level':
            filename = '%s_%d_%dmb.nc'%(variable, timestamps[n].year, args.pressure_level)
            
        # If the processed file already exists, skip this step
        if os.path.exists('./%s'%(filename)):
            # Processed file does exist: exit
            print("%s has already been downloaded and processed."%filename)
            continue
            
        while n < timestamps.size:
            # Initialize the data for a day
            print('On day %s'%timestamps[n].strftime('%b %d, %Y'))
            sname = args.var_snames_era5[v]
            
            # Download the data for each day
            # Construct the URL
            url = '%s/%s/%04d/%02d/%02d/%s'%(url_base, 
                                              args.era5_dataset, 
                                              timestamps[n].year,
                                              timestamps[n].month,
                                              timestamps[n].day,
                                              variable)
            fn = fn_base%(variable, 
                          timestamps[n].year,
                          timestamps[n].month,
                          timestamps[n].day,
                          level)
            if args.era5_dataset == 'date-variable-single_level':
                url_end = 'surface.nc'
            elif args.era5_dataset == 'date-variable-pressure_level':
                url_end = '%d.nc'%args.pressure_level
            
            if os.path.exists('./%s'%(fn)):
                # Processed file does exist: exit
                print("%s is already downloaded."%fn)
            else:
                # Download the data
                session = requests.Session()
                response = session.get('%s/%s'%(url, url_end))
                time.sleep(1) # Take a breather
            
                # Write the collected data to a file
                open('./%s'%(fn), 'wb').write(response.content)
    
            # Load in the data
            try:
                data = load_nc(sname, fn, path = './')
            except OSError: # Most likely error when loading the data was a bad download (file is truncated/became corrupted; only a few KB are downloaded in this case)
                if (os.path.getsize('./%s'%(fn)) < 1e5): # Check if the file is below 0.1 MB
                    print('Bad download occurred - removing it and trying again')
                    
                    os.remove('./%s'%(fn))
                    continue
                else:
                    raise Exception('OSError: A download error occurred, please remove %s and try downloading again'%fn)
        
            # Process the data?
            if args.process:
                 # Average down to the daily time scale
                 if (variable == 'total_precipitation') | (variable == 'total_column_rain_water') | (variable == 'total_column_snow_water') | (variable == 'evaporation') | (variable == 'potential_evaporation') | (variable == 'surface_net_solar_radiation'):
                     data_year[variable].append(np.nansum(data[sname], axis = 0)) # These variables are accumulation/sums
                 else:
                    data_year[variable].append(np.nanmean(data[sname], axis = 0))
    
    
                 # Write data for one year
                 try:
                    end_of_year = np.invert(timestamps[n].year == timestamps[n+1].year)
                 # One way to deal with the end of the dataset
                 except IndexError:
                    end_of_year = (timestamps[n] == timestamps[-1])
        
                 if end_of_year:
                    desc_base = "Daily ERA5 reanalysis data for %s. Metadata from Google Engine: \n"%variable

                    # print(data_year[variable].shape, data_year[variable])
                    data_year[variable] = np.array(data_year[variable])
                    
                    # ET and PET in ERA5 are negative; make them positive
                    # if (variable == 'evaporation') | (variable == 'potential_evaporation'):
                    #     print('Making evaporation/potential evaporation positive...')
                    #     data_year[variable] = -1 * data_year[variable]
                
                    # Shrink the datasize to be more manageable
                    data_year[variable] = data_year[variable].astype(np.float32)
                    
                    # Finish preparing the file description and filename
                    desc = desc_base + data['desc']
                
                    # Write the data
                    write_nc(data_year[variable], data['lat'], data['lon'], timestamps[t:n+1], 
                             filename = filename, var_sname = sname, description = desc)
                         
                    # Re-initialize the year of data
                    t = n + 1
                    data_year = {}
                    data_year[variable] = []
                        
                    # Delete the extra large file
                    print('Removing large datafiles...')
                    fname_base = 'era5_%s_%04d'%(variable, timestamps[n].year)
                    files = ['./%s'%(f) for f in os.listdir('./') if re.match(r'%s.+%s.nc'%(fname_base, level), f)]
                    for file in files:
                        os.remove(file)
                
            # Iterate the index
            n = n + 1

