"""
Script to download daily IMERG data from Earth Data HTTPs (1 year of data at the daily times scale is ~ 9.46 GB per variable)

NOTE: Earth Data requires authentication to access data. Earth Data login credentials in the .netrc file MUST be up to date for the script to run

Acceptable names for --variables and --var_snames_imerg (i.e., variable names in the downloaded GLDAS dataset):
    --variables:
    - Precipitation: precipitation
    - Random Error: random_error
    
    --var_snames_imerg:
    - Precipitation: precipitation
    - Random Error: randomError
    
Variable units in IMERG dataset:
    - Precipitation: mm/day
    - Random Error: mm/day
"""


import os, warnings
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
    
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='IMERG Downloader', fromfile_prefix_chars='@')

    parser.add_argument('--variables', type=str, nargs='+', default=['precipitation'], help='Variables to download')
    parser.add_argument('--var_snames_imerg', type=str, nargs='+', default=['precipitation'], help='The short name for --variables used in raw datafile')
    parser.add_argument('--start_year', type=int, default=2000, help='Beginning year to be downloaded.')
    parser.add_argument('--end_year', type=int, default=2023, help='Ending year to be downloaded.')
    
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
def load_nc(variables, filename, path = './'):
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
        lat = nc.variables['lat'][:]
        lon = nc.variables['lon'][:]

        X['lat'] = lat
        X['lon'] = lon

        # Collect the data itself
        # Assumes 3D data, in the shape of time by x by y
        for variable in variables:
            X[str(variable)] = nc.variables[str(variable)][0,:,:]
        
    return X


# Function to write netcdf files  
def write_nc(data, lat, lon, dates, mask = None, filename = 'tmp.nc', var_sname = 'tmp', description = 'Description', path = './'):
    '''
    Write data, and additional information such as latitude and longitude and timestamps, to a .nc file.
    
    Inputs:
    :param data: The variable being written (time x lat x lon format).
    :param lat: The latitude data with the same spatial grid as var.
    :param lon: The longitude data with the same spatial grid as var.
    :param dates: The timestamp for each pentad in data in a %Y-%m-%d format, same time grid as data.
    :param mask: Land-sea mask to be added to the dataset.
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
    T, J, I = data.shape
    T = len(dates)
    
    with Dataset(path + filename, 'w', format = 'NETCDF4') as nc:
        # Write a description for the .nc file
        nc.description = description

        
        # Create the spatial and temporal dimensions
        nc.createDimension('lat', size = I)
        nc.createDimension('lon', size = J)
        nc.createDimension('time', size = T)
        
        # Create the lat and lon variables       
        nc.createVariable('lat', lat.dtype, ('lat', ))
        nc.createVariable('lon', lon.dtype, ('lon', ))
        
        nc.variables['lat'][:] = lat[:]
        nc.variables['lon'][:] = lon[:]
        
        # Create the date variable
        nc.createVariable('date', str, ('time', ))
        for n in range(len(dates)):
            nc.variables['date'][n] = str(dates[n])
            
        # Create the main variable
        nc.createVariable(var_sname, data.dtype, ('time', 'lon', 'lat'))
        nc.variables[str(var_sname)][:,:,:] = data[:,:,:]
        
        if np.invert(mask is None):
             nc.createVariable('landmask', mask.dtype, ('lon', 'lat'))
             nc.variables['landmask'][:,:] = mask[:,:]
        
if __name__ == '__main__':
    # Parse and check incoming arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # Turn off warnings
    warnings.simplefilter('ignore')

    print('Initializing variables...')

    # Construct the base url and file names
    url_base = 'https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGDF.07/'
    fn_base = '3B-DAY.MS.MRG.3IMERG.%04d%02d%02d-S000000-E235959.V07B.nc4'

    # Load the Land-Sea mask
    with Dataset('GPM_IMERG_LandSeaMask.2.nc4', 'r') as nc:
        mask = nc.variables['landseamask'][:,:]

    # Create a list of timestamps
    start_time = datetime(args.start_year, 1, 1)
    end_time   = datetime(args.end_year, 12, 31)
    timestamps = date_range(start_time, end_time)
    
    timestamps = np.array([timestamp for timestamp in timestamps])
    
    # Initial a dataset 
    t = 0
    data_year = {}
    for variable in args.variables:
        data_year[variable] = []

    # Begin iterating for each timestamps
    print('Downloading IMERG data for each day...')
    n = 0
    while n < timestamps.size:
    # for n, timestamp in tqdm(enumerate(timestamps)):  
        # Construct the URL
        print('On day %s'%timestamps[n].strftime('%b %d, %Y'))
        
        url = '%s/%04d/%02d/'%(url_base, timestamps[n].year, timestamps[n].month)
        fn = fn_base%(timestamps[n].year, timestamps[n].month, timestamps[n].day)  
                
        if os.path.exists('./raw/%s'%fn):
            # Processed file does exist: exit
            print("%s is already downloaded."%fn)
        else:
            # Download the data
            session = requests.Session()
            response = session.get('%s/%s'%(url, fn))
            time.sleep(1) # Take a breather
            
            # Write the collected data to a file
            open('./raw/%s'%fn, 'wb').write(response.content)
    
        # Load in the data
        try:
            data = load_nc(args.var_snames_imerg, fn, path = './raw/')
        except OSError: # Most likely error when loading the data was a bad download (file is truncated/became corrupted; only a few KB are downloaded in this case)
            if (os.path.getsize('./raw/%s'%fn) < 2e7): # Check if the file is below 20 MB
                print('Bad download occurred - removing it and trying again')
                
                os.remove('./raw/%s'%fn)
                continue
            else:
                raise Exception('OSError: A download error occurred, please remove %s and try downloading again'%fn)
    
        # Average all data points in the day for a daily mean
        for variable, sname in zip(args.variables, args.var_snames_imerg):
            data[sname] = np.where(data[sname] == -9999.0, np.nan, data[sname]) # Replace missing values with NaN
    
            data_year[variable].append(data[sname])
    
        # Write data for one year
        try:
            end_of_year = np.invert(timestamps[n].year == timestamps[n+1].year)
        # One way to deal with the end of the dataset
        except IndexError:
            end_of_year = (timestamps[n] == timestamps[-1])
        
        if end_of_year:
            desc_base = 'Daily %s (%s) data for the Integrated Multi-satellitE Retrievals for the GPM (IMERG) dataset for %d.'
            desc_end  = 'Timestamps are string, IMERG land-sea mask is included, and data is time x lon x lat format. \n' +\
                        'title: GPM IMERG Final Precipitation L3 1 day 0.1 degree x 0.1 degree (GPM_3IMERGDF) \n' +\
                        'citation doi: https://doi.org/10.5067/GPM/IMERGDF/DAY/07 \n'
                        
            for variable in args.variables:
                data_year[variable] = np.array(data_year[variable])
                
                # Fill in missing values
                data_year[variable] = np.where(np.isnan(data_year[variable]), -9999.0, data_year[variable])
                
                # Shrink the datasize to be more manageable
                data_year[variable] = data_year[variable].astype(np.float32)
                
                # Determine the variable name and description in the nc file based on the variable being saved
                if variable == 'precipitation':
                    var_sname = 'precip'
                    desc = desc_base%(variable, 'mm/day', timestamps[n].year)
                    
                elif variable == 'random_error':
                    var_sname = 'rand_err'
                    desc = desc_base%(variable, 'mm/day', timestamps[n].year)
                    
                # Finish preparing the file description and filename
                desc = desc + desc_end
                filename = 'imerg.%s.daily.%d.nc'%(variable, timestamps[n].year)
                
                # Write the data
                write_nc(data_year[variable], data['lat'], data['lon'], timestamps[t:n+1], mask,
                         filename = filename, var_sname = var_sname, description = desc)
                     
            # Re-initialize the year of data
            t = n + 1
            data_year = {}
            for variable in args.variables:
                data_year[variable] = []
        
        # Iterate the index    
        n = n + 1
        
        
# List of nc shortnames in the raw IMERG files:
# 'time'
# 'lon'
# 'lat'
# 'time_bnds'
# 'precipitation'
# 'precipitation_cnt'
# 'precipitation_cnt_cond'
# 'MWprecipitation'
# 'MWprecipitation_cnt'
# 'MWprecipitation_cnt_cond'
# 'randomError'
# 'randomError_cnt'
# 'probabilityLiquidPrecipitation'

