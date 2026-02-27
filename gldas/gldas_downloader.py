"""
Script to download GLDAS data from Earth Data HTTPs and reduce it to 
a more managable daily timescale (1 year of data at the daily time 
scale is ~ 1.26 GB per variable)

NOTE: Earth Data requires authentication to access data. Earth Data 
login credentials in the .netrc file MUST be up to date for the script to run

Acceptable names for --variables and --var_snames_gldas 
(i.e., variable names in the downloaded GLDAS dataset):
    --variables:
    - Temperature: temperature
    - Precipitation: precipitation
    - Evaporation: evaporation
    - Potential Evaporation: potential_evaporation
    - Wind Speed: wind_speed
    - Net Radiation: net_radiation
    - Pressure: pressure
    - Soil Moisture (0-  10 cm): soil_moisture_0-10cm
    - Soil Moisture (10 - 40 cm): soil_moisture_10-40cm
    - Soil Moisture (40 - 100 cm): soil_moisture_40-100cm
    - Soil Moisture (100 - 200 cm): soil_moisture_100-200cm
    
    --var_snames_gldas:
    - Temperature: Tair_f_inst
    - Precipitation: Rainf_f_tavg
    - Evaporation: Evap_tavg
    - Potential Evaporation: PotEvap_tavg
    - Wind Speed: Wind_f_inst
    - Net Radiation: Swnet_tavg
    - Pressure: Psurf_f_inst
    - Soil Moisture (0 - 10 cm): SoilMoi0_10cm_inst
    - Soil Moisture (10 - 40 cm): SoilMoi10_40cm_inst
    - Soil Moisture (40 - 100 cm): SoilMoi40_100cm_inst
    - Soil Moisture (100 - 200 cm): SoilMoi100_200cm_inst
    
Variable units in GLDAS dataset:
    - Temperature: K
    - Precipitation: kg m^-2 s^-1
    - Evaporation: kg m^-2 s^-1
    - Potential Evaporation: W m^-2
    - Wind Speed: m s^-1
    - Net Radiation: W m^-2
    - Pressure: Pa
    - Soil Moisture: kg m^-2
"""

import os, warnings
import numpy as np
import argparse
import requests
import time
from typing import Dict
from tqdm import tqdm
from netCDF4 import Dataset
from datetime import datetime, timedelta

# Function to create a parser using the terminal
def create_parser():
    '''
    Create argument parser
    '''
 
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='GLDAS Downloader', fromfile_prefix_chars='@')

    parser.add_argument('--variables', type=str, nargs='+', default=['temperature'], help='Variables to download')
    parser.add_argument('--var_snames_gldas', type=str, nargs='+', default=['Tair_f_inst'], help='The short name for --variables used in raw datafile')
    parser.add_argument('--start_year', type=int, default=2000, help='Beginning year dat to be downloaded.')
    parser.add_argument('--end_year', type=int, default=2023, help='Ending year to be downloaded.')
    
    return parser
    
# Create a function to generate a range of datetimes
def date_range(start_date, end_date):
    '''
    This function takes in two dates and outputs all the dates inbetween
    those two dates
    
    Inputs:
    :param StartDate: Starting date of the interval (must be a datetime)
    :param EndDate: Ending date of the interval (must be a datetime)
        
    Outputs:
    - A generator of all dates between start_date and end_date (inclusive)
    '''
    for n in range(int((end_date - start_date).days) + 1):
        yield start_date + timedelta(n) 
    
# Create a function to load .nc files
def load_nc(
        variables, 
        filename, 
        path = './'
        ) -> Dict[str, np.ndarray]:
    '''
    Load a .nc file
    
    Inputs:
    :param variables: Short name of the variables being loaded. I.e., the name used
                      to call the variable in the .nc file
    :param filename: Name of the .nc file
    :param path: Path to the directory the .nc file is in
    
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
    Write data, and additional information such as latitude and longitude and timestamps, to a .nc file
    
    Inputs:
    :param data: The variable being written (np.ndarray with shape time x lat x lon)
    :param lat: Latitude labels (np.ndarray with shape lat)
    :param lon: Longitude labels (np.ndarray with shape lon)
    :param dates: Timestamps for each day in data in a %Y-%m-%d format (np.ndarray with shape time)
    :param mask: Land-sea mask to be added to the dataset (np.ndarray with shape lat x lon)
    :param filename: The filename of the .nc file being written
    :param var_sname: The short name of the variable being written. I.e., the name used
                      to call the variable in the .nc file.
    :param description: A description string for the data
    :param path: The path to the directory the data will be written in
    '''
    
    # Determine the spatial and temporal lengths
    T, I, J = data.shape
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
        nc.createVariable(var_sname, data.dtype, ('time', 'lat', 'lon'))
        nc.variables[str(var_sname)][:,:,:] = data[:,:,:]
        
        # Create the mask if needed
        if np.invert(mask is None):
             nc.createVariable('landmask', mask.dtype, ('lat', 'lon'))
             nc.variables['landmask'][:,:] = mask[:,:]


if __name__ == '__main__':
    # Parse and check incoming arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # Turn off warnings
    warnings.simplefilter('ignore')

    print("Initializing variables...")

    # Construct the base of the HTTPs GLDAS is located in, and base structure of GLDAS filenames
    url_base = 'https://hydro1.gesdisc.eosdis.nasa.gov/data/GLDAS/GLDAS_NOAH025_3H.2.1/'
    fn_base = 'GLDAS_NOAH025_3H.A%04d%02d%02d.%s.021.nc4'

    # All hours in a given day of GLDAS data
    hours = ['0000', '0300', '0600', '0900', '1200', '1500', '1800', '2100']

    # Load the Land-Sea mask
    # with Dataset('GPM_IMERG_LandSeaMask.2.nc4', 'r') as nc:
    #     mask = nc.variables['landseamask'][:,:]

    # Create a list of timestamps
    start_time = datetime(args.start_year+2000, 1, 1)
    end_time   = datetime(args.end_year+2000, 12, 31)
    timestamps = date_range(start_time, end_time)
    
    timestamps = np.array([timestamp for timestamp in timestamps])
    
    # Initialize a dataset 
    t = 0
    data_year = {}
    for variable in args.variables:
        data_year[variable] = []

    # Begin iterating for each timestamps
    print('Downloading GLDAS data for each date...')
    n = 0
    while n < timestamps.size:
    # for n, timestamp in tqdm(enumerate(timestamps)):
        # Initialize the data for a day
        print('On day %s'%timestamps[n].strftime('%b %d, %Y'))
        data_day = {}
        for sname in args.var_snames_gldas:
            data_day[sname] = []
            
        # Download the data for each 3 hour increment
        for m, hour in enumerate(hours):
            # Construct the URL
            day = (timestamps[n] - datetime(timestamps[n].year, 1, 1)).days + 1
            url = '%s/%04d/%03d/'%(url_base, timestamps[n].year, day)
            fn = fn_base%(timestamps[n].year, timestamps[n].month, timestamps[n].day, hour)
            
            if os.path.exists('./raw/%04d/%s'%(timestamps[n].year, fn)):
                # Processed file exists: exit
                print("%s is already downloaded."%fn)
            else:
                # Download the data
                session = requests.Session()
                response = session.get('%s/%s'%(url, fn))
                time.sleep(1) # Take a breather
            
                # Write the collected data to a file
                open('./raw/%04d/%s'%(timestamps[n].year, fn), 'wb').write(response.content)
    
            # Load in the data
            try:
                data = load_nc(args.var_snames_gldas, fn, path = './raw/%04d/'%timestamps[n].year)
            except OSError: # Most likely error when loading the data was a bad download 
                            # (file is truncated/became corrupted; only a few KB are downloaded in this case)
                if (os.path.getsize('./raw/%04d/%s'%(timestamps[n].year, fn)) < 1e7): # Check if the file is below 10 MB
                    print('Bad download occurred - removing it and trying again')
                    
                    os.remove('./raw/%04d/%s'%(timestamps[n].year, fn))
                    continue
                else:
                    raise Exception('OSError: A download error occurred, please remove %s and try downloading again'%fn)
        
            # Append for all variables
            for sname in args.var_snames_gldas:
                data_day[sname].append(data[sname])
    
        # Average all data points in the day for a daily mean
        for variable, sname in zip(args.variables, args.var_snames_gldas):
            data_day[sname] = np.array(data_day[sname])
            data_day[sname] = np.where(data_day[sname] == -9999.0, np.nan, data_day[sname]) # Replace missing values with NaN
            
            data_day[sname] = np.nanmean(data_day[sname], axis = 0)
    
            data_year[variable].append(data_day[sname])
    
        # Determine if the end of the year is reached
        try:
            end_of_year = np.invert(timestamps[n].year == timestamps[n+1].year)
        # One way to deal with the end of the dataset
        except IndexError:
            end_of_year = (timestamps[n] == timestamps[-1])
        
        # For end of the year, write the .nc file
        if end_of_year:
            # Make the file description
            desc_base = 'Daily average %s (%s) data for the Global Land Data Assimilation Dataset (GLDAS) Version 2 dataset for %d.'
            desc_end  = 'Timestamps are string and data is time x lat x lon format. \n' +\
                        'source: Noah_v3.6 forced with GDAS-AGRMET-GPCPv13rA1 \n' +\
                        'institution: NASA GSFC \n' +\
                        'missing_value: -9999.0 \n' +\
                        'title: GLDAS2.1 LIS land surface model output \n' +\
                        'references: Rodell_etal_BAMS_2004, Kumar_etal_EMS_2006, Peters-Lidard_etal_ISSE_2007 \n' +\
                        'citation doi: https://disc.gsfc.nasa.gov/datacollection/GLDAS_NOAH025_3H_2.1.html, http://dx.doi.org/10.1175/BAMS-85-3-381 \n' +\
                        'comment: website: https://ldas.gsfc.nasa.gov/gldas, https://lis.gsfc.nasa.gov/ \n' +\
                        'MAP_PROJECTION: EQUIDISTANT CYLINDRICAL \n' +\
                        'SOUTH_WEST_CORNER_LAT: -59.875 \n' +\
                        'SOUTH_WEST_CORNER_LON: -179.875 \n' +\
                        'DX: 0.25 \n' +\
                        'DY: 0.25 \n' +\
                        'CDO: Climate Data Operators version 1.9.8 (https://mpimet.mpg.de/cdo) \n'
            
            # Prepare each variable
            for variable in args.variables:
                data_year[variable] = np.array(data_year[variable])
                
                # Fill in missing values
                data_year[variable] = np.where(np.isnan(data_year[variable]), -9999.0, data_year[variable])
                
                # Shrink the datasize to be more manageable
                data_year[variable] = data_year[variable].astype(np.float32)
                
                # Determine the variable name and description in the nc file based on the variable being saved
                if variable == 'temperature':
                    var_sname = 'temp'
                    desc = desc_base%(variable, 'K', timestamps[n].year)
                    
                elif variable == 'evaporation':
                    var_sname = 'evap'
                    desc = desc_base%(variable, 'kg m-2 s-1', timestamps[n].year)
                    
                elif variable == 'potential_evaporation':
                    var_sname = 'pevap'
                    desc = desc_base%(variable, 'W m-2', timestamps[n].year)
                    
                elif variable == 'soil_moisture_0-10cm':
                    var_sname = 'soilm'
                    desc = desc_base%(variable, 'kg m-2', timestamps[n].year)
                    
                elif variable == 'soil_moisture_10-40cm':
                    var_sname = 'soilm'
                    desc = desc_base%(variable, 'kg m-2', timestamps[n].year)
                    
                elif variable == 'soil_moisture_40-100cm':
                    var_sname = 'soilm'
                    desc = desc_base%(variable, 'kg m-2', timestamps[n].year)
                    
                elif variable == 'soil_moisture_100-200cm':
                    var_sname = 'soilm'
                    desc = desc_base%(variable, 'kg m-2', timestamps[n].year)
                    
                elif variable == 'precipitation':
                    var_sname = 'precip'
                    desc = desc_base%(variable, 'kg m-2 s-1', timestamps[n].year)
                    
                elif variable == 'wind_speed':
                    var_sname = 'wspd'
                    desc = desc_base%(variable, 'm s-1', timestamps[n].year)
                    
                elif variable == 'net_radiation':
                    var_sname = 'rnet'
                    desc = desc_base%(variable, 'W m-2', timestamps[n].year)
                elif variable == 'pressure':
                    var_sname = 'pres'
                    desc = desc_base%(variable, 'Pa', timestamps[n].year)
                    
                # Finish preparing the file description and filename
                desc = desc + desc_end
                filename = 'gldas.%s.daily.%d.nc'%(variable, timestamps[n].year)
                
                # Write the data
                write_nc(data_year[variable], data['lat'], data['lon'], timestamps[t:n+1], 
                         filename = filename, var_sname = var_sname, description = desc)
                         
            # Re-initialize the year of data
            t = n + 1
            data_year = {}
            for variable in args.variables:
                data_year[variable] = []
                
        # Iterate the index
        n = n + 1

# List of nc shortnames in the raw GLDAS files:
# 'time'
# 'lon'
# 'lat'
# 'Swnet_tavg'
# 'Lwnet_tavg'
# 'Qle_tavg'
# 'Qh_tavg'
# 'Qg_tavg'
# 'Snowf_tavg'
# 'Rainf_tavg'
# 'Evap_tavg'
# 'Qs_acc'
# 'Qsb_acc'
# 'Qsm_acc'
# 'AvgSurfT_inst'
# 'Albedo_inst'
# 'SWE_inst'
# 'SnowDepth_inst'
# 'SoilMoi0_10cm_inst'
# 'SoilMoi10_40cm_inst'
# 'SoilMoi40_100cm_inst'
# 'SoilMoi100_200cm_inst'
# 'SoilTMP0_10cm_inst'
# 'SoilTMP10_40cm_inst'
# 'SoilTMP40_100cm_inst'
# 'SoilTMP100_200cm_inst'
# 'PotEvap_tavg'
# 'ECanop_tavg'
# 'Tveg_tavg'
# 'ESoil_tavg'
# 'RootMoist_inst'
# 'CanopInt_inst'
# 'Wind_f_inst'
# 'Rainf_f_tavg'
# 'Psurf_f_inst'
# 'Tair_f_inst'
# 'Qair_f_inst'
# 'SWdown_f_tavg'
# 'LWdown_f_tavg'
