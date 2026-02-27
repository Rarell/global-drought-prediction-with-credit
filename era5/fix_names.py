''' Fix variable short names in .nc files

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
'''

import os, warnings
import re
import numpy as np
from typing import Dict
from netCDF4 import Dataset
from datetime import datetime, timedelta

def load_nc(
        filename, 
        var
        ) -> Dict[str, np.ndarray]:
    '''
    Load a nc file

    Inputs:
    :param filename: Path + filename of the nc file
    :param var: Short name of the variables being loaded. I.e., the name used
                to call the variable in the .nc file.
    
    Outputs:
    :param X: A dictionary containing the data loaded from the .nc file. The 
              entry 'lat' contains latitude (np.ndarray with shape lat x lon), 
              'lon' contains longitude (np.ndarray with shape lat x lon), and 
              'variable' contains the variable data (np.ndarray with shape time x lat x lon).
    '''

    # Initialize the dictionary
    x = {}

    # Load the data
    with Dataset(filename, 'r') as nc:
        x['lat'] = nc.variables['lat'][:,:]
        x['lon'] = nc.variables['lon'][:,:]
        x['time'] = nc.variables['date'][:]

        x['description'] = nc.description

        x[var] = nc.variables['tair'][:,:,:]

    return x


def write_nc(
        var, 
        lat, 
        lon, 
        dates, 
        filename = 'tmp.nc', 
        var_sname = 'tmp', 
        description = 'Description', 
        path = './'
        ) -> None:
    '''
    Write data, and additional information such as latitude and longitude and timestamps, to a .nc file
    
    Inputs:
    :param var: The variable being written (np.ndarray with shape time x lat x lon)
    :param lat: The latitude data (np.ndarray with shape lat)
    :param lon: The longitude data (np.ndarray with shape lon)
    :param dates: Timestamp for each day in var in a %Y-%m-%d format (np.ndarray with shape time)
    :param filename: Filename of the .nc file being written
    :param var_sname: The short name of the variable being written. I.e., the name used
                      to call the variable in the .nc file
    :param description: A string descriping the data
    :param path: Directory path to the directory the data will be written in
    '''
    
    # Determine the spatial and temporal lengths
    T, I, J = var.shape
    T = len(dates)
    
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
        nc.variables['date'][:] = dates[:]
            
        # Create the main variable
        nc.createVariable(var_sname, var.dtype, ('time', 'x', 'y'))
        nc.variables[str(var_sname)][:,:,:] = var[:,:,:]

if __name__ == '__main__':
    # Filename information
    dataset_dir = './precipitation'
    fbase = 'total_precipitation_'
    true_sname = 'tp'

    # Collect filenames
    files = ['%s/%s'%(dataset_dir,f) for f in os.listdir(dataset_dir) if re.match(r'%s.+.nc'%(fbase), f)]
    files.sort()

    print(files)

    # Iterate through each file
    for file in files:
        print(file)

        # Load the data
        data = load_nc(file, true_sname)

        # Write the file again, with the correct name
        write_nc(data[true_sname], 
                 data['lat'], 
                 data['lon'], 
                 data['time'], 
                 filename = file, 
                 var_sname = true_sname, 
                 description = data['description'])