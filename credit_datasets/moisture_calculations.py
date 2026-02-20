"""General moisture calculations for data preparation

Provides functions for moisture related calculations,
such as total specific humidity, and surface specific humidity
"""

import numpy as np
from typing import Dict
from netCDF4 import Dataset

from path_to_raw_datasets import path_to_raw_datasets, get_var_shortname, get_fn

def calculate_q_total_upper_level(year, level) -> Dict[str, np.ndarray]:
    '''
    Calculate the total specific humidity for the upper air using ERA5
    specific humidity (kg kg^-1), total cloud liquid water content (kg kg^-1), 
    and total cloud ice water content (kg kg^-1)

    Inputs:
    :param year: The year of data to load moisture components and make calculations for
    :param level: Index of the elevation level (e.g., if 200 mb is first, its index/level = 0)

    Outputs:
    :param data_final: Dictionary with calculated total specific humidity data for one year 
                       (key is 'q_tot', np.ndarray with shape time x lat x lon),
                       gridded latitude and longitude (keys 'lat' and 'lon' and 
                       np.ndarrays with shape lat x lon), and time information
                       (key 'time', np.ndarray with shape time)
    '''

    # Components needed to calculate total specific humidity
    variables = ['specific_humidity', 'cloud_liquid_water_content', 'cloud_ice_water_content']

    # Get short names
    sname_qtot = get_var_shortname('total_specific_humidity')

    sname_q = get_var_shortname(variables[0])
    sname_clw = get_var_shortname(variables[1])
    sname_ciw = get_var_shortname(variables[2])
    snames = [sname_q, sname_clw, sname_ciw]

    data_final = {}
    for n, var in enumerate(variables):
        # Get the path and filename for the given variable
        path = path_to_raw_datasets(var, 'era5', level = level)
        filename = get_fn(var, year, level = level)

        # Load the moisture variable
        data = load_nc('%s/%s.nc'%(path, filename), snames[n])

        # If the total specific humidity is not initialized, initialize it
        if sname_qtot not in data_final.keys():
            data_final[sname_qtot] = data[snames[n]]
        else:
            # If the total specific humidity is initialized, add the next part (q_tot = q + clw + ciw and ERA5 units are already consistent)
            data_final[sname_qtot] = data_final[sname_qtot] + data[snames[n]]

        # Add the latitude, longitude, and time information
        data_final['lat'] = data['lat']; data_final['lon'] = data['lon']; data_final['time'] = data['time']

    return data_final

def calculate_q_total_surface(year) -> Dict[str, np.ndarray]:
    '''
    Calculate the total specific humidity for the surface using ERA5 
    dewpoint (K), pressure (Pa), and total column rain and snow water (kg m^-2))

    NOTE: THIS FUNCTION IS INCOMPLETE DUE TO AN ISSUE IN GETTING SNOW WATER COLUMN TO CONVERT TO THE PROPER UNITS
    (ISSUE WITH DETERMINING THE PROPER DENSITY OF SNOW FOR UNIT CONVERSION AND PROPER COMPARISON)

    Inputs:
    :param year: The year of data to load moisture components and make calculations for

    Outputs:
    :param data_final: Dictionary with calculated total specific humidity data for one year 
                       (key is 'q_tot', np.ndarray with shape time x lat x lon),
                       gridded latitude and longitude (keys 'lat' and 'lon' and 
                       np.ndarrays with shape lat x lon), and time information
                       (key 'time', np.ndarray with shape time)
    '''

    # Components needed to calculate total specific humidity
    intitial_variables = ['dewpoint', 'pressure']

    # Get short names
    sname_qtot = get_var_shortname('total_specific_humidity')
    data_final = {}

    sname_tdew = get_var_shortname(intitial_variables[0])
    sname_sp = get_var_shortname(intitial_variables[1])

    # Load in dewpoint
    path = path_to_raw_datasets(intitial_variables[0], 'era5')
    filename = get_fn(intitial_variables[0], year)

    tdew = load_nc('%s/%s.nc'%(path, filename), sname_tdew)

    # Load in surface pressure
    path = path_to_raw_datasets(intitial_variables[1], 'era5')
    filename = get_fn(intitial_variables[1], year)

    sp = load_nc('%s/%s.nc'%(path, filename), sname_sp)

    # Calculate the surface specific humidity
    data_final[sname_qtot] = calculate_q_surface(tdew[sname_tdew], sp[sname_sp])

    # Remaining variables for total specific humidity
    variables = ['total_rain_water', 'total_snow_water']

    # Get remaining variable short names
    sname_rw = get_var_shortname(variables[0])
    sname_sw = get_var_shortname(variables[1])
    snames = [sname_rw, sname_sw]

    # Density of liquid water and ice for unit conversion
    rho = [1000, 600] # kg m^-3; note there is strong variation in the density of snow

    for n, var in enumerate(variables):
        # Get the path and filename for the given variable
        path = path_to_raw_datasets(var, 'era5')
        filename = get_fn(var, year)

        # Load the moisture variable
        data = load_nc('%s/%s.nc'%(path, filename), snames[n])

        # Unit conversion from kg m^-2 to kg kg^-1
        # MISSING STEP TO GET column rain water and snow water to proper units

        # Add the next part (q_tot = q + rw + sw)
        data_final[sname_qtot] = data_final[sname_qtot] + data_final[snames[n]]

        # Add the latitude, longitude, and time information
        data_final['lat'] = data['lat']; data_final['lon'] = data['lon']

    return data_final

def calculate_q_surface(
        dewpoint, 
        pressure, 
        convert_to_celsius: bool = True
        ) -> np.ndarray:
    '''
    Calculate specific humidity, q (kg kg^-1), from the dewpoint temperature (K) and pressure (Pa)
    
    Inputs:
    :paran dewpoint: Input dewpoint temperature (np.ndarray with shape time x lat x lon)
    :param pressure: Input pressure (np.ndarray with shape time x lat x lon)
    :param convert_to_celsius: Boolean indicating whether convert dewpoint from Kelvin to Celsius 
                               (required to use empirical CC equation)

    Outputs:
    :param q: The calculated specific humidity (np.ndarray with shape time x lat x lon)
    '''

    # The empirical form of the CC equation uses temperature/dewpoint in Celsius
    if convert_to_celsius:
        dewpoint = dewpoint - 273.15

    # Coefficients to empirical CC equation
    e0 = 611.2 # Pa
    a = 17.67
    b = 243.5

    # Use the empirical CC equation to get vapor pressure from dewpoint
    exponent = a * dewpoint/(dewpoint + b)
    e = e0 * np.exp(exponent)

    # Convert vapor pressure to specific humidity
    eps = 0.622
    q = eps * e / (pressure - ((1 - eps) * e)) # Since pressure and e are in Pa, then q is in kg kg^-1

    return q

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