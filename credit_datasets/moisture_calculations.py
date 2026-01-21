import numpy as np
from netCDF4 import Dataset

from path_to_raw_datasets import path_to_raw_datasets, get_var_shortname, get_fn

def calculate_q_total_upper_level(year, level):
    '''
    Calculate the total specific humidity for the upper air using ERA5
    specific humidity (kg kg^-1), total cloud liquid water content (kg kg^-1), and total cloud ice water content (kg kg^-1)
    '''

    variables = ['specific_humidity', 'cloud_liquid_water_content', 'cloud_ice_water_content']

    # Get short names
    sname_qtot = get_var_shortname('total_specific_humidity')

    sname_q = get_var_shortname(variables[0])
    sname_clw = get_var_shortname(variables[1])
    sname_ciw = get_var_shortname(variables[2])
    snames = [sname_q, sname_clw, sname_ciw]

    data_final = {}
    for n, var in enumerate(variables):
        path = path_to_raw_datasets(var, 'era5', level = level)
        filename = get_fn(var, year, level = level)

        data = load_nc('%s/%s.nc'%(path, filename), snames[n])

        # If the total specific humidity is not initialized, initialize it
        if sname_qtot not in data_final.keys():
            data_final[sname_qtot] = data[snames[n]]
        else:
            # If the total specific humidity is initialized, add the next part (q_tot = q + clw + ciw and ERA5 units are already consistent)
            data_final[sname_qtot] = data_final[sname_qtot] + data[snames[n]]

        data_final['lat'] = data['lat']; data_final['lon'] = data['lon']; data_final['time'] = data['time']

    return data_final

def calculate_q_total_surface(year):
    '''
    Calculate the total specific humidity for the surface using ERA5 
    dewpoint (K), pressure (Pa), and total column rain and snow water (kg m^-2))
    '''

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

    # Remaining variables for use
    variables = ['total_rain_water', 'total_snow_water']

    # Get remaining variable short names
    sname_rw = get_var_shortname(variables[0])
    sname_sw = get_var_shortname(variables[1])
    snames = [sname_rw, sname_sw]

    # Density of liquid water and ice for unit conversion
    rho = [1000, 600] # kg m^-3; note there is strong variation in the density of snow

    for n, var in enumerate(variables):
        path = path_to_raw_datasets(var, 'era5')
        filename = get_fn(var, year)

        data = load_nc('%s/%s.nc'%(path, filename), snames[n])

        # Unit conversion from kg m^-2 to kg kg^-1

        # If the total specific humidity is initialized, add the next part (q_tot = q + rw + sw and ERA5 units are already consistent)
        data_final[sname_qtot] = data_final[sname_qtot] + data_final[snames[n]]

        data_final['lat'] = data['lat']; data_final['lon'] = data['lon']

    return data_final

def calculate_q_surface(dewpoint, pressure, convert_to_celsius = True):
    '''
    Calculate specific humidity, q (kg kg^-1), from the dewpoint temperature (K) and pressure (Pa)
    
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

def load_nc(filename, var):
    '''
    Load a nc file
    '''

    x = {}

    with Dataset(filename, 'r') as nc:
        x['lat'] = nc.variables['lat'][:,:]
        x['lon'] = nc.variables['lon'][:,:]
        x['time'] = nc.variables['date'][:]

        x[var] = nc.variables[var][:,:,:]

    return x