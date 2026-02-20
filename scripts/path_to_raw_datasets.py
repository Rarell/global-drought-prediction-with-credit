"""Provides functions to construct directory paths, 
filenames, and shortnames (used in zarr and .nc files)
based on variable examined
"""

# List of available variables in the complete dataset
AVAIL_VARIABLES = [
    'u', 
    'v', 
    'geopotential', 
    'specific_humidity', 
    'cloud_liquid_water_content', 
    'cloud_ice_water_content',
    'temperature',
    'precipitation',
    'precipitation_7day',
    'precipitation_14day',
    'precipitation_30day',
    'pressure',
    'dewpoint',
    'total_rain_water',
    'total_snow_water',
    'evaporation',
    'potential_evaporation',
    'total_specific_humidity',
    'radiation',
    'wind_speed',
    'wind_gusts',
    'high_vegetation_cover',
    'high_vegetation_type',
    'low_vegetation_cover',
    'low_vegetation_type',
    'sesr', 
    'soil_moisture_1',
    'soil_moisture_2',
    'soil_moisture_3',
    'soil_moisture_4'
    'fdii_1', 
    'fdii_2', 
    'fdii_3', 
    'fdii_4', 
    'land-sea', 'mask', # These last two are different names for the same thing
    ]

def path_to_raw_datasets(
        variable, 
        reanalysis, 
        level = None
        ) -> str:
    '''
    Construct a directory path to the raw data based 
    on the variable and reanalysis the data belongs to
    
    Inputs:
    :param variable: Name of the variable to point to
    :param reanalysis: Name of the reanalysis the raw data is located in
    :param level: Pressure level in mb/hPa (500 or 200) the data corresponds to (None if not upper air)
    
    Outputs:
    :param path: Directory path to the raw data files for variable
    '''

    base_path = '/ourdisk/hpc/ai2es/sedris/'

    # Get the name of the directory for the variable
    if variable == 'temperature':
        dir_name = '%s/%s'%(reanalysis, 'temperature')
    elif (variable == 'dewpoint') | (variable == 'total_rain_water') | (variable == 'total_snow_water'):
        dir_name = '%s/%s'%(reanalysis, 'moisture_surface')
    elif variable == 'pressure':
        dir_name = '%s/%s'%(reanalysis, 'pressure')
    elif 'precipitation' in variable:
        dir_name = '%s/%s'%(reanalysis, 'precipitation')
    elif variable == 'radiation':
        dir_name = '%s/%s'%(reanalysis, 'radiation')
    elif variable == 'evaporation':
        dir_name = '%s/%s'%(reanalysis, 'evaporation')
    elif variable == 'potential_evaporation':
        dir_name = '%s/%s'%(reanalysis, 'potential_evaporation')
    elif variable == 'wind_speed':
        dir_name = '%s/%s'%(reanalysis, 'wind_speed')
    elif variable == 'wind_gusts':
        dir_name = '%s/%s'%(reanalysis, 'wind_gusts')
    elif (variable == 'soil_moisture_1') | (variable == 'soil_moisture_2') | (variable == 'soil_moisture_3') | (variable == 'soil_moisture_4'):
        dir_name = '%s/%s'%(reanalysis, 'liquid_vsm')
    elif (variable == 'sesr') | (variable == 'fdii_1') | (variable == 'fdii_2') | (variable == 'fdii_3') | (variable == 'fdii_4'):
        dir_name = '%s/%s'%(reanalysis, 'fd_indices')
    elif (variable == 'high_vegetation_cover') | (variable == 'high_vegetation_type'):
        dir_name = '%s/%s'%(reanalysis, 'high_vegetation')
    elif (variable == 'low_vegetation_cover') | (variable == 'low_vegetation_type'):
        dir_name = '%s/%s'%(reanalysis, 'low_vegetation')
    elif variable == 'u':
        dir_name = '%s/%s_%smb'%(reanalysis, 'uwind', level)
    elif variable == 'v':
        dir_name = '%s/%s_%smb'%(reanalysis, 'vwind', level)
    elif variable == 'geopotential':
        dir_name = '%s/%s_%smb'%(reanalysis, 'geopotential', level)
    elif (variable == 'specific_humidity') | (variable == 'cloud_liquid_water_content') | (variable == 'cloud_ice_water_content'):
        dir_name = '%s/%s_%smb'%(reanalysis, 'moisture', level)
    elif (variable == 'land-sea') | (variable == 'mask'):
        dir_name = '%s'%(reanalysis) # Note the land-sea mask is sitting in an nc file in the main era5 directory (no sub-directory)

    # Construct the path
    path = base_path + dir_name

    return path

def get_var_shortname(variable) -> str:
    '''
    Retrieve the short name of a variable (i.e., the variable's key in .nc files)

    Inputs:
    :param variable: Name of the variable investigated

    Outputs:
    :param sname: Short name of the variable
    '''
    
    # Get the short name of the variable
    if variable == 'temperature':
        sname = 'tair'
    elif variable == 'dewpoint':
        sname = 'd2m'
    elif variable == 'total_rain_water':
        sname = 'tcrw'
    elif variable == 'total_snow_water':
        sname = 'tcsw'
    elif variable == 'pressure':
        sname = 'sp'
    elif variable == 'precipitation':
        sname = 'tp'
    elif variable == 'precipitation_7day':
        sname = 'tp_7d'
    elif variable == 'precipitation_14day':
        sname = 'tp_14d'
    elif variable == 'precipitation_30day':
        sname = 'tp_30d'
    elif variable == 'radiation':
        sname = 'ssr'
    elif variable == 'evaporation':
        sname = 'e'
    elif variable == 'potential_evaporation':
        sname = 'pev'
    elif variable == 'total_specific_humidity':
        sname = 'q_tot'
    elif variable == 'wind_speed':
        sname = 'ws'
    elif variable == 'wind_speed_u':
        sname = 'u10'
    elif variable == 'wind_speed_v':
        sname = 'v10'
    elif variable == 'wind_gusts':
        sname = 'fg10'
    elif variable == 'soil_moisture_1':
        sname = 'swvl1'
    elif variable == 'soil_moisture_2':
        sname = 'swvl2'
    elif variable == 'soil_moisture_3':
        sname = 'swvl3'
    elif variable == 'soil_moisture_4':
        sname = 'swvl4'
    elif variable == 'sesr':
        sname = 'sesr'
    elif variable == 'fdii_1':
        sname = 'fdii1'
    elif variable == 'fdii_2':
        sname = 'fdii2'
    elif variable == 'fdii_3':
        sname = 'fdii3'
    elif variable == 'fdii_4':
        sname = 'fdii4'
    elif variable == 'high_vegetation_cover':
        sname = 'cvh'
    elif variable == 'high_vegetation_type':
        sname = 'tvh'
    elif variable == 'low_vegetation_cover':
        sname = 'cvl'
    elif variable == 'low_vegetation_type':
        sname = 'tvl'
    elif variable == 'u':
        sname = 'u'
    elif variable == 'v':
        sname = 'v'
    elif variable == 'geopotential':
        sname = 'z'
    elif variable == 'specific_humidity':
        sname = 'q'
    elif variable == 'cloud_liquid_water_content':
        sname = 'clwc'
    elif variable == 'cloud_ice_water_content':
        sname = 'ciwc'
    elif (variable == 'land-sea') | (variable == 'mask'):
        sname = 'lsm'
    elif variable == 'surface_geopotential_var':
        sname = 'surface_geopotential_var'

    return sname

# function to get base filename
def get_fn(
        variable, 
        year, 
        level = None
        ) -> str:
    '''
    Construct the filename of the raw .nc dataset

    Inputs:
    :param variable: Name of the variable to point to
    :param year: The year of the data to be loaded
    :param level: Pressure level in mb/hPa (500 or 200) the data corresponds to (None if not upper air)
    
    Outputs:
    :param path: Directory path to the raw data files for variable
    '''

    # Get the base of the filename (based on variable)
    if variable == 'u':
        base = 'u_component_of_wind_'
    elif variable == 'v':
        base = 'v_component_of_wind_'
    elif variable == 'geopotential':
        base = 'geopotential_'
    elif variable == 'specific_humidity':
        base = 'specific_humidity_'
    elif variable == 'cloud_liquid_water_content':
        base = 'specific_cloud_liquid_water_content_'
    elif variable == 'cloud_ice_water_content':
        base = 'specific_cloud_ice_water_content_'
    elif 'precipitation' in variable:
        base = 'total_precipitation_'
    elif variable == 'temperature':
        base = '2m_temperature_'
    elif variable == 'dewpoint':
        base = '2m_dewpoint_temperature_'
    elif variable == 'total_rain_water':
        base = 'total_column_rain_water_'
    elif variable == 'total_snow_water':
        base = 'total_column_snow_water_'
    elif variable == 'pressure':
        base = 'surface_pressure_'
    elif variable == 'wind_speed_u':
        base = '10m_u_component_of_wind_'
    elif variable == 'wind_speed_v':
        base = '10m_v_component_of_wind_'
    elif variable == 'wind_gusts':
        base = '10m_wind_gust_since_previous_post_processing_'
    elif variable == 'evaporation':
        base = 'evaporation_'
    elif variable == 'potential_evaporation':
        base = 'potential_evaporation_'
    elif variable == 'high_vegetation_cover':
        base = 'high_vegetation_cover_'
    elif variable == 'high_vegetation_type':
        base = 'type_of_high_vegetation_'
    elif variable == 'low_vegetation_cover':
        base = 'low_vegetation_cover_'
    elif variable == 'low_vegetation_type':
        base = 'type_of_low_vegetation_'
    elif variable == 'radiation':
        base = 'surface_net_solar_radiation_'
    elif variable == 'soil_moisture_1':
        base = 'volumetric_soil_water_layer_1_'
    elif variable == 'soil_moisture_2':
        base = 'volumetric_soil_water_layer_2_'
    elif variable == 'soil_moisture_3':
        base = 'volumetric_soil_water_layer_3_'
    elif variable == 'soil_moisture_4':
        base = 'volumetric_soil_water_layer_4_'
    elif variable == 'sesr':
        base = 'sesr_'
    elif variable == 'fdii_1':
        base = 'fdii_1_'
    elif variable == 'fdii_2':
        base = 'fdii_2_'
    elif variable == 'fdii_3':
        base = 'fdii_3_'
    elif variable == 'fdii_4':
        base = 'fdii_4_'
    elif (variable == 'land-sea') | (variable == 'mask'):
        fn = 'land.nc' # For the land-sea mask, only 1 nc is needed, not a suite for each year
        return fn

    # Construct the filename
    fn = '%s%d'%(base, year) if level is None else '%s%d_%dmb'%(base, year, level)

    return fn