import numpy as np
# For GLDAS: wind_speed: wspd, temperature: temp, evaporation: evap, potential_evaporation: pevap, precipitation: precip, pressure: pres, net_radiation: rnet, soil_moisture: soilm 
# For IMERG: precipitation: precip
# For MODIS: potential_evaporation: pevap, evaporation: pevap
# TODO: Add pathing for MODIS and MODIS variables in path_to_raw_datasets, get_var_shortname, and get_fn
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
    'ndvi',
    'evi',
    'lai',
    'fpar',
    'enso',
    'amo',
    'nao',
    'pdo',
    'iod',
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
    'land_cover'
    ]

def path_to_raw_datasets(variable, reanalysis = 'era5', level = None):

    base_path = '/ourdisk/hpc/ai2es/sedris/'

    # Get the name of the directory for the variable
    if variable == 'temperature':
        dir_name = '%s/%s'%(reanalysis, 'temperature')
    elif (variable == 'dewpoint') | (variable == 'total_rain_water') | (variable == 'total_snow_water'):
        dir_name = '%s/%s'%(reanalysis, 'moisture_surface')
    elif variable == 'pressure':
        dir_name = '%s/%s'%(reanalysis, 'pressure')
    elif 'precipitation' in variable:
        dir_name = '%s/%s'%(reanalysis, 'precipitation') if np.invert(reanalysis == 'imerg') else reanalysis
    elif variable == 'radiation':
        dir_name = '%s/%s'%(reanalysis, 'radiation') if reanalysis == 'era5' else '%s/%s'%(reanalysis, 'net_radiation')
    elif variable == 'evaporation':
        if (reanalysis == 'era5') | (reanalysis == 'gldas'):
            dir_name = '%s/%s'%(reanalysis, 'evaporation')
        elif reanalysis == 'modis':
            dir_name = '%s/global/%s'%(reanalysis, 'evaporation')
    elif variable == 'potential_evaporation':
        if (reanalysis == 'era5') | (reanalysis == 'gldas'):
            dir_name = '%s/%s'%(reanalysis, 'potential_evaporation')
        elif reanalysis == 'modis':
            dir_name = '%s/global/%s'%(reanalysis, 'potential_evaporation')
    elif variable == 'ndvi':
        dir_name = '%s/global/%s'%(reanalysis, 'ndvi')
    elif variable == 'evi':
        dir_name = '%s/global/%s'%(reanalysis, 'evi')
    elif variable == 'lai':
        dir_name = '%s/global/%s'%(reanalysis, 'lai')
    elif variable == 'fpar':
        dir_name = '%s/global/%s'%(reanalysis, 'fpar')
    elif variable == 'wind_speed':
        dir_name = '%s/%s'%(reanalysis, 'wind_speed')
    elif variable == 'wind_gusts':
        dir_name = '%s/%s'%(reanalysis, 'wind_gusts')
    elif (variable == 'enso') | (variable  == 'amo') | (variable == 'nao') | (variable == 'pdo') | (variable == 'iod'):
        dir_name = 'climate_indices'
    elif (variable == 'soil_moisture_1') | (variable == 'soil_moisture_2') | (variable == 'soil_moisture_3') | (variable == 'soil_moisture_4'):
        if reanalysis == 'era5':
            dir_name = '%s/%s'%(reanalysis, 'liquid_vsm')
        elif reanalysis == 'gldas':
            if variable == 'soil_moisture_1':
                dir_name = '%s/%s'%(reanalysis, 'soil_moisture_0-10cm')
            elif variable == 'soil_moisture_2':
                dir_name = '%s/%s'%(reanalysis, 'soil_moisture_10-40cm')
            elif variable == 'soil_moisture_3':
                dir_name = '%s/%s'%(reanalysis, 'soil_moisture_40-100cm')
            elif variable == 'soil_moisture_4':
                dir_name = '%s/%s'%(reanalysis, 'soil_moisture_100-200cm')
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
    elif (variable == 'land_cover'):
        dir_name = '%s/global/%s'%(reanalysis, variable)

    # Construct the path
    path = base_path + dir_name

    return path

def get_var_shortname(variable, reanalysis = 'era5'):
    # TODO:
    # - Modify for wind speed and wind gust names
    # - Modify to get snames from GLDAS, IMERG, and MODIS
    
    # Get the short name of the variable
    if variable == 'temperature':
        sname = 'tair' if reanalysis == 'era5' else 'temp'
    elif variable == 'dewpoint':
        sname = 'd2m'
    elif variable == 'total_rain_water':
        sname = 'tcrw'
    elif variable == 'total_snow_water':
        sname = 'tcsw'
    elif variable == 'pressure':
        sname = 'sp' if reanalysis == 'era5' else 'pres'
    elif variable == 'precipitation':
        sname = 'tp' if reanalysis == 'era5' else 'precip' # sname is precip for both GLDAS2 and IMERG
    elif variable == 'precipitation_7day':
        sname = 'tp_7d'
    elif variable == 'precipitation_14day':
        sname = 'tp_14d'
    elif variable == 'precipitation_30day':
        sname = 'tp_30d'
    elif variable == 'radiation':
        sname = 'ssr' if reanalysis == 'era5' else 'rnet'
    elif variable == 'evaporation':
        if reanalysis == 'era5':
            sname = 'e'
        elif reanalysis == 'gldas':
            sname = 'evap'
        elif reanalysis == 'modis':
            sname = 'pevap' # From a typo/error in making the MODIS nc files
    elif variable == 'potential_evaporation':
        if reanalysis == 'era5':
            sname = 'pev'
        elif reanalysis == 'gldas':
            sname = 'pevap'
        elif reanalysis == 'modis':
            sname = 'pevap'
    elif variable == 'ndvi':
        sname = 'ndvi'
    elif variable == 'evi':
        sname = 'evi'
    elif variable == 'lai':
        sname = 'lai'
    elif variable == 'fpar':
        sname = 'fpar'
    elif variable == 'total_specific_humidity':
        sname = 'q_tot'
    elif variable == 'wind_speed':
        sname = 'ws' if reanalysis == 'era5' else 'wspd'
    elif variable == 'wind_speed_u':
        sname = 'u10'
    elif variable == 'wind_speed_v':
        sname = 'v10'
    elif variable == 'wind_gusts':
        sname = 'fg10'
    elif variable == 'enso':
        sname = 'enso'
    elif variable == 'amo':
        sname = 'amo'
    elif variable == 'nao':
        sname = 'nao'
    elif variable == 'pdo':
        sname = 'pdo'
    elif variable == 'iod':
        sname = 'iod'
    elif variable == 'soil_moisture_1':
        sname = 'swvl1' if reanalysis == 'era5' else 'soilm'
    elif variable == 'soil_moisture_2':
        sname = 'swvl2' if reanalysis == 'era5' else 'soilm'
    elif variable == 'soil_moisture_3':
        sname = 'swvl3' if reanalysis == 'era5' else 'soilm'
    elif variable == 'soil_moisture_4':
        sname = 'swvl4' if reanalysis == 'era5' else 'soilm'
    elif variable == 'sesr':
        sname = 'sesr' # sname should be sesr for all three datasets
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
    elif variable == 'land_cover':
        sname = 'lc'
    elif variable == 'surface_geopotential_var':
        sname = 'surface_geopotential_var'

    return sname

# function to get base filename
def get_fn(variable, year, reanalysis = 'era5', level = None):
    '''
    Collect filename
    '''

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
        if reanalysis == 'era5':
            base = 'total_precipitation_'
        elif reanalysis == 'gldas':
            base = 'gldas.precipitation.daily.'
        elif reanalysis == 'imerg':
            base = 'imerg.precipitation.daily.'
    elif variable == 'temperature':
        base = '2m_temperature_' if reanalysis == 'era5' else 'gldas.temperature.daily.'
    elif variable == 'dewpoint':
        base = '2m_dewpoint_temperature_'
    elif variable == 'total_rain_water':
        base = 'total_column_rain_water_'
    elif variable == 'total_snow_water':
        base = 'total_column_snow_water_'
    elif variable == 'pressure':
        base = 'surface_pressure_' if reanalysis == 'era5' else 'gldas.pressure.daily.'
    elif variable == 'wind_speed_u':
        base = '10m_u_component_of_wind_'
    elif variable == 'wind_speed_v':
        base = '10m_v_component_of_wind_'
    elif variable == 'wind_speed': # Only valid for GLDAS
        base = 'gldas.wind_speed.daily.'
    elif variable == 'wind_gusts':
        base = '10m_wind_gust_since_previous_post_processing_'
    elif variable == 'evaporation':
        if reanalysis == 'era5':
            base = 'evaporation_'
        elif reanalysis == 'gldas':
            base = 'gldas.evaporation.daily.'
        elif reanalysis == 'modis':
            base = 'modis.evaporation.8-day.%d'%year # Adding year at the end excludes any oddly named files
    elif variable == 'potential_evaporation':
        if reanalysis == 'era5':
            base = 'potential_evaporation_'
        elif reanalysis == 'gldas':
            base = 'gldas.potential_evaporation.daily.'
        elif reanalysis == 'modis':
            base = 'modis.potential_evaporation.8-day.%d'%year # Adding year at the end excludes any oddly named files
    elif variable == 'ndvi':
        base = 'modis.ndvi.8-day.%d'%year # Adding year at the end excludes any oddly named files
    elif variable == 'evi':
        base = 'modis.evi.8-day.%d'%year # Adding year at the end excludes any oddly named files
    elif variable == 'lai':
        base = 'modis.lai.8-day.%d'%year # Adding year at the end excludes any oddly named files
    elif variable == 'fpar':
        base = 'modis.fpar.8-day.%d'%year # Adding year at the end excludes any oddly named files
    elif variable == 'high_vegetation_cover':
        base = 'high_vegetation_cover_'
    elif variable == 'high_vegetation_type':
        base = 'type_of_high_vegetation_'
    elif variable == 'low_vegetation_cover':
        base = 'low_vegetation_cover_'
    elif variable == 'low_vegetation_type':
        base = 'type_of_low_vegetation_'
    elif variable == 'enso':
        return 'enso.timeseries'
    elif variable == 'amo':
        return 'amo.timeseries'
    elif variable == 'nao':
        return 'nao.long'
    elif variable == 'pdo':
        return 'pdo.timeseries.sstens'
    elif variable == 'iod':
        return 'dmi.had.long'
    elif variable == 'radiation':
        base = 'surface_net_solar_radiation_' if reanalysis == 'era5' else 'gldas.net_radiation.daily.'
    elif variable == 'soil_moisture_1':
        base = 'volumetric_soil_water_layer_1_' if reanalysis == 'era5' else 'gldas.soil_moisture_0-10cm.daily.'
    elif variable == 'soil_moisture_2':
        base = 'volumetric_soil_water_layer_2_' if reanalysis == 'era5' else 'gldas.soil_moisture_10-40cm.daily.'
    elif variable == 'soil_moisture_3':
        base = 'volumetric_soil_water_layer_3_' if reanalysis == 'era5' else 'gldas.soil_moisture_40-100cm.daily.'
    elif variable == 'soil_moisture_4':
        base = 'volumetric_soil_water_layer_4_' if reanalysis == 'era5' else 'gldas.soil_moisture_100-200cm.daily.'
    elif variable == 'sesr':
        if (reanalysis == 'era5') | (reanalysis == 'gldas'):
            base = 'sesr_'
        elif reanalysis == 'modis':
            base = 'modis.sesr.8-day.'
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
    elif variable == 'land_cover':
        fn = 'modis.land_cover.nc'
        return fn

    if np.invert(reanalysis == 'modis'):
        fn = '%s%d'%(base, year) if level is None else '%s%d_%dmb'%(base, year, level)
    elif reanalysis == 'modis':
        fn = '%d/%s'%(year, base)

    return fn