"""
Script to download ERA5 data and reduce it to a more managable daily timescale (1 year of data at the daily times scale is ~ 3 GB per variable)

NOTE: Though the script is set up to download multiple variables at a time, in practice 2 variables
violates some of the constraints in the grib_to_cdf code in cdsapi, and makes the dataset too large
for the classic netCDF format. In short, 2+ variables will return an error.

Acceptable names for --variable and --var_sname_era (i.e., variable names in the downloaded ERA5 dataset):
    --variable:
    
    (Variables for single levels dataset)
    - Temperature: 2m_temperature
    - Precipitation: total_precipitation
    - Dewpoint: 2m_dewpoint_temperature
    - Pressure: surface_pressure
    - Net Radiation: surface_net_radiation
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
      
    (Variables for daily statistics on pressure levels dataset)
    - Geopotential: geopotential
      (Geopotential height, Phi, can be obtained from geopotential divided by 9.80665 m s^-2)
    - u-component of Wind: u_component_of_wind
    - v-component of Wind: v_component_of_wind
    
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
    
    - Geopotential: z
    - u-component of Wind:
    - v-component of Wind:
    
Variable units in ERA5 dataset:
    - Temperature: K
    - Precipitation: m
    - Dewpoint: K
    - Pressure: Pa
    - Net Radiation: J m^-2
    - Evaporation: m
    - Potential Evaporation: m
    - Runoff: m
    - Soil Moisture: Unitless (m^3 m^-3)
    - Vegetation Variables: Dimensionless
    
    - Geopotential: m^2 s^-2
    - Wind components: m s^-1
"""

import os, warnings
import cdsapi
import numpy as np
import argparse
from netCDF4 import Dataset
from datetime import datetime, timedelta

from era5_downloader import downloader

# Function to create a parser using the terminal
def create_parser():
    '''
    Create argument parser
    '''
    
    # To add: args.time_series, args.climatology_plot, args.case_studies, args.case_study_years
              
    
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='ERA Downloader', fromfile_prefix_chars='@')

    parser.add_argument('--variable', type=str, nargs='+', default=['2m_temperature'], help='Variable to download (Note only 1 variable should be downloaded at a time)')
    parser.add_argument('--var_sname_era', type=str, nargs='+', default=['t2m'], help='The short name for --variable used in raw datafile')
    parser.add_argument('--var_sname', type=str, nargs='+', default=['tair'], help='What to call the short name for --variable in the processed nc file')
    parser.add_argument('--test', action='store_true', help='Perform a test download (only retrieves 1 year of data)?')
    parser.add_argument('--years', type=int, nargs=2, default=[1979,2021], help='Beginning and ending years to download data for.')
    parser.add_argument('--era5_dataset', type=str, default='reanalysis-era5-single-levels', help='ERA5 dataset to collect from (e.g., reanalysis-era5-single-levels or era5_dataset=derived-era5-pressure-levels-daily-statistics)')
    parser.add_argument('--pressure_level', type=int, default=500, help='Pressure level (in hPa) of the data to be downloaded (only used for datasets that use pressure levels)')
    parser.add_argument('--process', action='store_true', help='Process the data (downscales data to daily scale, which reduces the size by 1/6th)')
    
    return parser
    

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
    
    # Do a test run?
    if args.test:
        downloader(args.variable, 2021, dataset = args.era5_dataset, pressure = args.pressure_level)
    else:
        years = np.arange(args.years[0]+2000, args.years[1]+2000+1)
        print(years)
        for year in years:
            print('Downloading data for %d...'%year)

            fn = '%s_%d.nc'%(args.variable[0], year) if args.era5_dataset == 'reanalysis-era5-single-levels' else '%s_%d_mb_%d.nc'%(args.variable[0], args.pressure_level, year)
            if os.path.exists(fn):
                print('Dataset has already been reduced to daily timescale.')
                continue

            if args.era5_dataset == 'reanalysis-era5-single-levels':
                filename = 'era5_%s_%d.nc'%(args.variable[0], year)
                start_year = 1970
            elif (args.era5_dataset == 'derived-era5-pressure-levels-daily-statistics') | (args.era5_dataset == 'reanalysis-era5-pressure-levels'):
                filename = 'era5_%s_%d_%dmb.nc'%(args.variable[0], year, pressure)
                start_year = 1980
                
                
            if os.path.exists(filename):
            # Processed file does exist: exit
                print("Data is already downloaded.")
            else:
            	downloader(args.variable, year, dataset = args.era5_dataset, pressure = args.pressure_level)
            print('Data downloaded, processing to daily timescale')
            # Process the data so it isn't so large?
            if args.process:
                for v, variable in enumerate(args.variable):
                    fn = '%s_%d.nc'%(variable, year) if args.era5_dataset == 'reanalysis-era5-single-levels' else '%s_%d_mb_%d.nc'%(variable, args.pressure_level, year)
                    if os.path.exists(fn):
                        print('Dataset has already been reduced to daily timescale.')
                        continue

                    print('Reducing %s to daily scale for %d...'%(variable, year))
                    # First, load in the data
                    with Dataset(filename, 'r') as nc:
                        # Load in lat and lon
                        lat = nc.variables['latitude'][:]
                        lon = nc.variables['longitude'][:]
                    
                        # Collect the time + dates
                        time = nc.variables['valid_time'][:]
                        dates = np.asarray([datetime(start_year,1,1) + timedelta(seconds = int(t)) for t in time])
                    
                        # Initialize the main variable
                        T = time.size; I = lat.size; J = lon.size
                        var = np.ones((int(T/24), I, J)) * np.nan
                        n = 0
                    
                        # Mesh the lat/lon grid
                        lon, lat = np.meshgrid(lon, lat)
                    
                        # Rather than load in the entire (18 GB) dataset, load in a small portion and immediately reduce it to daily size to ease computation
                        for t in range(int(T/24)):
                            print('On day %d'%t)
                            if args.era5_dataset == 'reanalysis-era5-single-levels':
                                if (variable == 'total_precipitation') | (variable == 'total_column_rain_water') | (variable == 'total_column_snow_water'):
                                    var[t,:,:] = np.nansum(nc.variables[args.var_sname_era[v]][n:n+24,:,:], axis = 0)
                                else:
                                    var[t,:,:] = np.nanmean(nc.variables[args.var_sname_era[v]][n:n+24,:,:], axis = 0)
                            else:
                                if (variable == 'total_precipitation') | (variable == 'total_column_rain_water') | (variable == 'total_column_snow_water'):
                                    var[t,:,:] = np.nansum(nc.variables[args.var_sname_era[v]][n:n+24,0,:,:], axis = 0)
                                else:
                                    var[t,:,:] = np.nanmean(nc.variables[args.var_sname_era[v]][n:n+24,0,:,:], axis = 0)
                            n = n + 24
                
                    # Several variables need unit conversions for consistency with other datasets
                    #if (variable == 'total_precipitation') | (variable == 'evaporation') | (variable == 'potential_evaporation') | (variable == 'runoff'):
                    #    print('Converting units...')
                        # These variables are in units of meters; multiply by the density of water to put them in kg m^-2 (how much mass of water in a given area)
                    #    var = var * 1000
                    
                    # ET and PET in ERA5 are negative; make them positive
                    # if (variable == 'evaporation') | (variable == 'potential_evaporation'):
                    #     print('Making evaporation/potential evaporation positive...')
                    #     var = -1 * var

                    if 'type' in variable:
                        # Round to nearest integer for type of vegetation coverage
                        var = np.round(var)

                    # Compress the data a little
                    var = var.astype(np.float32)
                    
                    # Check the data
                    print('Date start and end points:', dates[0], dates[-1])
                    print('Data shape:', var.shape)
                
                    # Write the reduced data as a netcdf file
                    description = "Daily ERA5 reanalysis data for %s"%variable
                    fn = '%s_%d.nc'%(variable, year) if args.era5_dataset == 'reanalysis-era5-single-levels' else '%s_%d_mb_%d.nc'%(variable, args.pressure_level, year)
                    write_nc(var, lat, lon, dates[::24], filename = fn, var_sname = args.var_sname[v], description = description)
                
                # Delete the extra large file
                print('Removing large datafile...')
                os.remove(filename)
                
    print('Finished')
                    
