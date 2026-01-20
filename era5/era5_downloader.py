import cdsapi

def downloader(variable, year, dataset = 'reanalysis-era5-single-levels', pressure = 500):
    """
    API to download 1 year of hourly ERA5 data from cds.climate.copernicus.eu
    
    NOTE: If the daily statistics dataset is used, there is an assumption that the daily mean is desired
    
    Inputs:
    :param variable: Str. Name of the variable to download
    :param year: Year of the data to be downloaded
    :param dataset: Str. ERA5 dataset to collect data from. Default = 'reanalysis-era5-single-levels'
    :param pressure: Pressure level of the data downloaded (only used when dataset = 'derived-era5-pressure-levels-daily-statistics'). Default = 'reanalysis-era5-single-levels'
    
    Outputs:
    None. Data is downloaded into the current directory. 
    #### NOTE the data files are LARGE and require significant memory and time to download
         1 year of data ~17GB, and takes ~1 hour to download
    """
    # Create the request
    request = {'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': variable,
                'year': str(year),
                'month': [
                    '01', '02', '03',
                    '04', '05', '06',
                    '07', '08', '09',
                    '10', '11', '12',
                ],
                'day': [
                    '01', '02', '03',
                    '04', '05', '06',
                    '07', '08', '09',
                    '10', '11', '12',
                    '13', '14', '15',
                    '16', '17', '18',
                    '19', '20', '21',
                    '22', '23', '24',
                    '25', '26', '27',
                    '28', '29', '30',
                    '31',
                ],
                }
                
    if dataset == 'reanalysis-era5-single-levels':
        # Add the times for single levels
        request.update({
            'time': [
                '00:00', '01:00', '02:00',
                '03:00', '04:00', '05:00',
                '06:00', '07:00', '08:00',
                '09:00', '10:00', '11:00',
                '12:00', '13:00', '14:00',
                '15:00', '16:00', '17:00',
                '18:00', '19:00', '20:00',
                '21:00', '22:00', '23:00',
            ],
            'data_format': 'netcdf',
            'download_format': 'unarchived',
            })
        filename = 'era5_%s_%d.nc'%(variable[0], year)
    
    elif dataset == 'reanalysis-era5-pressure-levels':
        # Add the times and pressure level 
        request.update({
            'time': [
                '00:00', '01:00', '02:00',
                '03:00', '04:00', '05:00',
                '06:00', '07:00', '08:00',
                '09:00', '10:00', '11:00',
                '12:00', '13:00', '14:00',
                '15:00', '16:00', '17:00',
                '18:00', '19:00', '20:00',
                '21:00', '22:00', '23:00',
            ],
            'pressure_level': str(pressure),
            'data_format': 'netcdf',
            'download_format': 'unarchived',
            })
        filename = 'era5_%s_%d.%d_mb.nc'%(variable[0], year, pressure)
        
    elif dataset == 'derived-era5-pressure-levels-daily-statistics':
        # Add the pressure level, time zone, and frequency of the data
        request.update({
            'pressure_level': str(pressure),
            'daily_statistic': 'daily_mean',
            'time_zone': 'utc+00:00',
            'frequency': '1_hourly',
            'data_format': 'netcdf',
            'download_format': 'unarchived',
            })
        filename = 'era5_%s_%d.%d_mb.nc'%(variable[0], year, pressure)
    
    # Create the downloader
    c = cdsapi.Client()
    print(dataset)
    print(request)
    print(filename)

    # Retrieve data using the given information
    c.retrieve(
        dataset,
        request,
        filename)
