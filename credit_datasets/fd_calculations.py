"""Performs calculations for SESR and FDII and 
stores the information among existing variables

Also provides various flash drought calculation
functions
"""

import gc
import numpy as np
from typing import Tuple
from datetime import datetime, timedelta
from scipy import stats
from netCDF4 import Dataset
from argparse import ArgumentParser
from tqdm import tqdm

from inputs_outputs import load_nc

def calculate_climatology(
        dataset = 'era5',
        path_to_data = '../era5' 
        ) -> None:
    '''
    Calculates the climatological mean and standard deviation of ESR from daily evaporation 
    and potential evaporation data data. Climatological data is calculated for all grid 
    points and for all timestamps in the year.
    
    Note this function also makees some hard assumptions on the path to the datasets, and file names

    Inputs:
    :param dataset: Name of the dataset to load evaporation and potential evaporation from (era5 or gldas)
    :param path_to_data: path to the era5 data.
    '''
    
    # Load test dataset (needs leap year)
    filename = '../era5/evaporation/evaporation_2000.nc' if dataset == 'era5' else '../gldas/evaporation/gldas.evaporation.daily.2004.nc'
    sname = 'evap' if dataset == 'era5' else 'e'
    test = {}
    with Dataset(filename, 'r') as nc:
        test['e'] = nc.variables[sname][:]
        test['lat'] = nc.variables['lat'][:]
        test['lon'] = nc.variables['lon'][:]
        test['time'] = nc.variables['date'][:]

    T, I, J = test['e'].shape

    # All years in climatology calculations
    all_years = np.arange(2000, 2025) if dataset == 'era5' else np.arange(2001, 2025)

    # Get datetimes
    dates_year = np.array([datetime.fromisoformat(day) for day in test['time']])

    # Remvoe the unnecessary variable and clear some of the memory
    del test 
    gc.collect()

    # Initialize climatology means and standard deviations + time counts (for mean calculations)
    means = np.zeros((T, I, J), dtype = np.float32)
    stds = np.zeros((T, I, J), dtype = np.float32)

    N = np.zeros((T)) 

    print('Initialized variables, calculation means')

    # Conduct climatology calculations; 
    # load only 1 year of data at a time so that no more than 7.5 GB is loaded into the memory at a time
    for year in all_years:
        print(year)
        # Load the evaporation and potential evaporation data
        if dataset == 'era5':
            filename_e = '%s/evaporation/evaporation_%04d.nc'%(path_to_data, year)
            sname_e = 'e'
            filename_pet = '%s/potential_evaporation/potential_evaporation_%04d.nc'%(path_to_data, year)
            sname_pet = 'pev'
        elif dataset == 'gldas':
            filename_e = '%s/evaporation/gldas.evaporation.daily.%04d.nc'%(path_to_data, year)
            sname_e = 'evap'
            filename_pet = '%s/potential_evaporation/gldas.potential_evaporation.daily.%04d.nc'%(path_to_data, year)
            sname_pet = 'pevap'

        e = load_nc(filename_e, sname_e)
        pet = load_nc(filename_pet, sname_pet)

        # For GLDAS2 data only, convert PET from W m^-2 to kg m^-2 s^-2 by dividing by the latent heat of vaporization
        if np.invert(dataset == 'era5'):
            pet['pevap'] = pet['pevap'] / (2.5e6)

        # Construct ESR for the year
        esr = e[sname_e]/pet[sname_pet]

        # Remove values that exceed a certain limit as they are likely an error
        esr[esr < 0] = np.nan
        esr[esr > 3] = np.nan 

        # Construct datetime information
        dates_current_year = np.array([datetime.fromisoformat(day) for day in e['time']])

        days = np.array([day.day for day in dates_current_year])
        months = np.array([day.month for day in dates_current_year])

        for t, date in enumerate(dates_year):
            # Get the days in the current corresponding to the day in the loop 
            # (may not be the same as t as current year may not be a leap year)
            ind = np.where( (date.day == days) & (date.month == months) )[0]

            # For non-leap years, ind can be empty; skip this day
            if len(ind) < 1:
                continue
            else:
                # Accumulate ESR values
                means[t,:,:] = np.nansum([means[t,:,:], esr[ind[0],:,:]], axis = 0) # np.nansum to account for any NaNs
                N[t] = N[t] + 1

    # At the end, loop again to get to divide the sums by N to get the means
    for t, date in enumerate(dates_year):
        means[t,:,:] = means[t,:,:]/N[t]

    means = means.astype(np.float32)

    print('Means calculated, calculating standard deviations')

    # Loop again to do the standard deviation calculation
    for year in all_years:
        print(year)
        # Load the evaporation and potential evaporation data
        if dataset == 'era5':
            filename_e = '%s/evaporation/evaporation_%04d.nc'%(path_to_data, year)
            sname_e = 'e'
            filename_pet = '%s/potential_evaporation/potential_evaporation_%04d.nc'%(path_to_data, year)
            sname_pet = 'pev'
        elif dataset == 'gldas':
            filename_e = '%s/evaporation/gldas.evaporation.daily.%04d.nc'%(path_to_data, year)
            sname_e = 'evap'
            filename_pet = '%s/potential_evaporation/gldas.potential_evaporation.daily.%04d.nc'%(path_to_data, year)
            sname_pet = 'pevap'

        e = load_nc(filename_e, sname_e)
        pet = load_nc(filename_pet, sname_pet)

        # For GLDAS2 data only, convert PET from W m^-2 to kg m^-2 s^-2 by dividing by the latent heat of vaporization
        if np.invert(dataset == 'era5'):
            pet['pevap'] = pet['pevap'] / (2.5e6)

        # Construct ESR for the year
        esr = e[sname_e]/pet[sname_pet]

        # Remove values that exceed a certain limit as they are likely an error
        esr[esr < 0] = np.nan
        esr[esr > 3] = np.nan 

        # Construct datetime information
        dates_current_year = np.array([datetime.fromisoformat(day) for day in e['time']])
        
        days = np.array([day.day for day in dates_current_year])
        months = np.array([day.month for day in dates_current_year])

        for t, date in enumerate(dates_year):
            # Get the days in the current corresponding to the day in the loop 
            # (may not be the same as t as current year may not be a leap year)
            ind = np.where( (date.day == days) & (date.month == months) )[0]
            
            # For non-leap years, ind can be empty; skip this day
            if len(ind) < 1:
                continue
            else:
                # Accumulate the square error into a large sum
                error = (esr[ind[0],:,:] - means[t,:,:])**2
                stds[t,:,:] = np.nansum([stds[t,:,:], error], axis = 0)

    # One final loop to finish STD calculations
    for t, date in enumerate(dates_year):
        stds[t,:,:] = np.sqrt(stds[t,:,:]/(N[t] - 1))

    stds = stds.astype(np.float32)
    print(np.min(means), np.max(means))
    print(np.min(stds), np.max(stds))

    print('Standard deviations calculated, writing results')

    # Save the results
    with Dataset('%s/esr_climatologies.nc'%(path_to_data), 'w', format = 'NETCDF4') as nc:
        nc.description = 'Daily climatology for ESR using daily %s reanalysis data for ET and PET'%dataset.upper()

        # Create Dimensions
        nc.createDimension('latitude', size = I)
        nc.createDimension('longitude', size = J)
        nc.createDimension('time', size = T)

        # Create the lat, lon, and time information
        if dataset == 'era5':
            lat_shape = ('latitude', 'longitude')
            lon_shape = ('latitude', 'longitude')
        elif dataset == 'gldas':
            lat_shape = ('latitude', )
            lon_shape = ('longitude', )

        nc.createVariable('lat', e['lat'].dtype, lat_shape)
        nc.createVariable('lon', e['lon'].dtype, lon_shape)
        nc.createVariable('date', str, ('time'))

        # Create datetimes as strings to store
        dates_iso = np.array([date.isoformat() for date in dates_year])

        # Assign lat, lon, and time data
        nc.variables['lat'][:] = e['lat'][:]
        nc.variables['lon'][:] = e['lon'][:]
        nc.variables['date'][:] = dates_iso[:]

        # Create and save the means
        nc.createVariable('means', means.dtype, ('time', 'latitude', 'longitude'))
        nc.variables['means'][:] = means[:]

        # Create and save the standard deviations
        nc.createVariable('stds', stds.dtype, ('time', 'latitude', 'longitude'))
        nc.variables['stds'][:] = stds[:]

    print('Finished writing climatology data.')

def calculate_sesr(
        et, 
        pet, 
        year, 
        path_to_data: str = '../era5'
        ) -> np.ndarray:
    '''
    Calculate the standardized evaporative stress ratio (SESR) from ET and PET for 1 year.
    
    Full details on SESR can be found in Christian et al. 2019 (for SESR): https://doi.org/10.1175/JHM-D-18-0198.1.
    
    Inputs:
    :param et: Input evapotranspiration (ET) data (np.ndarray with shape time x lat x lon)
    :param pet: Input potential evapotranspiration (PET) data (np.ndarray with shape time x lat x lon)
                should be in the same units as et
    :param year: The year SESR will be calculated for
    :param path_to_data: Directory path to the ESR climatology files.
        
    Outputs:
    :param sesr: Calculated SESR (np.ndarray with shape time x lat x lon)
    
    '''

    # Get date information
    T, I, J = et.shape
    dates = np.array([datetime(year, 1, 1) + timedelta(days = t) for t in range(T)])

    # Obtain the evaporative stress ratio (ESR); the ratio of ET to PET
    esr = et/pet

    # Remove values that exceed a certain limit as they are likely an error
    esr[esr < 0] = 0
    esr[esr > 3] = 3

    # Load the climatological data for the ESR
    clim = {}
    with Dataset('%s/esr_climatologies.nc'%path_to_data, 'r') as nc:
        clim['means'] = nc.variables['means'][:]
        clim['stds'] = nc.variables['stds'][:]
        clim['time'] = np.array([datetime.fromisoformat(date) for date in nc.variables['date']])

        months = np.array([date.month for date in clim['time']])
        days = np.array([date.day for date in clim['time']])

    # Initialize SESR dataset
    sesr = np.ones((T, I, J)) * np.nan

    # Calculate SESR (standardized ESR) for each respective day in the year
    for t, date in enumerate(dates):
        ind = np.where( (date.month == months) & (date.day == days) )[0]
        
        sesr[t,:,:] = (esr[t,:,:] - clim['means'][ind[0],:,:])/clim['stds'][ind[0],:,:]
            

    # Remove any unrealistic points
    sesr = np.where(sesr < -5, -5, sesr)
    sesr = np.where(sesr > 5, 5, sesr)

    sesr = sesr.astype(np.float32)
    
    print(np.min(sesr), np.max(sesr), np.mean(sesr))
    return sesr

def calculate_fdii(
        smp, 
        apply_runmean = True, 
        use_mask = True, 
        path_to_mask = '../era5'
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Calculate the flash drought intensity index (FDII) from soil moisture percentiles.
    FDII index is on the same time scale as the input data.
    
    Full details on FDII can be found in Otkin et al. 2021: https://doi.org/10.3390/atmos12060741
    
    Inputs:
    :param smp: Input soil moisture percentiles (np.ndarray with shape time x lat x lon)
    :param apply_runmean: Boolean indicating whether to apply a length 5 centered running mean 
                          to the percentiles before FDII calculations (recommended for daily data)
    :param use_mask: Indicates whether to use a land-sea mask to improve speed computation
                     (land-sea mask is used to skip FDII calculations over sea values)
    :param path_to_mask: String path to the mask data if it is used 
                         (note this should be None if a non-ERA5 data is used)

    Outputs:
    :param fdii: The FDII flash drought index (np.ndarray with shape time x lat x lon)
    :param fd_int: The strength of the rapid intensification of the flash drought (np.ndarray with shape time x lat x lon)
    :param dro_sev: Severity of the drought component of the flash drought (np.ndarray with shape time x lat x lon)
    '''

    # Next, FDII can be calculated with the standardized soil moisture, or percentiles.
    # Percentiles are used here for consistancy with Otkin et al. 2021
       
    print('Initializing some variables')
    # Define some base constants
    PER_BASE = 15 # Minimum percentile drop in 4 pentads
    T_BASE   = 4*5
    DRO_BASE = 20 # Percentiles must be below the 20th percentile to be in drought

    T, I, J = smp.shape

    # Apply a 5 day running mean to the data if necessary
    if apply_runmean:
        print('Applying 5 day running mean')
        runmean = 5
        start_ind = int(np.round((runmean - 1)/2))
        end_ind = int(T + runmean - 1 - start_ind)
        for i in tqdm(range(I), desc = 'Applying running mean'):
            for j in range(J):
                # Use convolve in ones to obtain the running mean; the start_ind:end_ind determines the centered mean
                smp[:,i,j] = np.convolve(smp[:,i,j], np.ones((runmean))/runmean)[start_ind:end_ind]
    
    # Load the mask? (for ERA5 only)
    if use_mask & (path_to_mask is None):
        with Dataset('%s/land.nc'%path_to_mask, 'r') as nc:
            mask = nc.variables['lsm'][0,:,:]
    
    # Output percentile information as a check
    print(np.nanmin(smp), np.nanmax(smp))
    print(np.nanmean(smp))
    
    print('Calculating rapid intensification of flash drought')
    # Determine the rapid intensification based on percentile changes 
    # based on equation 1 in Otkin et al. 2021 (and detailed in section 2.2 of the same paper)
    fd_int = np.zeros((T, I, J))

    # Determine the intensification index
    for i in tqdm(range(I), desc = 'Calculating FD_INT'):
        for j in range(J):
            if use_mask:
                # Ignore sea values
                if (path_to_mask is None) & (np.nansum(np.isnan(smp[:,i,j])) == smp.shape[0]): # For GLDAS2 only
                    continue
                elif mask[i,j] == 0: # ERA5 only
                    continue
            
            for t in range(T-10): # Note the last two pentads are excluded as there is no change to examine
            
                obs = np.zeros((9*5)) # Note, the method detailed in Otkin et al. 2021 involves looking 
                                      # ahead 2 to 10 pentads (9 entries total)
                for nday in np.arange(2*5, 10*5+5, 1):
                    nday = int(nday)
                    if (t+nday) >= T: # If t + npend is in the future (beyond the dataset), break the loop 
                        break         # and use 0s for obs instead. 
                    else:
                        obs[nday-10] = (smp[t+nday,i,j] - smp[t,i,j])/nday # Note nday is the number of days 
                                                                           # the system is corrently looking ahead to.
                if np.nansum(np.isnan(obs)) == obs.size:
                    print('All NaNs: ', t, nday, i, j)

                # If the maximum change in percentiles is less than the base change requirement (15 percentiles in 4 pentads), 
                # set FD_INT to 0. Otherwise, determine FD_INT according to eq. 1 in Otkin et al. 2021
                if np.nanmax(obs) < (PER_BASE/T_BASE):
                    fd_int[t,i,j] = 0
                else:
                    fd_int[t,i,j] = ((PER_BASE/T_BASE)**(-1)) * np.nanmax(obs)

    # Print the flash drought intensification component to check it   
    print(np.min(fd_int), np.max(fd_int), np.mean(fd_int))
    
    
    print('Calculating drought severity')
    # Next determine the drought severity component using equation 2 in Otkin et al. 2021 
    # (and detailed in section 2.2 of the same paper)
    dro_sev = np.zeros((T, I, J)) # Initialize to 0, the the first entry will stay 0 since 
                                  # there is no rapid intensification before it

    for i in tqdm(range(I), desc = 'Calculating DRO_SEV'):
        for j in range(J):
            if use_mask:
                # Ignore sea values
                if (path_to_mask is None) & (np.nansum(np.isnan(smp[:,i,j])) == smp.shape[0]): # For GLDAS2 only
                    continue
                elif mask[i,j] == 0: # ERA5 only
                    continue
            
            for t in range(1, T-5):
                if (fd_int[t,i,j] > 0):
                    
                    dro_sum = 0
                    # In Otkin et al. 2021, the DRO_SEV can look up to 18 pentads (90 days) in the future for its calculation
                    for nday in np.arange(0, 18*5+5, 1): 
                        
                        if (t+nday) >= T:      # For simplicity, set DRO_SEV to 0 when near the end of the dataset
                            dro_sev[t,i,j] = 0
                            break
                        else:
                            dro_sum = dro_sum + (DRO_BASE - smp[t+nday,i,j])
                            
                            if smp[t+nday,i,j] > DRO_BASE: # Terminate the summation and calculate DRO_SEV if 
                                                           # SM is no longer below the base percentile for drought
                                if nday < 4*5:
                                    # DRO_SEV is set to 0 if drought was not consistent for at least 4 pentads after 
                                    # rapid intensificaiton (i.e., little to no impact)
                                    dro_sev[t,i,j] = 0
                                    break
                                else:
                                    dro_sev[t,i,j] = dro_sum/nday # Terminate the loop and determine the drought 
                                    break                         # severity if the drought condition is broken
                                
                            # Calculate the drought severity of the loop goes out 90 days, but the drought does not end
                            elif (nday >= 18*5): 
                                dro_sev[t,i,j] = dro_sum/nday
                                break
                            else:
                                pass
                
                # In continuing consistency with Otkin et al. 2021, if the pentad does not immediately 
                # follow rapid intensification, drought is set 0
                else:
                    dro_sev[t,i,j] = 0
                    continue
    
    # Print the drought severity component to check it
    print(np.min(dro_sev), np.max(dro_sev), np.mean(dro_sev))
    
    print('Calculating FDII')
    
    # Finally, FDII is the product of the components
    fdii = fd_int * dro_sev
    
    # Print the FDII values
    print(np.min(fdii), np.max(fdii), np.mean(fdii))

    # Remove values less than 0
    fd_int[fd_int <= 0] = 0
    dro_sev[dro_sev <= 0] = 0
    fdii[fdii <= 0] = 0

    print('Done')
    
    return fdii, fd_int, dro_sev


def calculate_sm_percentiles(
        year, 
        level = 1, 
        dataset = 'era5', 
        path_to_data = '../era5'
        ) -> None:
    '''
    Calculate the soil moisture percentiles for a singular year and save the results to a .nc file

    Note this method is NOT space efficient. It loads in the full soil moisture dataset (35 GB for 
    quarter degree resolution) to produce timely computations.

    Note this function also makees some hard assumptions on the path to the datasets, and file names

    Inputs:
    :param year: The year the soil moisture percentiles are being calculated for
    :param level: The soil moisture level being loaded (e.g., 1 = 0 - 7/10 cm, 2 = 7 - 28/10 - 40 cm, etc.)
    :param dataset: Name of the dataset the soil moisture stems from (era5 or gldas)
    :param path_to_data: Directory path to the soil moisture dataset
    '''
    
    # Short name, (.nc key) of the soil moisture and percentile variables
    sm_sname = 'soilm' if dataset == 'gldas' else 'swvl%d'%level
    smp_sname = 'smp%d'%level

    # For GLDAS2 only:
    if dataset == 'gldas':
        if level == 1:
            var_name = 'soil_moisture_0-10cm'
        elif level == 2:
            var_name = 'soil_moisture_10-40cm'
        elif level == 3:
            var_name = 'soil_moisture_40-100cm'
        elif level == 4:
            var_name = 'soil_moisture_100-200cm'

    # Note a complete time series is needed to do the percentile calculations; first part is getting information for that time series

    # Import a test dataset (without leap years) to get the time series size
    if dataset == 'gldas':
        filename = '%s/%s/gldas.%s.daily.%04d.nc'%(path_to_data, var_name, var_name, year)
    elif dataset == 'era5':
        filename = '%s/liquid_vsm/volumetric_soil_water_layer_%d_%04d.nc'%(path_to_data, level, year)

    test = {}
    with Dataset(filename, 'r') as nc:
        test[sm_sname] = nc.variables[sm_sname][:]
        test['lat'] = nc.variables['lat'][:]
        test['lon'] = nc.variables['lon'][:]
        test['time'] = nc.variables['date'][:]

    print('Loaded test set')
    
    # Obtain shape information
    T, I, J= test[sm_sname].shape

    # Get the information for 1 year in a datetime array
    days_per_year = 365
    date_intial = datetime(year, 1, 1)    
    dates_year = np.array([date_intial + timedelta(days = t) for t in range(T)])
    
    # All years in the time series
    # Note GLDAS starts at 2001
    all_years = np.arange(2001, 2025) if dataset == 'gldas' else np.arange(2000, 2025)
    N_leap_days = np.sum((all_years % 4) == 0) # Note datetimes follow the scheme of a leap day every 4 years, 
    T_full = (days_per_year * all_years.size) + N_leap_days                # even if it isn't exactly correct

    # Calculate all dates in the full time series
    date_initial = datetime(all_years[0], 1, 1)
    dates = np.array([date_initial + timedelta(days = t) for t in range(T_full)])
    years = np.array([date.year for date in dates])
    months = np.array([date.month for date in dates])
    days = np.array([date.day for date in dates])

    # For ERA5 only
    # Load a land-sea mask; using this to skip sea points can save time
    if dataset == 'era5':
        with Dataset('%s/land.nc'%path_to_data ,'r') as nc:
            mask = nc.variables['lsm'][0,:,:]
        
    # Initialize percentile dataset
    smp = np.zeros((T, I, J))
    print('Initialized percentile dataset for level %d and %d'%(level, year))

    print('Loading SM dataset')
    sm = []
    # Load the full soil moisture dataset
    for y in all_years:
        # Determine the soil moisture filename
        if dataset == 'gldas':
            filename = '%s/%s/gldas.%s.daily.%04d.nc'%(path_to_data, var_name, var_name, y)
        elif dataset == 'era5':
            filename = '%s/liquid_vsm/volumetric_soil_water_layer_%d_%04d.nc'%(path_to_data, level, y)

        # Load the data and add to the list
        with Dataset(filename, 'r') as nc:
            sm.append(nc.variables[sm_sname][:].astype(np.float32))

    # sm = np.concatenate(sm, axis = 0)
    print('Loaded dataset')

    # n = 0
    for i in tqdm(range(I), desc = "Calculating Percentiles"):
        for j in range(J):
            # Skip sea values
            if dataset == 'gldas':
                if np.nansum(np.isnan(sm[0][:,i,j])) == sm[0].shape[0]: # For GLDAS2 only
                    continue
            elif dataset == 'era5':
                if mask[i,j] == 0: # ERA5 only
                    continue

            # Initialize the full time series for a grid cell
            sm_time_series = []

            # Load in all times for 1 grid point at a time
            # Note this method is more space efficient (concatenating 
            # the SM dataset earlier would temporarly double RAM usage)
            for y, _ in enumerate(all_years):
                sm_time_series.append(sm[y][:,i,j])
            
            sm_time_series = np.concatenate(sm_time_series)
            
            # Calculate the SM percentiles for all points in the year for the grid cell
            for t, date in enumerate(dates_year):
                # Obtain all indices for the current day of the year
                ind = np.where((date.day == days) & (date.month == months))[0] 

                # Determine what index to use for the current percentile calculation
                current_day = np.where((date.day == days) & (date.month == months) & (date.year == years))[0] 

                # Calculate the SM percentile based on the current day of the year
                smp[t,i,j] = stats.percentileofscore(sm_time_series[ind], sm_time_series[current_day[0]])

    smp = smp.astype(np.float32)

    # Get the latitude and longitude information
    lat = test['lat'].astype(np.float32); lon = test['lon'].astype(np.float32)

    # Save the results once finished with the calculations
    print('Finished soil moisture percentile calculations for %d'%year)
    with Dataset('%s/soil_moisture_percentiles_%d_%04d.nc'%(path_to_data, level, year), 'w', format = 'NETCDF4') as nc:
        nc.description = 'Daily %s reanalysis data for percentiles of soil moisture layer %d'%(dataset.upper(), level)

        # Create Dimensions
        nc.createDimension('latitude', size = I)
        nc.createDimension('longitude', size = J)
        nc.createDimension('time', size = T)

        # Create the lat, lon, and time information
        if dataset == 'gldas':
            lat_shape = ('latitude',)
            lon_shape = ('longitude',)
        elif dataset == 'era5':
            lat_shape = ('latitude', 'longitude')
            lon_shape = ('latitude', 'longitude')

        nc.createVariable('lat', lat.dtype, lat_shape)
        nc.createVariable('lon', lon.dtype, lon_shape)
        nc.createVariable('date', str, ('time'))

        # Add the latitude, longitude, and time information
        nc.variables['lat'][:] = lat[:]
        nc.variables['lon'][:] = lon[:]
        nc.variables['date'][:] = test['time'][:]

        # Create and save the percentiles to the nc file
        nc.createVariable(smp_sname, smp.dtype, ('time', 'latitude', 'longitude'))
        nc.variables[smp_sname][:] = smp[:]

    print('Finished writing data. Done with %d'%year)

def create_aridity_mask(
        dataset: str = 'era5', 
        path_to_data: str = '../era5'
        ) -> None:
    '''
    Create an aridity mask based on aridity index and potential evaporation 
    (aridity mask is similar in concept to a land-sea mask, but covers highly arid locations)

    NOTE: This function makes some hard assumptions about the paths and filenames for variables
    '''

    # All the years used for the aridity index calculation
    # Note GLDAS2 data starts at 2001
    years = np.arange(2000, 2025) if dataset == 'era5' else np.arange(2001, 2025)

    # Load a test dataset to get the data size
    if dataset == 'era5':
        filename = '%s/potential_evaporation/potential_evaporation_2000.nc'%path_to_data
        sname = 'pev'
    elif dataset == 'gldas':
        filename = '%s/potential_evaporation/gldas.potential_evaporation.daily.2004.nc'%path_to_data
        sname = 'pevap'

    test = load_nc(filename, sname) 
    T, I, J = test[sname].shape

    # Initialize datasets
    p_annual = np.ones((years.size, I, J), dtype = np.float32) * np.nan
    pet_annual = np.ones((years.size, I, J), dtype = np.float32) * np.nan

    # Load in and sum over all precipitation and PET values to get annual accumulations
    for t, year in enumerate(years): # Note here that P and PET should both be in units of m; also this gives total annual accumulation
        # Determine filenames and .nc keys
        if dataset == 'era5':
            filename_p = '%s/precipitation/total_precipitation_%04d.nc'%(path_to_data, year)
            filename_pet = '%s/potential_evaporation/potential_evaporation_%04d.nc'%(path_to_data, year)
            sname_p = 'tp'
            sname_pet = 'pev'
        elif dataset == 'gldas':
            filename_p = '%s/precipitation/gldas.precipitation.daily.%04d.nc'%(path_to_data, year)
            filename_pet = '%s/potential_evaporation/gldas.potential_evaporation.daily.%04d.nc'%(path_to_data, year)
            sname_p = 'precip'
            sname_pet = 'pevap'
        
        # Load the precipitation and PET data
        p = load_nc(filename_p, sname_p)
        pet = load_nc(filename_pet, sname_pet)

        # Scale reduction if desired
        #p_reduced, _, _ = reduce_spatial_scale(p, 'tp') 
        #pet_reduced, _, _ = reduce_spatial_scale(pet, 'pev')

        # For GLDAS2 data only, convert PET from W m^-2 to kg m^-2 s^-1
        # (same units as precipitation in GLDAS2) 
        if dataset == 'gldas':
            # Divide by the latent heat of vaporization
            pet['pevap'] = pet['pevap'] / (2.5e6)

        # Accumulate P and PET over the whole year
        p_annual[t,:,:] = np.nansum(p[sname_p], axis = 0)
        pet_annual[t,:,:] = np.nansum(pet[sname_pet], axis = 0)

        # For GLDAS2 only, convert PET from kg m^-2 to m 
        if dataset == 'gldas':
            # Divide by the density of water to convert units of m (used in daily PET calculation)
            pet_annual[t,:,:] = pet_annual[t,:,:] / 1000


    # Aridity index is the mean annual precipitation accumulation divided by mean annual PET accumulation
    # For the purposes of the ratio, PET is assumed positive in the aridity index calculations
    arid_index = np.nanmean(p_annual, axis = 0)/np.abs(np.nanmean(pet_annual, axis = 0)) 

    # When creating the aridity mask, the daily PET is also needed
    # nanmean delivers the annual mean PET accumulation; division by 365 approximates average daily accumulation
    pet_daily = np.abs(np.nanmean(pet_annual, axis = 0)) / 365 

    # Aridity mask will be based on PET from across the entire year
    # Note the requirement is mean daily PET < 1 mm/day = 0.001 m/day and aridity index < 0.2
    ai_mask = np.where( (arid_index < 0.2) | (pet_daily < 0.001), 0, 1) 

    # ERA5 Only: Note there is a strange behavoir in ERA5 that tries to mask out the 
    # Congo Basin, despite it not being arid. Correct this error
    if dataset == 'era5':
        # Lat/lon box around the Congo Basin
        condition = (np.abs(test['lat']) < 10) & ((test['lon'] >= 11.5) & (test['lon'] <= 30))

        # Mask the area in that lat/lon box
        ai_mask = np.where(condition, 1, ai_mask)

    # Put the aridity mask to 3D in the same format as the ERA5 land-sea mask
    aridity_mask = np.zeros((1, I, J))
    aridity_mask[0,:,:] = ai_mask.astype(np.int16)
  
    # Write the aridity mask
    with Dataset('%s/aridity_mask.nc'%path_to_data, 'w', format = 'NETCDF4') as nc:
        nc.description = 'Global aridity index based on daily %s reanalysis precipitation and potential evaporation'%dataset.upper()

        # Create the dimension information (same format as the land-sea mask)
        nc.createDimension('latitude', size = I)
        nc.createDimension('longitude', size = J)
        nc.createDimension('time', size = 1)

        # Create the latitude and longitude information
        if dataset == 'gldas':
            lat_shape = ('latitude',)
            lon_shape = ('longitude',)
        elif dataset == 'era5':
            lat_shape = ('latitude', 'longitude')
            lon_shape = ('latitude', 'longitude')
        nc.createVariable('latitude', test['lat'].dtype, lat_shape)
        nc.createVariable('longitude', test['lon'].dtype, lon_shape)
  
        # Add the latitude and longitude information
        nc.variables['latitude'][:] = test['lat'][:,0]
        nc.variables['longitude'][:] = test['lon'][0,:]

        # Create and store the aridity mask
        nc.createVariable('aim', aridity_mask.dtype, ('time', 'latitude', 'longitude'))
        nc.variables['aim'][:] = aridity_mask[:]

if __name__ == '__main__':
    # Define the parser for command line
    description = 'Create aridity variables and indices related to flash drought for future ML use'
    parser = ArgumentParser(description = description)
    parser.add_argument('--calculate_aridity_mask', action = 'store_true', help = 'Create an mask based on the aridity index and save as an nc file')
    parser.add_argument('--calculate_climatology', action = 'store_true', help = 'Calculate the ESR climatologies and save as an nc file')
    parser.add_argument('--calculate_percentiles', action = 'store_true', help = 'Calculate 1 year of soil moisture percentiles based on year and level')
    parser.add_argument('--calculate_indices', action = 'store_true', help = 'Calculate 1 year SESR and FDII flash drought indices based on year')

    parser.add_argument('--year', type = int, default = 0, help = 'Year to perform percentile/index calculations (note the actual year used is year + 2000)')
    parser.add_argument('--dataset', type = str, default = 'era5', help = 'Dataset to use for FD calculations (era5 or gldas)')
    parser.add_argument('--level', type = int, default = 1, 
                        help = 'Soil moisture level used for percentile and FDII caluclations (must be 1 - 4); 1 = 0 - 7/10 cm depth, 2 = 7 - 28/10 - 40 cm depth, etc.')

    # Parse command line arguments
    args = parser.parse_args()

    # Determine the year
    year = args.year + 2000

    # Determine the path to the dataset
    path_to_data = '../era5' if args.dataset == 'era5' else '../gldas'

    # Calculate aridity mask?
    if args.calculate_aridity_mask:
        create_aridity_mask(dataset = args.dataset, path_to_data = path_to_data)

    if args.calculate_climatology:
        calculate_climatology(dataset = args.dataset, path_to_data = path_to_data)

    # Calculate percentiles?
    if args.calculate_percentiles:
        calculate_sm_percentiles(year, 
                                 level = args.level, 
                                 dataset = args.dataset, 
                                 path_to_data = path_to_data)

    # Calculate the flash drought indices?
    if args.calculate_indices:
        # Load ET and PET for SESR
        print('Loading ET and PET for SESR calculations')

        if args.dataset == 'era5':
            filename_et = '../era5/evaporation/evaporation_%04d.nc'%(year)
            filename_pet = '../era5/potential_evaporation/potential_evaporation_%04d.nc'%(year)
            sname_et = 'e'
            sname_pet = 'pev'
        elif args.dataset == 'gldas':
            filename_et = '../gldas/evaporation/gldas.evaporation.daily.%04d.nc'%(year)
            filename_pet = '../gldas/potential_evaporation/gldas.potential_evaporation.daily.%04d.nc'%(year)
            sname_et = 'evap'
            sname_pet = 'pevap'

        et = load_nc(filename_et, sname_et)
        pet = load_nc(filename_pet, sname_pet)

        # For GLDAS2 data only, convert PET from W m^-2 to kg m^-2 s^-2 
        if args.dataset == 'gldas':
            # Divide by the latent heat of vaporization
            pet['pevap'] = pet['pevap'] / (2.5e6)
            # Also remove NaNs to avoid skewing test calculations
            # et['evap'][et['evap'] <= -900] = np.nan
            # pet['pevap'][pet['pevap'] <= -900] = np.nan

        # Calculate SESR
        print('Calculating SESR')
        sesr = calculate_sesr(et[sname_et], pet[sname_pet], year, path_to_data = path_to_data)

        # Fille in NaNs
        sesr[np.isnan(sesr)] = -9999

        sesr = sesr.astype(np.float32)
        print(sesr.shape, et['lat'].shape)

        # Write SESR results to nc file
        print('Saving SESR results')
        with Dataset('%s/sesr_%04d.nc'%(path_to_data, year), 'w', format = 'NETCDF4') as nc:
            nc.description = 'Daily %s reanalysis data for SESR, calculated from evaporation and potential evaporaiton'%args.dataset.upper()

            # Create Dimensions
            T, I, J = sesr.shape
            nc.createDimension('latitude', size = I)
            nc.createDimension('longitude', size = J)
            nc.createDimension('time', size = T)

            # Create the lat, lon, and time information
            if args.dataset == 'gldas':
                lat_shape = ('latitude',)
                lon_shape = ('longitude',)
            elif args.dataset == 'era5':
                lat_shape = ('latitude', 'longitude')
                lon_shape = ('latitude', 'longitude')

            nc.createVariable('lat', et['lat'].dtype, lat_shape)
            nc.createVariable('lon', et['lon'].dtype, lon_shape)
            nc.createVariable('date', str, ('time',))

            # Add the latitude, longitude, and time information
            nc.variables['lat'][:] = et['lat'][:]
            nc.variables['lon'][:] = et['lon'][:]
            nc.variables['date'][:] = et['time'][:]

            # Create and save the SESR values
            nc.createVariable('sesr', sesr.dtype, ('time', 'latitude', 'longitude'))
            nc.variables['sesr'][:] = sesr[:]

        # Remove large variables to free up space
        del et, pet, sesr
        gc.collect()

        # Load SM percentiles for FDII calculations
        print('Loading soil moisture percentiles')
        smp = load_nc('%s/soil_moisture_percentiles/soil_moisture_percentiles_%d_%04d.nc'%(path_to_data, args.level, year), 'smp%d'%args.level)

        # Calculate FDII
        print('Calculating FDII for layer %d'%args.level)
        fdii, fd_int, dro_sev = calculate_fdii(smp['smp%d'%args.level], 
                                               year, 
                                               apply_runmean = True, 
                                               use_mask = True, 
                                               path_to_data = path_to_data if args.dataset == 'era5' else None) # GLDAS mask isn't present

        fdii = fdii.astype(np.float32); fd_int = fd_int.astype(np.float32); dro_sev = dro_sev.astype(np.float32)

        # Write the FDII results to a nc file
        print('Saving FDII results')
        with Dataset('%s/fdii_%d_%04d.nc'%(path_to_data, args.level, year), 'w', format = 'NETCDF4') as nc:
            nc.description = 'Daily %s reanalysis data for FDII and associated components, calculated from soil moisture percentiles for layer %d'%(args.dataset, args.level)

            # Create Dimensions
            T, I, J = fdii.shape
            nc.createDimension('latitude', size = I)
            nc.createDimension('longitude', size = J)
            nc.createDimension('time', size = T)

            # Create the lat, lon, and time information
            if args.dataset == 'gldas':
                lat_shape = ('latitude',)
                lon_shape = ('longitude',)
            elif args.dataset == 'era5':
                lat_shape = ('latitude', 'longitude')
                lon_shape = ('latitude', 'longitude')

            nc.createVariable('lat', et['lat'].dtype, lat_shape)
            nc.createVariable('lon', et['lon'].dtype, lon_shape)
            nc.createVariable('date', str, ('time', ))

            # Store the latitude, longitude, and time information
            nc.variables['lat'][:] = smp['lat'][:]
            nc.variables['lon'][:] = smp['lon'][:]
            nc.variables['date'][:] = smp['time'][:]

            # Create and save the FDII information and its components
            nc.createVariable('fdii%d'%args.level, fdii.dtype, ('time', 'latitude', 'longitude'))
            nc.variables['fdii%d'%args.level][:] = fdii[:]

            nc.createVariable('fd_int%d'%args.level, fd_int.dtype, ('time', 'latitude', 'longitude'))
            nc.variables['fd_int%d'%args.level][:] = fd_int[:]

            nc.createVariable('dro_sev%d'%args.level, dro_sev.dtype, ('time', 'latitude', 'longitude'))
            nc.variables['dro_sev%d'%args.level][:] = dro_sev[:]

