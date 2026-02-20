"""Provides a set of functions designed to facilitate
the calculation of the Standardized Evaporative Stress
Ratio (SESR) and Flash Drought Intensity Index (FDII)
for flash drought (FD) monitoring and prediction
"""

import gc
from typing import Tuple
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
from netCDF4 import Dataset
from argparse import ArgumentParser
from tqdm import tqdm

# from inputs_outputs import load_nc

def calculate_climatology(
        e, 
        pet, 
        dates_all, 
        days_per_year: int = 366
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Calculates the climatological mean and standard deviation of ESR from daily ERA5 data.
    Climatological data is calculated for all grid points and for all timestamps in the year.

    Inputs:
    :param e: Evaporation dataset (np.ndarray, with shape time x lat x lon)
    :param pet: Potential evaporation dataset (np.ndarray with shape time x lat x lon)
    :param dates_all: Datetimes labels for each time step in e and pet (np.ndarray with shape time)
    :param days_per_year: Total number of days in one year of data (use 366 if using daily data to include leap day)

    Outputs:
    :param means: Mean of ESR for each grid and date in year (np.ndarray of shape time_for_one_year x lat x lon)
    :param stds: Standard deviations for each grid and date in year (np.ndarray of shape time_for_one_year x lat x lon)
    :param one_year: Datetime labels for each time step in means and stds (np.ndarray of shape time_for_one_year)
    '''
    

    T, I, J = e.shape
    T = days_per_year # Numbers of days in a year

    # All years in climatology calculations
    all_years = np.unique([date.year for date in dates_all])
    years = np.array([date.year for date in dates_all])

    days = np.array([day.day for day in dates_all])
    months = np.array([day.month for day in dates_all])

    # Get datetimes for one year (includes leap day)
    dates_year = np.array([datetime(2012,1,1) + timedelta(days = day) for day in range(days_per_year)])

    # Initialize climatology means and standard deviations + counts
    means = np.zeros((T, I, J), dtype = np.float32)
    stds = np.zeros((T, I, J), dtype = np.float32)

    N = np.zeros((T))

    # Construct ESR
    esr = e/pet

    # Remove values that exceed a certain limit as they are likely an error
    esr[esr < 0] = np.nan
    esr[esr > 3] = np.nan 
    # print(np.nansum(np.isnan(esr)))

    print('Initialized variables, calculation means')

    # Conduct climatology calculations
    for t, date in enumerate(dates_year):
        # Get all days in the current date in the loop
        ind = np.where( (date.day == days) & (date.month == months) )[0]

        # Sum over all all ESR in a given day
        tmp_sum = np.nansum(esr[ind,:,:], axis = 0)
        means[t,:,:] = np.nansum([means[t,:,:], tmp_sum], axis = 0) # np.nansum to account for any NaNs
        N[t] = N[t] + len(ind)

    # At the end, loop again to get to divide the sums by N to get the means
    for t, date in enumerate(dates_year):
        means[t,:,:] = means[t,:,:]/N[t]

    means = means.astype(np.float32)

    print('Means calculated, calculating standard deviations')
    
    # Loop over each day in the year for standard deviation (requires means)
    for t, date in enumerate(dates_year):
        # Get all days in the current date in the loop
        ind = np.where( (date.day == days) & (date.month == months) )[0]
        
        # Sum over the squared errors in a given date
        error = np.nansum((esr[ind,:,:] - means[t,:,:])**2, axis = 0)
        stds[t,:,:] = np.nansum([stds[t,:,:], error], axis = 0)

    # One final loop to finish standard deviation calculations
    for t, date in enumerate(dates_year):
        stds[t,:,:] = np.sqrt(stds[t,:,:]/(N[t] - 1))

    stds = stds.astype(np.float32)
    print(np.min(means), np.max(means))
    print(np.min(stds), np.max(stds))

    print('Standard deviations calculated')

    # Create datetime labels for each timestep in the means and standard deviations
    ind = np.where(years == 2012)[0]
    one_year = dates_all[ind]

    return means, stds, one_year

def calculate_sesr(
        et, 
        pet, 
        dates, 
        means, 
        stds, 
        one_year
        ) -> np.ndarray:
    '''
    Calculate the standardized evaporative stress ratio (SESR) from ET and PET.
    
    Full details on SESR can be found in Christian et al. 2019 (for SESR): https://doi.org/10.1175/JHM-D-18-0198.1.
    
    Inputs:
    :param et: Evapotranspiration (ET) dataset (np.ndarray of shape time x lat x lon)
    :param pet: Potential evapotranspiration (PET) dataset (np.ndarray of shape time x lat x lon)
    :param dates: Datetime labels for each timestep in et and pet (np.ndarray of shape time)
    :param means: Mean of ESR for each grid point and date in the year (np.ndarray of shape time_for_one_year x lat x lon)
    :param stds: Standard deviation of ESR for each grid point and date in the year (np.ndarray of shape time_for_one_year x lat x lon)
    :param one_year: Datetime labels for each time step in means and stds (np.ndarray of shape time_for_one_year)
        
    Outputs:
    :param sesr: Calculate SESR (np.ndarray with shape time x lat x lon)
    '''

    # Get shape information
    T, I, J = et.shape
    # dates = np.array([datetime(year, 1, 1) + timedelta(days = t) for t in range(T)])


    # Obtain the evaporative stress ratio (ESR); the ratio of ET to PET
    esr = et/pet

    # Remove values that exceed a certain limit as they are likely an error
    esr[esr < 0] = np.nan
    esr[esr > 3] = np.nan
    # print(np.nansum(np.isnan(esr)))

    # Collect date information
    months = np.array([date.month for date in one_year])
    days = np.array([date.day for date in one_year])

    # Initialize SESR
    sesr = np.ones((T, I, J)) * np.nan

    for t, date in enumerate(dates):
        # Find the date index for the one year range
        ind = np.where( (date.month == months) & (date.day == days) )[0]
        
        # Standardize the ESR to get SESR
        sesr[t,:,:] = (esr[t,:,:] - means[ind[0],:,:])/stds[ind[0],:,:]
            

    # Remove any unrealistic points
    sesr = np.where(sesr < -5, -5, sesr)
    sesr = np.where(sesr > 5, 5, sesr)

    sesr = sesr.astype(np.float32)
    
    print(np.nanmin(sesr), np.nanmax(sesr), np.nanmean(sesr))
    # print(np.sum(sesr <= -4.5), np.nansum(sesr >= 4.5))
    return sesr

def calculate_fdii(
        smp, 
        dates, 
        apply_runmean = True, 
        mask = None
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Calculate the flash drought intensity index (FDII) from a soil moisture percentiles.
    FDII is on the same time scale as the input data.
    
    Full details on FDII can be found in Otkin et al. 2021: https://doi.org/10.3390/atmos12060741

    Note FDII can be calculated with the standardized soil moisture, or percentiles.
    Percentiles are used here for consistancy with Otkin et al. 2021
    
    Inputs:
    :param smp: Soil moisture percentile dataset (np.ndarray with shape time x lat x lon)
    :param year: Datetime labels for each timestep in smp (np.ndarray of shape time)
    :param apply_runmean: Apply a centered running mean (length 5) to the percentiles before FDII calculations (recommended for daily data)
    :param use_mask: Indicates whether to use a land-sea mask to improve computation speed
    :param mask: Land-sea mask with values 1 for land and 0 for sea (np.ndarray with shape lat x lon)

    Outputs:
    :param fdii: FDII drought index (np.ndarray with shape time x lat x lon)
    :param fd_int: The strength of the rapid intensification of the flash drought (np.ndarray with shape time x lat x lon)
    :param dro_sev: Severity of the drought component of the flash drought (np.ndarray with shape time x lat x lon)
    '''
       
    print('Initializing some variables')
    # Define some base constants
    PER_BASE = 15 # Minimum percentile drop for FD is 15 percentiles in 4 pentads
    T_BASE   = 4*5
    DRO_BASE = 20 # Percentiles must be below the 20th percentile to be in drought
    
    T, I, J = smp.shape

    # Make the years, months, and/or days variables
    years = np.array([date.year for date in dates])
    months = np.array([date.month for date in dates])
    days = np.array([date.day for date in dates])

    # Apply a 5 day running mean requested by the user
    if apply_runmean:
        print('Applying 5 day running mean')
        runmean = 5

        # Determine the appropriate start and end index for a centered running mean 
        start_ind = int(np.round((runmean - 1)/2))
        end_ind = int(T + runmean - 1 - start_ind)

        # Apply running mean for each grid point
        for i in tqdm(range(I), desc = 'Applying running mean'):
            for j in range(J):
                smp[:,i,j] = np.convolve(smp[:,i,j], np.ones((runmean))/runmean)[start_ind:end_ind]
    
    
    print(np.nanmin(smp), np.nanmax(smp))
    print(np.nanmean(smp))
    
    print('Calculating rapid intensification of flash drought')
    # Determine the rapid intensification based on percentile changes 
    # based on equation 1 in Otkin et al. 2021 (and detailed in section 2.2 of the same paper)
    fd_int = np.zeros((T, I, J))

    # Determine the intensification index
    # Note many time related values are multiplied by 5 to correspond to daily data instead of pentad
    for i in tqdm(range(I), desc = 'Calculating FD_INT'):
        for j in range(J):
            # Ignore sea points
            if mask[i,j] == 0: # ERA5 only
                continue
        
            for t in range(T-10): # Note the last two pentads are excluded as there is not enough time for significant SM drop
            
                obs = np.zeros((9*5)) # Note, the method detailed in Otkin et al. 2021 involves looking ahead 2 to 10 pentads (9 entries total)
                for nday in np.arange(2*5, 10*5+5, 1):
                    nday = int(nday)
                    if (t+nday) >= T: # If t + npend is in the future (beyond the dataset), break the loop and use 0s for obs instead
                        break         # This should only effect results in one November and December
                    else:
                        obs[nday-10] = (smp[t+nday,i,j] - smp[t,i,j])/nday # Note npend is the number of pentads the system is currently looking ahead to.

                # If the maximum change in percentiles is less than the base change requirement (15 percentiles in 4 pentads), set FD_INT to 0.
                #  Otherwise, determine FD_INT according to eq. 1 in Otkin et al. 2021
                if np.nanmax(obs) < (PER_BASE/T_BASE):
                    fd_int[t,i,j] = 0
                else:
                    fd_int[t,i,j] = ((PER_BASE/T_BASE)**(-1)) * np.nanmax(obs)
                
    print(np.min(fd_int), np.max(fd_int), np.mean(fd_int))
    
    
    print('Calculating drought severity')
    # Next determine the drought severity component using equation 2 in Otkin et al. 2021 
    # (and detailed in section 2.2 of the same paper)
    dro_sev = np.zeros((T, I, J)) # Initialize the first entry to 0, since there is no rapid intensification before it

    for i in tqdm(range(I), desc = 'Calculating DRO_SEV'):
        for j in range(J):
            
            # Ignore sea values
            if mask[i,j] == 0: # ERA5 only
                continue
            
            for t in range(1, T-5):
                if (fd_int[t,i,j] > 0):
                    
                    dro_sum = 0
                    for nday in np.arange(0, 18*5+5, 1): # In Otkin et al. 2021, the DRO_SEV can look up to 18 pentads (90 days) in the future for its calculation
                        
                        if (t+nday) >= T:      # For simplicity, set DRO_SEV to 0 when near the end of the dataset 
                            dro_sev[t,i,j] = 0 # (this should only impact results near the end of the dataset)
                            break
                        else:
                            dro_sum = dro_sum + (DRO_BASE - smp[t+nday,i,j])
                            
                            if smp[t+nday,i,j] > DRO_BASE: # Terminate the summation and calculate DRO_SEV if SM is no longer below the base percentile for drought
                                if nday < 4*5:
                                    # DRO_SEV is set to 0 if drought was not consistent for at least 4 pentads after rapid intensificaiton (i.e., little to no impact)
                                    dro_sev[t,i,j] = 0
                                    break
                                else:
                                    dro_sev[t,i,j] = dro_sum/nday # Terminate the loop and determine the drought severity if the drought condition is broken
                                    break
                                
                            elif (nday >= 18*5): # Calculate the drought severity of the loop goes out 90 days, but the drought does not end
                                dro_sev[t,i,j] = dro_sum/nday
                                break
                            else:
                                pass
                
                # In continuing consistency with Otkin et al. 2021, if the pentad does not immediately follow rapid intensification, drought is set 0
                else:
                    dro_sev[t,i,j] = 0
    
    print(np.min(dro_sev), np.max(dro_sev), np.mean(dro_sev))
    
    print('Calculating FDII')
    
    # Finally, FDII is the product of the rapid intensification and drought severity components
    fdii = fd_int * dro_sev
    
    print(np.min(fdii), np.max(fdii), np.mean(fdii))

    # Remove values less than 0
    fd_int[fd_int <= 0] = 0
    dro_sev[dro_sev <= 0] = 0
    fdii[fdii <= 0] = 0

    print('Done')
    
    return fdii, fd_int, dro_sev


def calculate_sm_percentiles(
        sm, 
        sm_all, 
        dates, 
        dates_all, 
        mask = None, 
        ) -> np.ndarray:
    '''
    Calculate the soil moisture percentiles using a larger popularion of soil moisture data

    Note this method is NOT space efficient. It requires in the full soil moisture dataset which 
    can be large to produce timely computations.

    Inputs:
    :param sm: Soil moisture dataset (np.ndarray with shape time x lat  x lon)
    :param sm_all: Full soil moisture dataset 
                   (list, with each list entry being one year of sm data, an np.ndarray with shape time_for_one_year x lat x lon)
    :param dates: Datetime labels for each time step in sm (np.ndarray with shape time)
    :param dates_all: Datetime labels for each time step in sm_all (np.ndarray with shape time_for_all_dates)
    :param mask: Land-sea mask with values 1 for land and 0 for sea (np.ndarray with shape lat x lon)
    
    Outputs:
    :param smp: Percentiles for each value in sm (np.ndarray with shape time x lat x lon)
    '''
        
    T, I, J= sm.shape # Obtain the dataset size to intialize the percentile dataset
    
    # All years in the time series
    all_years = np.unique([date.year for date in dates_all])

    # Calculate all years in the full time series
    years = np.array([date.year for date in dates_all])
    months = np.array([date.month for date in dates_all])
    days = np.array([date.day for date in dates_all])

        
    # Initialize percentile dataset
    smp = np.zeros((T, I, J))
 
    # sm = np.concatenate(sm, axis = 0)

    # n = 0
    for i in tqdm(range(I)):
        for j in range(J):
            # Skip sea values
            if mask[i,j] == 0:
                continue

            #print('%d/%d'%(n, I*J))
            sm_time_series = []

            # Determine the complete time series for a given grid point
            for y, _ in enumerate(all_years):
                sm_time_series.append(sm_all[y][:,i,j])
            
            sm_time_series = np.concatenate(sm_time_series)
            
            # Calculate the SM percentiles for all points in the time axis for a given grid point
            for t, date in enumerate(dates):
                # Obtain all indices for the current day of the year
                ind = np.where((date.day == days) & (date.month == months))[0] 

                # Calculate the SM percentile based on the current day of the year
                smp[t,i,j] = stats.percentileofscore(sm_time_series[ind], sm[t,i,j])

            # n = n+1

    # print(np.nansum(smp <10), np.nansum(smp > 90))
    smp = smp.astype(np.float32)

    return smp

