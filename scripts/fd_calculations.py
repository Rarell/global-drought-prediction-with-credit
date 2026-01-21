import gc
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
from netCDF4 import Dataset
from argparse import ArgumentParser
from tqdm import tqdm

# from inputs_outputs import load_nc

def calculate_climatology(e, pet, dates_all, days_per_year = 366):
    '''
    Calculates the climatological mean and standard deviation of ESR from daily ERA5 data.
    Climatological data is calculated for all grid points and for all timestamps in the year.
    
    Note this function also makees some hard assumptions on the path to the datasets, and file names

    Inputs:
    :param path_to_data: path to the era5 data.
    '''
    

    T, I, J = e.shape
    T = days_per_year # Numbers of days in a year

    # All years in climatology calculations
    all_years = np.unique([date.year for date in dates_all])
    years = np.array([date.year for date in dates_all])

    days = np.array([day.day for day in dates_all])
    months = np.array([day.month for day in dates_all])

    # Get datetimes
    dates_year = np.array([datetime(2012,1,1) + timedelta(days = day) for day in range(days_per_year)])

    # Initialize climatology means and standard deviations + counts
    means = np.zeros((T, I, J), dtype = np.float32)
    stds = np.zeros((T, I, J), dtype = np.float32)

    N = np.zeros((T))

    # Construct ESR for the year
    esr = e/pet

    # Remove values that exceed a certain limit as they are likely an error
    esr[esr < 0] = np.nan
    esr[esr > 3] = np.nan 
    # print(np.nansum(np.isnan(esr)))

    print('Initialized variables, calculation means')

    # Conduct climatology calculations; load only 1 year of data at a time so that no more than 7.5 GB is loaded into the memory at a time
    # for year in all_years:

    #     # Construct datetime information
    #     ind = np.where(year == years)[0]
    #     dates_current_year = np.array([day for day in dates_all[ind]])

    for t, date in enumerate(dates_year):
        # Get the days in the current corresponding to the day in the loop (may not be the same as t as current year may not be a leap year)
        ind = np.where( (date.day == days) & (date.month == months) )[0]

        # For non-leap years, ind can be empty; skip this day
        if len(ind) < 1:
            continue
        else:
            tmp_sum = np.nansum(esr[ind,:,:], axis = 0)
            means[t,:,:] = np.nansum([means[t,:,:], tmp_sum], axis = 0) # np.nansum to account for any NaNs
            N[t] = N[t] + len(ind)

    # At the end, loop again to get to divide the sums by N to get the means
    for t, date in enumerate(dates_year):
        means[t,:,:] = means[t,:,:]/N[t]

    means = means.astype(np.float32)

    print('Means calculated, calculating standard deviations')

    # Loop again to do the standard deviation calculation
    # for year in all_years:

    #     # Construct datetime information
    #     ind = np.where(year == years)[0]
    #     dates_current_year = np.array([day for day in dates_all[ind]])
    

    for t, date in enumerate(dates_year):
        # Get the days in the current corresponding to the day in the loop (may not be the same as t as current year may not be a leap year)
        ind = np.where( (date.day == days) & (date.month == months) )[0]
        
        # For non-leap years, ind can be empty; skip this day
        if len(ind) < 1:
            continue
        else:
            error = np.nansum((esr[ind,:,:] - means[t,:,:])**2, axis = 0)
            stds[t,:,:] = np.nansum([stds[t,:,:], error], axis = 0)

    # One final loop to finish STD calculations
    for t, date in enumerate(dates_year):
        stds[t,:,:] = np.sqrt(stds[t,:,:]/(N[t] - 1))

    stds = stds.astype(np.float32)
    print(np.min(means), np.max(means))
    print(np.min(stds), np.max(stds))

    print('Standard deviations calculated')

    ind = np.where(years == 2012)[0]
    one_year = dates_all[ind]

    return means, stds, one_year

def calculate_sesr(et, pet, dates, means, stds, one_year):
    '''
    Calculate the standardized evaporative stress ratio (SESR) from ET and PET for 1 year.
    
    Full details on SESR can be found in Christian et al. 2019 (for SESR): https://doi.org/10.1175/JHM-D-18-0198.1.
    
    Inputs:
    :param et: Input evapotranspiration (ET) data
    :param pet: Input potential evapotranspiration (PET) data. Should be in the same units as et
    :param year: The year the FDII is being calculated for
    :param path_to_data: path to the era5 data.
        
    Outputs:
    :param sesr: Calculate SESR, has the same shape and size as et/pet
    
    '''

    # Get date information
    T, I, J = et.shape
    # dates = np.array([datetime(year, 1, 1) + timedelta(days = t) for t in range(T)])


    # Obtain the evaporative stress ratio (ESR); the ratio of ET to PET
    esr = et/pet

    # Remove values that exceed a certain limit as they are likely an error
    esr[esr < 0] = np.nan
    esr[esr > 3] = np.nan
    # print(np.nansum(np.isnan(esr)))

    months = np.array([date.month for date in one_year])
    days = np.array([date.day for date in one_year])

    sesr = np.ones((T, I, J)) * np.nan

    for t, date in enumerate(dates):
        ind = np.where( (date.month == months) & (date.day == days) )[0]
        
        sesr[t,:,:] = (esr[t,:,:] - means[ind[0],:,:])/stds[ind[0],:,:]
            

    # Remove any unrealistic points
    sesr = np.where(sesr < -5, -5, sesr)
    sesr = np.where(sesr > 5, 5, sesr)

    sesr = sesr.astype(np.float32)
    
    print(np.nanmin(sesr), np.nanmax(sesr), np.nanmean(sesr))
    # print(np.sum(sesr <= -4.5), np.nansum(sesr >= 4.5))
    return sesr

def calculate_fdii(smp, dates, apply_runmean = True, mask = None):
    '''
    Calculate the flash drought intensity index (FDII) from a soil moisture percentiles.
    FDII index is on the same time scale as the input data.
    
    Full details on FDII can be found in Otkin et al. 2021: https://doi.org/10.3390/atmos12060741
    
    Inputs:
    :param smp: Input soil moisture percentiles. Time x lat x lon format
    :param year: The year the FDII is being calculated for
    :param apply_runmean: Apply a 5 length centered running mean to the percentiles before FDII calculations (recommended for daily data)
    :param use_mask: Indicates whether to use a land-sea mask to speed computation
    :param path_to_mask: String path to the mask data if it is used

    Outputs:
    :param fdii: The FDII drought index, has the same shape and size as vsm
    :param fd_int: The strength of the rapid intensification of the flash drought, has the same shape and size as vsm
    :param dro_sev: Severity of the drought component of the flash drought, has the same shape and size as vsm

    TODO: May need to add some filter to clear out the signal in sm percentiles
    '''
       
    print('Initializing some variables')
    # Define some base constants
    PER_BASE = 15 # Minimum percentile drop in 4 pentads
    T_BASE   = 4*5
    DRO_BASE = 20 # Percentiles must be below the 20th percentile to be in drought
    
    # Next, FDII can be calculated with the standardized soil moisture, or percentiles.
    # Use percentiles for consistancy with Otkin et al. 2021
    T, I, J = smp.shape

    # Make the years, months, and/or days variables
    years = np.array([date.year for date in dates])
    months = np.array([date.month for date in dates])
    days = np.array([date.day for date in dates])

    if apply_runmean:
        print('Applying 5 day running mean')
        runmean = 5
        start_ind = int(np.round((runmean - 1)/2))
        end_ind = int(T + runmean - 1 - start_ind)
        for i in tqdm(range(I), desc = 'Applying running mean'):
            for j in range(J):
                smp[:,i,j] = np.convolve(smp[:,i,j], np.ones((runmean))/runmean)[start_ind:end_ind]
    
    # Load the mask? (for ERA5 only)
    # if use_mask:
    #     with Dataset('%s/land.nc'%path_to_mask, 'r') as nc:
    #         mask = nc.variables['lsm'][0,:,:]
    
    print(np.nanmin(smp), np.nanmax(smp))
    print(np.nanmean(smp))
    
    print('Calculating rapid intensification of flash drought')
    # Determine the rapid intensification based on percentile changes based on equation 1 in Otkin et al. 2021 (and detailed in section 2.2 of the same paper)
    fd_int = np.zeros((T, I, J))

    # Determine the intensification index
    for i in tqdm(range(I), desc = 'Calculating FD_INT'):
        for j in range(J):
            # Ignore sea points
            if mask[i,j] == 0: # ERA5 only
                continue
        
            for t in range(T-10): # Note the last two days are excluded as there is no change to examine
            
                obs = np.zeros((9*5)) # Note, the method detailed in Otkin et al. 2021 involves looking ahead 2 to 10 pentads (9 entries total)
                for nday in np.arange(2*5, 10*5+5, 1):
                    nday = int(nday)
                    if (t+nday) >= T: # If t + npend is in the future (beyond the dataset), break the loop and use 0s for obs instead
                        break          # This should not effect results as this will only occur in November to December, outside of the growing season.
                    else:
                        obs[nday-10] = (smp[t+nday,i,j] - smp[t,i,j])/nday # Note npend is the number of pentads the system is corrently looking ahead to.
                if np.nansum(np.isnan(obs)) == obs.size:
                    print('All NaNs: ', t, nday, i, j)
                # If the maximum change in percentiles is less than the base change requirement (15 percentiles in 4 pentads), set FD_INT to 0.
                #  Otherwise, determine FD_INT according to eq. 1 in Otkin et al. 2021
                if np.nanmax(obs) < (PER_BASE/T_BASE):
                    fd_int[t,i,j] = 0
                else:
                    fd_int[t,i,j] = ((PER_BASE/T_BASE)**(-1)) * np.nanmax(obs)
                
    print(np.min(fd_int), np.max(fd_int), np.mean(fd_int))
    
    
    print('Calculating drought severity')
    # Next determine the drought severity component using equation 2 in Otkin et al. 2021 (and detailed in section 2.2 of the same paper)
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
                        
                        if (t+nday) >= T:      # For simplicity, set DRO_SEV to 0 when near the end of the dataset (this should not impact anything as it is not in
                            dro_sev[t,i,j] = 0 # the growing season)
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
                    continue
    
    print(np.min(dro_sev), np.max(dro_sev), np.mean(dro_sev))
    
    print('Calculating FDII')
    
    # Finally, FDII is the product of the components
    fdii = fd_int * dro_sev
    
    print(np.min(fdii), np.max(fdii), np.mean(fdii))

    # Remove values less than 0
    fd_int[fd_int <= 0] = 0
    dro_sev[dro_sev <= 0] = 0
    fdii[fdii <= 0] = 0

    print('Done')
    
    return fdii, fd_int, dro_sev


def calculate_sm_percentiles(sm, sm_all, dates, dates_all, mask = None, level = 1, path_to_data = '../era5'):
    '''
    Calculate the soil moisture percentiles for a singular year and save the results to a .nc file

    Note this method is NOT space efficient. It loads in the full soil moisture dataset (35 GB for 
    quarter degree resolution) to produce timely computations.

    Note this function also makees some hard assumptions on the path to the datasets, and file names
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

            # Load in all times for 1 grid point at a time
            for y, _ in enumerate(all_years):
                sm_time_series.append(sm_all[y][:,i,j])
            
            sm_time_series = np.concatenate(sm_time_series)
            
            # Calculate the SM percentiles for all points in the year for the grid point
            for t, date in enumerate(dates):
                ind = np.where((date.day == days) & (date.month == months))[0] # Obtain all indices for the current day of the year

                # Calculate the SM percentile based on the current day of the year
                smp[t,i,j] = stats.percentileofscore(sm_time_series[ind], sm[t,i,j])

            # n = n+1

    # print(np.nansum(smp <10), np.nansum(smp > 90))
    smp = smp.astype(np.float32)

    return smp

# def create_aridity_mask(path_to_data = '../era5'):
#     '''
#     Create an aridity mask based on aridity index and potential evaporation 
#     (aridity mask is similar in concept to a land-sea mask, but covers highly arid locations)

#     NOTE: This function makes some hard assumptions about the paths and filenames for variables
#     '''

#     # All the years used for the aridity index calculation
#     # years = np.arange(2001, 2024)
#     years = np.arange(2000, 2025)

#     # Load a test dataset to get the data size
#     # test = load_nc('%s/potential_evaporation/gldas.potential_evaporation.daily.2004.nc'%path_to_data, 'pevap') # Note, for a  reduced scale, scale reduction is done here
#     test = load_nc('%s/potential_evaporation/potential_evaporation_2000.nc'%path_to_data, 'pev') # Note, for a  reduced scale, scale reduction is done here
#     T, I, J = test['pevap'].shape
#     # T, I, J = test['pev'].shape

#     # Initialize datasets
#     p_annual = np.ones((years.size, I, J), dtype = np.float32) * np.nan
#     pet_annual = np.ones((years.size, I, J), dtype = np.float32) * np.nan

#     # Load in and sum over all precipitation and PET values to get annual accumulations
#     for t, year in enumerate(years): # Note here that P and PET should both be in units of m; also this gives total annual accumulation
#         # p = load_nc('%s/precipitation/gldas.precipitation.daily.%04d.nc'%(path_to_data, year), 'precip')
#         p = load_nc('%s/precipitation/total_precipitation_%04d.nc'%(path_to_data, year), 'tp')
#         #p_reduced, _, _ = reduce_spatial_scale(p, 'tp') # Scale reduction also done here if necessary
#         # pet = load_nc('%s/potential_evaporation/gldas.potential_evaporation.daily.%04d.nc'%(path_to_data, year), 'pevap')
#         pet = load_nc('%s/potential_evaporation/potential_evaporation_%04d.nc'%(path_to_data, year), 'pev')
#         #pet_reduced, _, _ = reduce_spatial_scale(pet, 'pev') # Scale reduction also done here if necessary

#         # For GLDAS2 data only, convert PET from W m^-2 to kg m^-2 s^-1 (consistent with precip) by dividing by the latent heat of vaporization
#         # pet['pevap'] = pet['pevap'] / (2.5e6)

#         # p_annual[t,:,:] = np.nansum(p['precip'], axis = 0)
#         # pet_annual[t,:,:] = np.nansum(pet['pevap'], axis = 0)
#         p_annual[t,:,:] = np.nansum(p['tp'], axis = 0)
#         pet_annual[t,:,:] = np.nansum(pet['pev'], axis = 0)

#         # For GLDAS2 only, convert PET from kg m^-2 to m be dividing by the density of water for consistent units of m (used in daily PET calculation)
#         # pet_annual[t,:,:] = pet_annual[t,:,:] / 1000


#     # Aridity index is the mean annual precipitation accumulation divided by mean annual PET accumulation
#     arid_index = np.nanmean(p_annual, axis = 0)/np.abs(np.nanmean(pet_annual, axis = 0)) # For the purposes of the ratio, PET is assumed positive in the aridity index calculations

#     # When creation the aridity mask, the daily PET is also needed
#     pet_daily = np.abs(np.nanmean(pet_annual, axis = 0)) / 365 # Nanmean delivers the annual mean PET accumulation; division by 365 approximates average daily accumulation

#     # Aridity mask will be based on PET from across the entire year (instead of growing season; for now...)
#     ai_mask = np.where( (arid_index < 0.2) | (pet_daily < 0.001), 0, 1) # Note the requirement is mean daily PET < 1 mm/day = 0.001 m/day

#     # ERA5 Only: Note there is a strange behavoir in ERA5 that tries to mask out the Congo Basin, despite it not being arid. This correct that error
#     condition = (np.abs(test['lat']) < 10) & ((test['lon'] >= 11.5) & (test['lon'] <= 30))
#     ai_mask = np.where(condition, 1, ai_mask)

#     # Put the aridity mask to 3D in the same format as the ERA5 land-sea mask
#     aridity_mask = np.zeros((1, I, J))
#     aridity_mask[0,:,:] = ai_mask.astype(np.int16)
  
#     # Write the aridity mask
#     with Dataset('%s/aridity_mask.nc'%path_to_data, 'w', format = 'NETCDF4') as nc:
#         nc.description = 'Global aridity index based on daily ERA5 reanalysis precipitation and potential evaporation'
#         # nc.description = 'Global aridity index based on daily GLDAS2 reanalysis precipitation and potential evaporation'

#         # Create the dimension information (same format as the land-sea mask)
#         nc.createDimension('latitude', size = I)
#         nc.createDimension('longitude', size = J)
#         nc.createDimension('time', size = 1)

#         # Create the latitude and longitude information
#         nc.createVariable('latitude', test['lat'].dtype, ('latitude',))
#         nc.createVariable('longitude', test['lon'].dtype, ('longitude',))
#         # nc.variables['latitude'][:] = test['lat'][:]
#         # nc.variables['longitude'][:] = test['lon'][:]
#         nc.variables['latitude'][:] = test['lat'][:,0]
#         nc.variables['longitude'][:] = test['lon'][0,:]

#         # Create and store the aridity mask
#         nc.createVariable('aim', aridity_mask.dtype, ('time', 'latitude', 'longitude'))
#         nc.variables['aim'][:] = aridity_mask[:]

if __name__ == '__main__':
    description = 'Create aridity variables and indices related to flash drought for future ML use'
    parser = ArgumentParser(description = description)
    parser.add_argument('--calculate_aridity_mask', action = 'store_true', help = 'Create an mask based on the aridity index and save as an nc file')
    parser.add_argument('--calculate_climatology', action = 'store_true', help = 'Calculate the ESR climatologies and save as an nc file')
    parser.add_argument('--calculate_percentiles', action = 'store_true', help = 'Calculate 1 year of soil moisture percentiles based on year and level')
    parser.add_argument('--calculate_indices', action = 'store_true', help = 'Calculate 1 year SESR and FDII flash drought indices based on year')

    parser.add_argument('--year', type = int, default = 0, help = 'Year to perform percentile/index calculations (note the actual year used is year + 2000)')
    parser.add_argument('--level', type = int, default = 1, help = 'ERA5 soil moisture level used percentile and FDII caluclations (must be 1 - 4)')

    args = parser.parse_args()

    year = args.year + 2000

    # Calculate aridity mask?
    if args.calculate_aridity_mask:
        create_aridity_mask()#path_to_data = '../gldas')

    if args.calculate_climatology:
        calculate_climatology()#path_to_data = '../gldas')

    # Calculate percentiles?
    if args.calculate_percentiles:
        calculate_sm_percentiles(year, level = args.level)#, path_to_data = '../gldas')

    # Calculate the flash drought indices?
    if args.calculate_indices:
        # Load ET and PET for SESR
        print('Loading ET and PET for SESR calculations')

        et = load_nc('../gldas/evaporation/gldas.evaporation.daily.%04d.nc'%(year), 'evap')
        pet = load_nc('../gldas/potential_evaporation/gldas.potential_evaporation.daily.%04d.nc'%(year), 'pevap')
        # et = load_nc('../era5/evaporation/evaporation_%04d.nc'%(year), 'e')
        # pet = load_nc('../era5/potential_evaporation/potential_evaporation_%04d.nc'%(year), 'pev')

        # For GLDAS2 data only, convert PET from W m^-2 to kg m^-2 s^-2 by dividing by the latent heat of vaporization
        # Also remove NaNs to avoid skewing test calculations
        pet['pevap'] = pet['pevap'] / (2.5e6)
        # et['evap'][et['evap'] <= -900] = np.nan
        # pet['pevap'][pet['pevap'] <= -900] = np.nan

        # Calculate SESR
        print('Calculating SESR')
        sesr = calculate_sesr(et['evap'], pet['pevap'], year, path_to_data = '../gldas')
        # sesr = calculate_sesr(et['e'], pet['pev'], year)

        sesr[np.isnan(sesr)] = -9999

        sesr = sesr.astype(np.float32)
        print(sesr.shape, et['lat'].shape)

        # Write results to nc file
        print('Saving SESR results')
        with Dataset('../gldas/sesr_%04d.nc'%(year), 'w', format = 'NETCDF4') as nc:
        # with Dataset('../era5/sesr_%04d.nc'%(year), 'w', format = 'NETCDF4') as nc:
            nc.description = 'Daily ERA5 reanalysis data for SESR, calculated from evaporation and potential evaporaiton'
            # nc.description = 'Daily GLDAS2 reanalysis data for SESR, calculated from evaporation and potential evaporaiton'

            # Create Dimensions
            T, I, J = sesr.shape
            nc.createDimension('latitude', size = I)
            nc.createDimension('longitude', size = J)
            nc.createDimension('time', size = T)

            # Create the lat, lon, and time information
            nc.createVariable('lat', et['lat'].dtype, ('latitude',))
            nc.createVariable('lon', et['lon'].dtype, ('longitude',))
            # nc.createVariable('lat', et['lat'].dtype, ('latitude', 'longitude'))
            # nc.createVariable('lon', et['lon'].dtype, ('latitude', 'longitude'))
            nc.createVariable('date', str, ('time',))

            nc.variables['lat'][:] = et['lat'][:]
            nc.variables['lon'][:] = et['lon'][:]
            nc.variables['date'][:] = et['time'][:]

            # Create and save the percentiles
            nc.createVariable('sesr', sesr.dtype, ('time', 'latitude', 'longitude'))
            nc.variables['sesr'][:] = sesr[:]

        # Remove large variables to free up space
        del et, pet, sesr
        gc.collect()

        # Load SM percentiles for FDII calculations
        print('Loading soil moisture percentiles')
        smp = load_nc('../gldas/soil_moisture_percentiles/soil_moisture_percentiles_%d_%04d.nc'%(args.level, year), 'smp%d'%args.level)
        # smp = load_nc('../era5/soil_moisture_percentiles/soil_moisture_percentiles_%d_%04d.nc'%(args.level, year), 'smp%d'%args.level)
        #smp = load_nc('../era5/soil_moisture_percentiles_%d_%04d.nc'%(args.level, year), 'smp%d'%args.level)

        # Calculate FDII
        print('Calculating FDII for layer %d'%args.level)
        fdii, fd_int, dro_sev = calculate_fdii(smp['smp%d'%args.level], year, apply_runmean = True, use_mask = True)#, path_to_data = '../gldas')

        fdii = fdii.astype(np.float32); fd_int = fd_int.astype(np.float32); dro_sev = dro_sev.astype(np.float32)

        # Write the results to a nc file
        print('Saving FDII results')
        with Dataset('../gldas/fdii_%d_%04d.nc'%(args.level, year), 'w', format = 'NETCDF4') as nc:
        # with Dataset('../era5/fdii_%d_%04d.nc'%(args.level, year), 'w', format = 'NETCDF4') as nc:
            nc.description = 'Daily GLDAS2 reanalysis data for FDII and associated components, calculated from soil moisture percentiles for layer %d'%args.level
            nc.description = 'Daily ERA5 reanalysis data for FDII and associated components, calculated from soil moisture percentiles for layer %d'%args.level

            # Create Dimensions
            T, I, J = fdii.shape
            nc.createDimension('latitude', size = I)
            nc.createDimension('longitude', size = J)
            nc.createDimension('time', size = T)

            # Create the lat, lon, and time information
            nc.createVariable('lat', smp['lat'].dtype, ('latitude', ))
            nc.createVariable('lon', smp['lon'].dtype, ('longitude', ))
            # nc.createVariable('lat', smp['lat'].dtype, ('latitude', 'longitude'))
            # nc.createVariable('lon', smp['lon'].dtype, ('latitude', 'longitude'))
            nc.createVariable('date', str, ('time', ))

            nc.variables['lat'][:] = smp['lat'][:]
            nc.variables['lon'][:] = smp['lon'][:]
            nc.variables['date'][:] = smp['time'][:]

            # Create and save the percentiles
            nc.createVariable('fdii%d'%args.level, fdii.dtype, ('time', 'latitude', 'longitude'))
            nc.variables['fdii%d'%args.level][:] = fdii[:]

            nc.createVariable('fd_int%d'%args.level, fd_int.dtype, ('time', 'latitude', 'longitude'))
            nc.variables['fd_int%d'%args.level][:] = fd_int[:]

            nc.createVariable('dro_sev%d'%args.level, dro_sev.dtype, ('time', 'latitude', 'longitude'))
            nc.variables['dro_sev%d'%args.level][:] = dro_sev[:]


# test = {}
# with Dataset('../era5/liquid_vsm/volumetric_soil_water_layer_1_2001.nc', 'r') as nc:
#      test['swvl1'] = nc.variables['swvl1'][:]
#      test['lat'] = nc.variables['lat'][:]
#      test['lon'] = nc.variables['lon'][:]
#      test['time'] = nc.variables['date'][:]
# all_years = np.arange(2000, 2024)
# T, I, J = test['swvl1'].shape
# smp = np.ones((T, I, J)) * np.nan
# N_leap_days = np.sum((all_years % 4) == 0)
# T = (T * all_years.size) + N_leap_days
# date_initial = datetime(all_years[0], 1, 1)
# dates = np.array([date_initial + timedelta(days = t) for t in range(T)])
# years = np.array([date.year for date in dates])
# months = np.array([date.month for date in dates])
# days = np.array([date.day for date in dates])
# with Dataset('../era5/land.nc' ,'r') as nc:
#      mask = nc.variables['lsm'][0,:,:]
# for year in all_years:
#      with Dataset('../era5/liquid_vsm/volumetric_soil_water_layer_1_%04d.nc'%year, 'r') as nc:
#          test['swvl1'] = nc.variables['swvl1'][:]
#          test['lat'] = nc.variables['lat'][:]
#          test['lon'] = nc.variables['lon'][:]
#          test['time'] = nc.variables['date'][:]
#          dates_year = np.array([datetime.fromisoformat(date) for date in test['time']])
#      print('Loaded test set')
#      T, I, J = test['swvl1'].shape
#      smp = np.ones((T, I, J)) * np.nan
#      print('Re-initialized percentile dataset for %d'%year)
#      n = 0
#      for i in range(I):
#          for j in range(J):
#              if mask[i,j] == 0:
#                  n = n+1
#                  continue
#              sm_time_series = []
#              print('%d/%d'%(n, I*J))
#              for y in all_years:
#                  with Dataset('../era5/liquid_vsm/volumetric_soil_water_layer_1_%04d.nc'%y, 'r') as nc:
#                      for t in range(nc.variables['swvl1'].shape[0]):
#                          sm_time_series.append(nc.variables['swvl1'][t,i,j])
#              sm_time_series = np.array(sm_time_series)
#              for t, date in enumerate(dates_year):
#                  ind = np.where((date.day == days) & (date.month == months))[0]
#                  current_day = np.where((date.day == days) & (date.month == months) & (date.year == years))[0]
#                  smp[t,i,j] = stats.percentileofscore(sm_time_series[ind], sm_time_series[current_day])
#              n = n+1
#      print('Finished soil moisture percentile calculations for %d'%year)
#      with Dataset('soil_moisture_percentiles_1_%04d.nc'%year, 'w', format = 'NETCDF4') as nc:
#          nc.createDimension('latitude', size = I)
#          nc.createDimension('longitude', size = J)
#          nc.createDimension('time', size = T)
#          nc.createVariable('lat', test['lat'].dtype, ('latitude', 'longitude'))
#          nc.createVariable('lon', test['lon'].dtype, ('latitude', 'longitude'))
#          nc.createVariable('date', str, ('time'))
#          nc.variables['lat'][:] = test['lat'][:]
#          nc.variables['lon'][:] = test['lon'][:]
#          nc.variables['date'][:] = test['time'][:]
#          nc.createVariable('smp1', smp.dtype, ('time', 'latitude', 'longitude'))
#          nc.variables['smp1'][:] = smp[:]
#      print('Finished writing data. Done with %d'%year)


# Climatologies:
