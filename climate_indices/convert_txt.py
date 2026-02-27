"""Converts climate index datasets from the
downloaded text files into .csv files with
consistent formatting files that have two 
columns (time, index data)
"""

import numpy as np
from io import StringIO
from datetime import datetime, timedelta

def convert_standard_txt(
        filename, 
        out_fname, 
        ind = 'nao', 
        max_rows = 75, 
        path = './'
        ) -> None:
    '''
    Convert the standard index text file, that is tab separated, to the same comma separated files used in the PSL

    Inputs:
    :param filename: Filename of the original text file to load
    :param out_fname: Filename of the created csv file
    :parm ind: Name of the index data (nao or others)
    :param max_rows: Maximum number of rows in the text file to load
    :param path: Directory path to the index data
    '''
    # Replace the spaces that separate numbers to commas for easier loading
    # Note the amount of separation is different between the NAO and AMO files
    string = open('%s/%s'%(path, filename)).read().replace("  ", ",") if 'nao' in filename else open(filename).read().replace("   ", ",")
    
    # Load the data
    timestamps = np.loadtxt(StringIO(string), 
                            delimiter = ',', 
                            skiprows = 1, 
                            usecols = 0, 
                            max_rows = max_rows)
    data = np.loadtxt(StringIO(string), 
                      delimiter = ',', 
                      skiprows = 1, 
                      usecols = np.arange(1, 13), 
                      max_rows = max_rows)
    # Formatted as [Jan. Feb. ... Dec.] per row

    # Make the time axis to datetimes, then to standard ISO
    months = np.arange(1, 13)
    dates = [datetime(int(year), month, 1).isoformat() for year in timestamps for month in months] # Second loop goes first

    # Flatten the data unto a column
    data = data.flatten() # Note flatten prioritizes rows for flattening by default, so entries go Jan, Feb, ...

    # Write the data as a standard, csv file with two columns (time and data)
    with open('%s/%s'%(path, out_fname), 'w') as file:
        file.write('Time, %s\n'%ind) # Header

        # Write the data
        for date, datum in zip(dates, data):
            file.write('%s,%10.3f\n'%(date,datum))

def convert_amo_txt(
        filename, 
        out_fname, 
        path = './'
        ) -> None:
    '''
    Convert the standard index AMO text file, that is tab separated, to the same comma separated files used in the PSL
    This is done separately from other index data due to the way it is formatted

    Inputs:
    :param filename: Filename of the original text file to load
    :param out_fname: Filename of the created csv file
    :param max_rows: Maximum number of rows in the text file to load
    :param path: Directory path to the index data
    '''

    years = []
    months = []
    data = []

    # Load the AMO data
    with open('%s/%s'%(path, filename), 'r') as file:
        lines = file.readlines() # Collect the text
        
        # Collect the time information
        # For each line, isolate the strings for the dates, and individual values
        for line in lines[2:]:
            years.append(int(line[0:4]))
            months.append(int(line[9:11]))
            data.append(float(line[16:21]))

    years = np.array(years)
    months = np.array(months)
    data = np.array(data)

    # Convert the dates into datetimes
    dates = np.array([datetime(years[t], months[t], 1) for t in range(years.size)])

    # Make the time axis to datetimes, then to standard ISO
    months = np.arange(1, 13)
    #dates = [datetime(int(year), month, 1).isoformat() for year in timestamps for month in months] # Second loop goes first

    # Write the data
    with open('%s/%s'%(path, out_fname), 'w') as file:
        file.write('Time, AMO\n') # Header

        # Write the data
        for date, datum in zip(dates, data):
            file.write('%s,%10.3f\n'%(date,datum))
    

def convert_enso_txt(
        filename, 
        out_fname, 
        path = './'
        ) -> None:
    '''
    Convert the ENSO text file into the same comma separated files used in the PSL.
    Note this requires extra care as the text file has SST and SSTAs, and there is no space between them when
    there is a minus sign

    Inputs:
    :param filename: Filename of the original text file to load
    :param out_fname: Filename of the created csv file
    :param path: Directory path to the index data
    '''

    # Initialize lists for the different datasets
    timestamps = []
    nino12_sst = []; nino12_ssta = []
    nino3_sst  = []; nino3_ssta  = []
    nino34_sst = []; nino34_ssta = []
    nino4_sst  = []; nino4_ssta  = []

    # Load the ENSO data
    with open('%s/%s'%(path, filename), 'r') as file:
        lines = file.readlines() # Collect the text
        
        # For each line, isolate the strings for the dates, and individual values
        # Dates are converted to datetimes and data to floats
        for line in lines[4:]:
            timestamps.append(datetime.strptime(line[1:10], '%d%b%Y'))
            nino12_sst.append(float(line[15:19])); nino12_ssta.append(float(line[19:23]))
            nino3_sst.append(float(line[28:32])); nino3_ssta.append(float(line[32:36]))
            nino34_sst.append(float(line[41:45])); nino34_ssta.append(float(line[45:49]))
            nino4_sst.append(float(line[54:58])); nino4_ssta.append(float(line[58:62]))

    # Convert timestamps to ISO strings
    dates = [date.isoformat() for date in timestamps]

    # Write the ENSO data
    with open('%s/%s'%(path, out_fname), 'w') as file:
        # Header
        file.write('Time, Nino1+2_SST, Nino1+2_SSTa, Nino3_SST, Nino3_SSTa, Nino34_SST, Nino34_SSTa, Nino4_SST, Nino4_SSTa\n')

        # Write the data
        for date, n12sst, n12ssta, n3sst, n3ssta, n34sst, n34ssta, n4sst, n4ssta in zip(dates, 
                                                                                        nino12_sst, 
                                                                                        nino12_ssta, 
                                                                                        nino3_sst, 
                                                                                        nino3_ssta, 
                                                                                        nino34_sst, 
                                                                                        nino34_ssta, 
                                                                                        nino4_sst, 
                                                                                        nino4_ssta):
            file.write('%s,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f\n'%(date,
                                                                                       n12sst, 
                                                                                       n12ssta,
                                                                                       n3sst, 
                                                                                       n3ssta,
                                                                                       n34sst, 
                                                                                       n34ssta,
                                                                                       n4sst, 
                                                                                       n4ssta))



if __name__ == '__main__':
    # Test function to load in text files and test correct values were loaded
    # Note the test function has been lost
    # data, dates = load_index_data('pdo.timeseries.sstens.csv', timestamps_prepared = False)
    # print(data[:5], dates[:5])
    
    # data, dates = load_index_data('amo.timeseries.csv', timestamps_prepared = True)
    # print(data[:5], dates[:5])

    # data, dates = load_index_data('enso.timeseries.csv', timestamps_prepared = True, enso = True)
    # print(data[:5,:], dates[:5])

    # Convert NAO to a csv file
    convert_standard_txt('nao_index.txt', 'nao.timeseries.csv', max_rows = 74, ind = 'nao')

    # Convert AMO index into a csv file
    # convert_standard_txt('amo_index.txt', 'amo.timeseries.csv', max_rows = 168, ind = 'amo')
    convert_amo_txt('amo_timeseries.txt', 'amo.timeseries.csv', max_rows = -1)

    # Convert ENSO to a csv file
    convert_enso_txt('enso_index.txt', 'enso.timeseries.csv')