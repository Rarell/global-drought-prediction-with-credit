import numpy as np
from io import StringIO
from datetime import datetime, timedelta

def convert_standard_txt(filename, out_fname, ind = 'nao', max_rows = 75, path = './'):
    '''
    Convert the standard index text ile, that is tab separated, to the same comma separated files used in the PSL
    '''
    # Replace the spaces that separate numbers to commas for easier loading
    # note the amount of separation is different between the NAO and AMO files
    string = open('%s/%s'%(path, filename)).read().replace("  ", ",") if 'nao' in filename else open(filename).read().replace("   ", ",")
    
    # Load the data
    timestamps = np.loadtxt(StringIO(string), delimiter = ',', skiprows = 1, usecols = 0, max_rows = max_rows)
    data = np.loadtxt(StringIO(string), delimiter = ',', skiprows = 1, usecols = np.arange(1, 13), max_rows = max_rows)
    # Formatted as [Jan. Feb. ... Dec.] per row

    # Make the time axis to datetimes, then to standard ISO
    months = np.arange(1, 13)
    dates = [datetime(int(year), month, 1).isoformat() for year in timestamps for month in months] # Second loop goes first

    # Flatten the data unto a column
    data = data.flatten() # Note flatten prioritizes rows for flattening by default, so entries go Jan, Feb, ...

    # Write the data
    with open('%s/%s'%(path, out_fname), 'w') as file:
        file.write('Time, %s\n'%ind)
        for date, datum in zip(dates, data):
            file.write('%s,%10.3f\n'%(date,datum))

def convert_amo_txt(filename, out_fname, max_rows = 75, path = './'):
    '''
    Convert the standard index text ile, that is tab separated, to the same comma separated files used in the PSL
    '''

    years = []
    months = []
    data = []

    # Load the AMO data
    with open('%s/%s'%(path, filename), 'r') as file:
        lines = file.readlines() # Collect the text
        
        # For each line, isolate the strings for the dates, and individual values
        for line in lines[2:]:
            years.append(int(line[0:4]))
            months.append(int(line[9:11]))
            data.append(float(line[16:21]))

    years = np.array(years)
    months = np.array(months)
    data = np.array(data)

    dates = np.array([datetime(years[t], months[t], 1) for t in range(years.size)])

    # Make the time axis to datetimes, then to standard ISO
    months = np.arange(1, 13)
    #dates = [datetime(int(year), month, 1).isoformat() for year in timestamps for month in months] # Second loop goes first

    # Write the data
    with open('%s/%s'%(path, out_fname), 'w') as file:
        file.write('Time, AMO\n')
        for date, datum in zip(dates, data):
            file.write('%s,%10.3f\n'%(date,datum))
    

def convert_enso_txt(filename, out_fname, path = './'):
    '''
    Convert the ENSO text file into the same comma separated files used in the PSL.
    Note this requires extra care as the text file has SST and SSTAs, and there is no space between them when
    there is a minus sign
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
        file.write('Time, Nino1+2_SST, Nino1+2_SSTa, Nino3_SST, Nino3_SSTa, Nino34_SST, Nino34_SSTa, Nino4_SST, Nino4_SSTa\n')
        for date, n12sst, n12ssta, n3sst, n3ssta, n34sst, n34ssta, n4sst, n4ssta in zip(dates, 
                                                                                        nino12_sst, nino12_ssta, 
                                                                                        nino3_sst, nino3_ssta, 
                                                                                        nino34_sst, nino34_ssta, 
                                                                                        nino4_sst, nino4_ssta):
            file.write('%s,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f\n'%(date,
                                                                                       n12sst, n12ssta,
                                                                                       n3sst, n3ssta,
                                                                                       n34sst, n34ssta,
                                                                                       n4sst, n4ssta))

    return


if __name__ == '__main__':
    # data, dates = load_index_data('pdo.timeseries.sstens.csv', timestamps_prepared = False)
    # print(data[:5], dates[:5])

    # data, dates = load_index_data('amo.timeseries.csv', timestamps_prepared = True)
    # print(data[:5], dates[:5])

    # data, dates = load_index_data('enso.timeseries.csv', timestamps_prepared = True, enso = True)
    # print(data[:5,:], dates[:5])

    # Prepare the text files
    #convert_standard_txt('nao_index.txt', 'nao.timeseries.csv', max_rows = 74, ind = 'nao')

    # convert_standard_txt('amo_index.txt', 'amo.timeseries.csv', max_rows = 168, ind = 'amo')
    convert_amo_txt('amo_timeseries.txt', 'amo.timeseries.csv', max_rows = -1)

    #convert_enso_txt('enso_index.txt', 'enso.timeseries.csv')