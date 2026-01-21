'''
Check if the LSCRATCH directory created/initialized correctly; exit the program if not

Credit to Dr. Marina Vincens-Miquel for the code
'''

import os, sys
lscratch_path = os.environ.get('LSCRATCH', '/lscratch')

if not os.path.isdir(lscratch_path):
    print(f'Error: {lscratch_path} directory does not exist. Exiting')
    sys.exit(1)