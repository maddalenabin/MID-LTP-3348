# run the script with the following command with the first and last run number as arguments
# python check_analysed.py -rf {run1} -rl {run2}

import argparse, os
import numpy as np

# - parsers
parser = argparse.ArgumentParser(description='Check analysis')
parser.add_argument('-rf', '--run_first', type=int, required=True, help='first run number')
parser.add_argument('-rl', '--run_last', type=int, required=True, help='last run number')
args = parser.parse_args()

def check_runs_done(first, last):

    path = '/gpfs/exfel/exp/MID/202305/p003348/scratch/xpcs/'
    done_saxs = []
    done_xpcs = []

    for run in np.arange(first, last):
        if (os.path.isfile(f'{path}r{run:04d}/SAXS.npy') ):
            done_saxs.append(run)
        if (os.path.isfile(f'{path}r{run:04d}/TTCF_corrected.nc') ):
            done_xpcs.append(run)
    print("\n")
    print(f"SAXS run analysed: {done_saxs}")
    print("\n")
    print(f"XPCS run analysed: {done_xpcs}")
    print("\n")
    
check_runs_done(args.run_first, args.run_last)