import os, sys

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

from extra_geom import AGIPD_1MGeometry
import extra_data as ex
import xarray as xr

import dask.array as da
from dask.distributed import Client, progress
from dask_jobqueue import SLURMCluster
from dask.diagnostics import ProgressBar

import importlib

import pyFAI
from pyFAI.distortion import Distortion

import cfut

from time import sleep

def get_statistics(run, arr, out_dir, method):

    
    np.save(out_dir+'/tid.npy',arr.trainId.data) ## trainIds
    
    from pathlib import Path
    output_file = Path(out_dir+'/mean_pixelcell.nc')
     
    if output_file.is_file():
        print('loading statistics from file')
        mean_pixelcell = xr.open_dataarray(out_dir+'/mean_pixelcell.nc')
        var_pixelcell = xr.open_dataarray(out_dir+'/var_pixelcell.nc')
        var_pixelrow = xr.open_dataarray(out_dir+'/var_pixelrow.nc')
    else:
        print('calculate statistics')
        #get the pulseIds corresponding to each row of storage cells
        reprate = run['MID_EXP_AGIPD1M1/MDL/FPGA_COMP','bunchStructure.repetitionRate'].as_single_value()
        pulseIds_of_rows = []
        all_possible_pulseIds = np.arange(0, 11*32*(4.5//reprate),(4.5//reprate)).reshape(-1,32)
        for pid in all_possible_pulseIds:
            pid_in_data = np.intersect1d(pid,arr.pulseId.data) #check if pulseIds are in arr array
            if len(pid_in_data) > 0:
                pulseIds_of_rows.append(pid_in_data)
                
                
        def calculate_statistics(i):
            # get pixel/storage cell statistics for one module
            print('module '+str(i))
            #(1)
            _mean = arr.sel(module=i).mean('trainId',skipna=True)
            #(2)
            _var = arr.sel(module=i).var('trainId',skipna=True)
            #(3)
            _var_pixelrow = []
            for pIds in pulseIds_of_rows:
                _var_pixelrow.append(arr.sel(module=i).sel(pulseId=pIds).var(('trainId','pulseId'),skipna=True))
            _var_pixelrow = xr.concat(_var_pixelrow,'row')

            #with ProgressBar(dt=30):
            #    _m, _vpc, _vpr = da.compute(_mean, _var, _var_pixelrow)

            #return _m, _vpc, _vpr
            return _mean, _var, _var_pixelrow
        
        mean_pixelcell =  [] #(1) mean per pixel and storage cell
        var_pixelcell =  []  #(2) variance per pixel and storage cell
        var_pixelrow =  []   #(3) variance per pixel and block of 32 storage cells 
        
        if method == 'dask_distributed':

            #partition = 'exfel'  # For EuXFEL staff
            partition = 'upex'   # For users

            cluster = SLURMCluster(
                queue=partition,
                processes=6, cores=70, memory='600GB',nanny=True,death_timeout=100,walltime="4:00:00", job_name='xpcs',
                local_directory = out_dir+'/dask',log_directory = out_dir+'/dask',
            )

            nworkers = 30
            cluster.scale(nworkers)
            print('waiting for requested nodes')
            while ((cluster.status == "running") and (len(cluster.scheduler.workers) < nworkers)):
                sleep(1.0)

            client = Client(cluster)
            print("Created dask client:", client)
            
            # link to dask dashbord
            print(cluster.dashboard_link)
            
        if method == 'dask_distributed' or method == 'dask_threads':
            
            for module in range(16):
                with ProgressBar():
                    results = da.compute(calculate_statistics(module))
                mean_pixelcell.append(results[0][0])
                var_pixelcell.append(results[0][1])
                var_pixelrow.append(results[0][2])
            

        elif method == 'cfut':
            
            def compute_statistics(i):
                return da.compute(calculate_statistics(i))
            
            with cfut.SlurmExecutor(True, keep_logs=True, additional_setup_lines=["#SBATCH -p upex -t 06:00:00"]) as executor:
                for results in executor.map(compute_statistics, range(16) ):
                    mean_pixelcell.append(results[0][0])
                    var_pixelcell.append(results[0][1])
                    var_pixelrow.append(results[0][2])
                    
        else:
            print('Not matching method')
            return

        if method == 'dask_distributed':
            print('closing dask cluster')
            client.close()
            cluster.close()


        mean_pixelcell = xr.concat(mean_pixelcell,dim='module').transpose('pulseId', 'module', 'dim_0', 'dim_1')
        var_pixelcell = xr.concat(var_pixelcell,dim='module').transpose('pulseId', 'module', 'dim_0', 'dim_1')
        var_pixelrow = xr.concat(var_pixelrow,dim='module').transpose('row', 'module', 'dim_0', 'dim_1')

        mean_pixelcell.to_netcdf(out_dir+'/mean_pixelcell.nc')
        var_pixelcell.to_netcdf(out_dir+'/var_pixelcell.nc')
        var_pixelrow.to_netcdf(out_dir+'/var_pixelrow.nc')
        
    return mean_pixelcell, var_pixelcell, var_pixelrow
  