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


def get_saxsxpcs(arr, ai, index, out_dir,method, chunk_length):
       
    def get_Iq(_data):
        #function that does the azimuthal integration on a single image    

        mask = np.isnan(_data.flatten()).astype('bool')
        q,_I = ai.integrate1d(_data.flatten(),300,mask=mask,unit="q_A^-1",dummy=np.nan)

        return _I

    #chunks of trainIds to distribute
    trainId_chunks = np.array_split(arr.trainId.data,len(arr.trainId.data)//chunk_length)

    #print(str(len(trainId_chunks))+' chunks')

    def get_saxsxpcs_of_chunk(tid_chunk):

        #print(tid_chunk,flush=True)

        SAXS= []
        TTCF = []
        TTCF_off = []

        first_train_of_chunk = tid_chunk[0]
        position_of_first_train = np.where(arr.trainId.data == first_train_of_chunk)[0][0]
        if position_of_first_train != 0:
            previous_train_trainId = arr.trainId.data[position_of_first_train-1]
        else:
            previous_train_trainId = []

        data_chunk = arr.sel(trainId=np.append(tid_chunk,previous_train_trainId))
        data_chunk.data = data_chunk.data.rechunk((1,176,16,512,128))

        if previous_train_trainId:
            previous_train = data_chunk.sel(trainId = previous_train_trainId)
        else:
            previous_train = None

        #get lazy SAXS of this chunk
        SAXS = da.apply_along_axis(get_Iq,1,data_chunk.sel(trainId=tid_chunk).data.reshape((-1,16*512*128)),dtype='float32',shape=(300,))#.compute()


        #get kazy XPCS of this chunk
        for tid in tid_chunk:

            #AGIPD data of this train
            data_train = data_chunk.sel(trainId=tid)

            _TTCF = []
            _TTCF_off = []

            #sequentially for each q-bin the XPCS data
            for no_q_bin in range(len(index)):
                #print('q-bin '+str(no_q_bin)+' started')

                #data of the q-bin
                roi_data = data_train.data.reshape(-1,16*512*128)[:,index[no_q_bin].flatten()]
                int_mean = da.nanmean(roi_data,-1) # mean intensity of each shot

                #calculate two-time correlation function
                _TTCF_train = da.nanmean(da.einsum('ib,ob->bio', roi_data, roi_data),0)/da.einsum('i,o->io',int_mean,int_mean)

                #calculate two-time correlation function between actual and previous train
                if previous_train is not None:
                    prev_roi_data = previous_train.data.reshape(-1,16*512*128)[:,index[no_q_bin].flatten()]
                    prev_int_mean = da.nanmean(prev_roi_data,-1) # mean intensity of each shot
                    _prev_TTCF_train = da.nanmean(da.einsum('ib,ob->bio', roi_data, prev_roi_data),0)/da.einsum('i,o->io',int_mean,prev_int_mean)
                else:
                    _prev_TTCF_train = np.ones_like(_TTCF_train)

                _TTCF.append(_TTCF_train)
                _TTCF_off.append(_prev_TTCF_train)

            TTCF.append(da.stack(_TTCF))
            TTCF_off.append(da.stack(_TTCF_off))

            previous_train = data_train

        return SAXS.reshape((-1,len(roi_data),300)), da.stack(TTCF), da.stack(TTCF_off)
    
    def compute_saxsxpcs(i):
        with ProgressBar(dt=10):
            out = da.compute(get_saxsxpcs_of_chunk(i))
        return out
   

    SAXS =  [] 
    TTCF =  []  
    TTCF_off =  [] 
    
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
            print(cluster.status)
            sleep(1.0)

        client = Client(cluster)
        print("Created dask client:", client)
        
        # link to dask dashbord
        print(cluster.dashboard_link)

    if 'dask' in method:

        for tid in trainId_chunks:
            print(str(np.round(100*(tid[0]-np.min(arr.trainId.data))/
                           (np.max(arr.trainId.data)-np.min(arr.trainId.data))))+' % of SAXS & XPCS processing finished')
            results = compute_saxsxpcs(tid)
            SAXS.append(results[0][0])
            TTCF.append(results[0][1])
            TTCF_off.append(results[0][2])   

    elif method == 'cfut':

        with cfut.SlurmExecutor(True, keep_logs=True, additional_setup_lines=["#SBATCH -p upex -t 06:00:00"]) as executor:
            for results in executor.map(compute_saxsxpcs, trainId_chunks):
                SAXS.append(results[0][0])
                TTCF.append(results[0][1])
                TTCF_off.append(results[0][2])   
                
    if method == 'dask_distributed':
        client.close()
        cluster.close()

    SAXS = np.vstack(SAXS)
    TTCF = np.vstack(TTCF)
    TTCF_off = np.vstack(TTCF_off)

    TTCF = xr.DataArray(
        data = TTCF,
        dims = ['trainId','qBin','pulse_1','pulse_2'],
        coords = dict(
            trainId = arr.trainId.data,
            qBin = np.arange(len(index)),
            pulse_1 = arr.pulseId.data,
            pulse_2 = arr.pulseId.data))
    TTCF.to_netcdf(out_dir+'TTCF.nc')  


    TTCF_off = xr.DataArray(
        data = TTCF_off,
        dims = ['trainId','qBin','pulse_1','pulse_2'],
        coords = dict(
            trainId = arr.trainId.data,
            qBin = np.arange(len(index)),
            pulse_1 = arr.pulseId.data,
            pulse_2 = arr.pulseId.data))
    TTCF_off.to_netcdf(out_dir+'TTCF_off.nc')  

    q = np.load(out_dir+'/q.npy')

    SAXS = xr.DataArray(
        data = SAXS,
        dims = ['trainId','pulseId','q'],
        coords = dict(
            trainId = arr.trainId.data,
            pulseId = arr.pulseId.data,
            q = q))
    SAXS.to_netcdf(out_dir+'SAXS.nc') 
    