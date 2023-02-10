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

from xpcs_analysis import *

def open_directories(run_no, proposal, out_dir):
    
    if isinstance(run_no,int):
        if proposal is not None:
            run = ex.open_run(proposal=proposal, run=run_no,data='all')
            #run.info()
            if out_dir is None:
                #print(run.files[0].filename)
                if 'RAW' in run.files[0].filename:
                    out_dir = run.files[0].filename.replace('/raw/','/scratch/JM/xpcs/')[:-3].split('RAW')[0]+"/"
                elif 'CORR' in run.files[0].filename:
                    out_dir = run.files[0].filename.replace('/proc/','/scratch/JM/xpcs/')[:-3].split('CORR')[0]+"/"
                else:
                    print('unknown path')
            print('output saved in '+out_dir)
        else:
            print('please provide a proposal number')
            return
    elif isinstance(run_no,str):
        if out_dir is None:
            print('Please provide a ouput directory (out_dir=)')
            return
        else:
            run = ex.RunDirectory(run_no)
            
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    return out_dir, run


def get_agipd(run, mask = None, photonize = False, photon_energy = None, adu_per_photon = 1, selection_pulseIds = None):
    from extra_data.components import AGIPD1M
    agp = AGIPD1M(run,min_modules=16)
    arr = agp.get_dask_array('image.data')

    # Apply mask from calibration pipeline
    _mask = agp.get_dask_array('image.mask')
    arr.data[(_mask.data > 0)] = np.nan

    # Apply manual mask
    if mask is not None:
        mask = np.load(mask).astype('bool')
        arr = arr.where(~mask[:,None,:,:])

    arr = arr.unstack('train_pulse').transpose('trainId', 'pulseId', 'module', 'dim_0', 'dim_1').astype('float32')
    
    # remove unwanted storage cells
    if selection_pulseIds is not None:
        arr = arr.sel(pulseId=selection_pulseIds)

    
    # Photonize  !!!Only do this if the PROC data is not photonized!!!
    if photonize:
        if photon_energy is None:
            wavelength = run["SA2_XTD1_XGM/XGM/DOOCS",'pulseEnergy.wavelengthUsed'].as_single_value(rtol=1e-2)
            from scipy.constants import h,c,e
            photon_energy_kev = (h * c / e) / (wavelength * 1e-6)
            print('loaded photon energy is '+str(photon_energy_kev))
            
        arr.data /= adu_per_photon

        arr.data = da.round(arr.data/photon_energy_kev)
        arr.data[arr.data<0] = 0
        
    return arr
      
    
def get_geometry(run, px, py, sdd, photon_energy, geom):
    
    if geom is None:
        print('loading positions of AGIPD quadrants')
        def load_AGIPD_Motors(run =run):

            motVal=[]
            for name in ( ['q1m1','q2m1','q3m1','q4m1','q1m2','q2m2','q3m2','q4m2']):
                prop = f'{name}ActualPosition'
                try:
                    arr = run['MID_AGIPD_MOTION/MDL/DOWNSAMPLER',prop].xarray().data[0]
                    motVal.append(arr)
                except:
                    motVal.append(0)

            return motVal

        q1m1, q2m1, q3m1, q4m1, q1m2, q2m2, q3m2, q4m2 = load_AGIPD_Motors(run)

        q1_x = -542 + 0*q1m1
        q1_y = 660 + q1m1/(-0.2)
        q2_x = -608 + 0*q2m1
        q2_y = -35 + q2m1/0.2
        q3_x = 534 + 0*q3m1
        q3_y = -221 + q3m1/0.2
        q4_x = 588 + 0*q4m1
        q4_y = 474 + q4m1/(-0.2)    
        quad_pos = [(q1_x, q1_y),
                       (q2_x, q2_y),
                       (q3_x, q3_y),
                       (q4_x, q4_y)]  

        geom = AGIPD_1MGeometry.from_quad_positions(quad_pos=quad_pos)
    else:
        print('loading AGIPD geometry from file '+str(geom))
        geom = AGIPD_1MGeometry.from_crystfel_geom(geom)

    #If not provided, read sample detector distance and X-ray wavelenght form the files
    if sdd is None:
        sdd = run['MID_AGIPD_MOTION/MDL/DOWNSAMPLER','t4EncoderPosition'].ndarray()[0]
    if photon_energy is None:
        wavelength = run["SA2_XTD1_XGM/XGM/DOOCS",'pulseEnergy.wavelengthUsed'].as_single_value(rtol=1e-2)*1e-9
        print('loaded wavelength is '+str(wavelength))

    #setup pyFAI scattering geometry
    agipd = pyFAI.detectors.Detector(200e-6,200e-6)
    agipd.aliases = ["AGIPD1M"]
    agipd.shape = (16*512,128)
    agipd.mask = np.zeros((16*512,128))
    agipd.IS_CONTINIOUS = False
    agipd.IS_FLAT = True
    agipd.set_pixel_corners(geom.to_distortion_array())

    ai = pyFAI.AzimuthalIntegrator(detector = agipd)
    ai.setFit2D(sdd,px,py)
    ai.wavelength = wavelength
    
    return agipd, ai, geom


def get_rois(mean_image, ai, geom, out_dir,
             refine_beamcenter, px, py, q_min, q_max, qq):
    ## get the regions of interest for the XPCS calculation
    ## if requested, find optimized beamcenter first

    import scipy.optimize

    param = [px,py,ai.dist*1000] # beamcenter in x and y pixel, sample detector distance in mm

    qA = ai.array_from_unit(unit='q_A^-1').reshape(16,512,128) #this calculates the q-value for each pixel using our previous geometry definition
    index = (qA>q_min) & (qA<q_max)

    plt.figure(figsize=(9,9))
    plt.imshow(geom.position_modules_fast(mean_image.data)[0],origin='lower',cmap='jet',norm=mpl.colors.LogNorm())
    plt.imshow(geom.position_modules_fast(index)[0],origin='lower',cmap='jet',norm=mpl.colors.LogNorm(),alpha=0.5)
    plt.savefig(out_dir+'/refinement_range.jpg',dpi=300)
     
    def get_SAXS_var(xy,dist,_data,q_min,q_max,plot=False,title = " "):
        #calculates the I(q) in 8 cones and returns the mean variance between the slices

        number_of_cones = 8
        cone_width = 360//number_of_cones
        _mask = np.isnan(_data.reshape(16*512,128)).astype('bool')

        x,y = xy
        #print(x,y)
        if plot:
            plt.figure()

        ai.setFit2D(dist,x,y)
        _I = []
        for i in range(number_of_cones):
            q,I = ai.integrate1d(_data.reshape(16*512,128),300,radial_range=(q_min,q_max),azimuth_range = (i*cone_width,(i+1)*cone_width),mask=_mask.reshape(16*512,128),unit="q_A^-1",dummy=np.nan)
            if plot:
                plt.plot(q,I)
            _I.append(I)

        if plot:
            plt.vlines(q_min,np.nanmin(I),np.nanmax(I))
            plt.vlines(q_max,np.nanmin(I),np.nanmax(I))
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel(r'$q$ / [$\AA^{-1}$]',fontsize='x-large')
            plt.ylabel(r'$I(q)$ / [ph/px]',fontsize='x-large')
            plt.title(title)
            plt.savefig(out_dir+'/'+title+'_saxs.jpg',dpi=300)

        return np.nanmean(np.nanvar(np.log(np.array(_I)),0),0)
    
    get_SAXS_var([param[0],param[1]],param[2],mean_image.data,q_min,q_max,plot=True,title='Without beam center refinement')

   
    if refine_beamcenter:
        #fit beamcenter on mean image

        opt_result = scipy.optimize.minimize(get_SAXS_var,
                                                 [param[0],param[1]],
                                                 args=(param[2],mean_image.data,q_min,q_max,False),
                                                 method="Nelder-Mead",
                                                 bounds=[(param[0]-50,param[1]-50),(param[0]+50,param[1]+50)],
                                                 options={ "maxfev": 100, })

        param = [opt_result.x[0],opt_result.x[1],ai.dist*1000]
        get_SAXS_var([param[0],param[1]],param[2],mean_image.data,q_min,q_max,plot=True,title='After beam center refinement')

        ai.setFit2D(param[2],param[0],param[1])
        
    print('Scattering geometry:')
    print(ai)
    ai.save(out_dir+'/scattering_geometry.ai')
    
    _mask = np.isnan(mean_image.data).astype('bool')
    q,I = ai.integrate1d(mean_image.data.reshape(16*512,128),300,mask=_mask.reshape(16*512,128),unit="q_A^-1",dummy=np.nan)
    plt.figure()
    plt.plot(q,I,'k')
    plt.title('q-bins for XPCS analysis')
    
    qA = ai.array_from_unit(unit='q_A^-1').reshape(16,512,128)
    index = [] #indices of pixels which correspond to a certain qBin
    for i in range(len(qq)-1):
        index.append((qA>qq[i]) & (qA<qq[i+1])) # define pixel indices of ROIs
        print(str(np.sum(index[-1]))+' pixel in q-bin # '+str(i)) # number of pixel in this q-bin

    for i in range(len(qq)-1):
        plt.axvspan(qq[i], qq[i+1], facecolor=mpl.cm.jet(i / float(np.size(qq))), alpha=0.5)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$q$ / [$\AA^{-1}$]',fontsize='x-large')
    plt.ylabel(r'$I(q)$ / [ph/px]',fontsize='x-large')
    plt.savefig(out_dir+'/q_bins.jpg',dpi=300)
    
    np.save(out_dir+'/q.npy',q) # q range of SAXS azimuthal integration
    np.save(out_dir+'/qq.npy',qq) ## q-bins of XPCS analysis
    
    return ai, index # return azimuthalIntegration object of pyFAI (scattering geometry definition) and list of pixels in each qBin (index)


def outlier_masking(arr, ai, qq, index, mean_pixelcell, out_dir):
    #mask outlier based on statistical analysis
    
    std_of_mean = np.sqrt((mean_pixelcell**2).mean('pulseId',skipna=True)).data
    
    qA = ai.array_from_unit(unit='q_A^-1').reshape(16,512,128)
    
    plt.figure()  
    _mask = np.isnan(mean_pixelcell.mean('pulseId',skipna=True).data).astype('bool')
    q,I = ai.integrate1d(mean_pixelcell.mean('pulseId',skipna=True).data.reshape(16*512,128),300,mask=_mask.reshape(16*512,128),unit="q_A^-1",dummy=np.nan)
    plt.plot(q,I,'r',label='before outlier removal')    

    ppp = np.zeros((16,512,128))
    for i in range(len(q)-1):
        _ind = (qA>q[i]) & (qA<q[i+1])
        ppp[_ind] = std_of_mean[_ind] / np.nanmedian(std_of_mean[_ind])

    mask_std_low = ppp<0.75
    print(str(np.sum(mask_std_low))+' pixel masked because std_of_mean too low compared to median of q-bin')
    mask_std_high = ppp>1.75
    print(str(np.sum(mask_std_high))+' pixel masked because std_of_mean too high compared to median of q-bin')

    outlier_mask = mask_std_low+mask_std_high

    #apply mask on mean image and full data array
    mean_pixelcell_masked = mean_pixelcell.where(~outlier_mask[None,:,:,:])
    mean_image_masked = mean_pixelcell_masked.mean('pulseId',skipna=True)

    arr = arr.where(~outlier_mask[None,None,:,:,:])
    
    _mask = np.isnan(mean_image_masked.data).astype('bool')
    q,I = ai.integrate1d(mean_image_masked.data.reshape(16*512,128),300,mask=_mask.reshape(16*512,128),unit="q_A^-1",dummy=np.nan)
    plt.plot(q,I,'k',label='after outlier removal')    

    for i in range(len(qq)-1):
        plt.axvspan(qq[i], qq[i+1], facecolor=mpl.cm.jet(i / float(np.size(qq))), alpha=0.5)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$q$ / [$\AA^{-1}$]',fontsize='x-large')
    plt.ylabel(r'$I(q)$ / [ph/px]',fontsize='x-large')
    plt.legend()
    plt.savefig(out_dir+'/saxs.jpg',dpi=300)
    
    plt.figure()
    plt.hist(ppp.flatten(),bins=np.logspace(-1,3,1000))
    plt.vlines(0.75,1,10000)
    plt.vlines(1.75,1,10000)
    plt.yscale('log')
    plt.xscale('log')
    plt.title('Distribution of std_of_mean and outlier thershold')
    plt.savefig(out_dir+'/outlier_distribution.jpg',dpi=300)
    
    return arr

    
def average_xpcs(run,out_dir,motor_moving,motor_range):
            
    SAXS = xr.open_dataarray(out_dir+'/SAXS.nc')
    TTCF = xr.open_dataarray(out_dir+'/TTCF.nc')
    TTCF_off = xr.open_dataarray(out_dir+'/TTCF_off.nc')

    q = np.load(out_dir+'q.npy')   # q points of SAXS analysis
    qq = np.load(out_dir+'qq.npy') # q point of XPCS analysis


    _run = run.select([
        ('MID_DET_AGIPD1M-1/DET/*CH0:xtdf', '*'),
        (motor_moving, '*'),
    ], require_all=True) # require_all makes sure that only trains are used where all of the selected data sources are saved.

    motor = _run.get_array(motor_moving ,'actualPosition')

    good_trains = motor.trainId[(motor.data>motor_range[0]) & (motor.data<motor_range[1])]

    fig, ax = plt.subplots()
    ax.set_xlabel('trainId')
    ax.set_ylabel('motor position '+motor_moving, color='r')
    ax.plot(motor.trainId,motor.data,'r')
    ax.tick_params(axis='y', labelcolor='r')
    ax2 = ax.twinx() 
    ax2.plot(SAXS.trainId,SAXS.isel(q=range(15)).mean(('pulseId','q'),skipna=True).data,'.-b')
    ax2.set_ylabel('mean intensity low q SAXS', color='b')  
    ax.plot(motor.sel(trainId=good_trains).trainId,motor.sel(trainId=good_trains).data,'.g',label='good motor positions')
    plt.legend()
    plt.savefig(out_dir+'/spots_on_sample.jpg',dpi=300)


    plt.figure()
    for i in SAXS.pulseId:
        plt.plot(q,SAXS.sel(pulseId=i).mean('trainId',skipna=True),label=i.data)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.xlabel(r'$q$ / [$\AA^{-1}$]',fontsize='x-large')
    plt.ylabel(r'$I(q)$ / [ph/px]',fontsize='x-large')
    for i in range(len(qq)-1):
        plt.axvspan(qq[i], qq[i+1], facecolor=mpl.cm.jet(i / float(np.size(qq))), alpha=0.5)
    plt.title('mean SAXS for each pulseId')
    plt.tight_layout()
    plt.savefig(out_dir+'/mean_saxs.png',dpi=300)

    reduced_TTCF = TTCF.sel(trainId=good_trains).mean('trainId',skipna=True) - TTCF_off.sel(trainId=good_trains).mean('trainId',skipna=True)
    reduced_TTCF.to_netcdf(out_dir+'/TTCF_corrected.nc')

    plt.figure(figsize=(9,9))
    q_bin_no = 1
    plt.imshow(reduced_TTCF.sel(qBin=q_bin_no),vmin=0,vmax=0.05,cmap='jet')
    plt.title('two-time correlation function of q-bin '+str(q_bin_no))
    plt.xlabel('pulse 1')
    plt.ylabel('pulse 2')
    plt.savefig(out_dir+'/mean_ttcf.png',dpi=300)


    def get_cf(darr,skip_last=10):
        return np.array([np.nanmean(np.diagonal(darr,offset=i)) for i in range(1,len(darr)-skip_last)])

    CF_corr = []
    for no_q_bin in range(len(qq)-1):
        CF_corr.append(get_cf(reduced_TTCF.sel(qBin=no_q_bin)))
        
    reprate = run['MID_EXP_AGIPD1M1/MDL/FPGA_COMP','bunchStructure.repetitionRate'].as_single_value()

    #plt.figure()
    #for no_q_bin in range(len(qq)-1):
    #    plt.plot(np.arange(1,len(CF_corr[no_q_bin])+1)/reprate*1000,CF_corr[no_q_bin],'-o',label='bin '+str(no_q_bin))
    #plt.xscale('log')
    #plt.xlabel(r't / [ns]',fontsize='x-large')
    #plt.ylabel(r'g${}_2$(t)',fontsize='x-large')
    #plt.legend()
    #plt.title('average g2 over all pulses')
    #plt.ylim([-0.01,0.1])
    #plt.tight_layout()
    #plt.savefig(out_dir+'/mean_g2.png',dpi=300)
  
    #convert ot pyXPCS format
    cc = np.ones((len(CF_corr[0])+1,len(qq)))
    cc[0,1:] = np.array([(qq[i+1]+qq[i])/2 for i in range(len(qq)-1)])
    cc[1:,0] = np.arange(1,len(CF_corr[0])+1)/reprate/1e6
    cc[1:,1:] = np.array(CF_corr).transpose()
    
    np.save(out_dir+'cc.npy',cc)
    
    plot_xpcs(cc)
    
