
from xpcs_statistics import *
from xpcs_calculations import *
from xpcs_functions import *
from xpcs_analysis import *
    
def get_saxs_xpcs(run_no = None,
              proposal = None, 
              out_dir = None,
                  
              method = 'cfut',
              chunk_length = 200,
                  
              mask = None,
              photonize = False,
              adu_per_photon = 1,
              selection_pulseIds = None,
              photon_energy = None,
                  
              geom = None,
              sdd = None,
              refine_beamcenter = True, # find beamcenter
              px = 605, #beamcenter in pixel coordinates
              py = 690,
              q_min = 0.015, #smallest and largest q for beamcenter fit
              q_max = 0.05,
              qq = np.arange(0.0095,0.0175,0.0015), #range of q-bins to do the XPCS analysis on
                  
              mask_outlier = 'som_full_pixel',
            
              calculate_saxsxpcs = True, # do pulse resolved SAXS and MHz XPCS calculations
                  
              xpcs_reduction = False,
              motor_moving = 'MID_SAE_FSSS/MOTOR/SCANNERX',
              motor_range = None,
                  
              fit_result = False
             ):


    ## creats output folder, opens run
    out_dir, run = open_directories(run_no, proposal, out_dir)
    
    ## get agipd data array (lazy xarray), mask pixel from file mask and cal pipeline mask, photonize&slice if needed
    arr = get_agipd(run, mask, photonize, photon_energy, adu_per_photon, selection_pulseIds)
    
    ## get averages and variances
    mean_pixelcell, var_pixelcell, var_pixelrow = get_statistics(run, arr, out_dir, method)
    mean_image = mean_pixelcell.mean('pulseId',skipna=True)
    
    ## get geometry
    agipd, ai, geom = get_geometry(run, px, py, sdd, photon_energy, geom)
    
    ## get the regions of interest (and refine beam center/update ai)
    ai, index = get_rois(mean_image, ai, geom, out_dir,
             refine_beamcenter, px, py, q_min, q_max, qq)
    
    ## mask additionally outliers based on statistical analysis
    if mask_outlier == 'som_full_pixel':
        print('remove outlier by std of mean, masking full pixel')
        arr = outlier_masking(arr, ai, qq, index, mean_pixelcell, out_dir)
    
    ## calculate TTCF and SAXS data
    if calculate_saxsxpcs:
        get_saxsxpcs(arr, ai, index, out_dir, method, chunk_length) 
        
    ## reduce and average the TTCF data
    if xpcs_reduction:
        average_xpcs(run,out_dir,motor_moving,motor_range)
        
    ## fit reduced result
    if fit_result:
        cc = np.load(out_dir+'cc.npy')
        fit_xpcs(cc,model = 'exp',beta = 'local', dr=True, plot=True) # fit with single exponential / brownian motion
    
    