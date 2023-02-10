import os, sys

from matplotlib.colors import colorConverter
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib as mpl

import numpy as np

import itertools

import scipy
import scipy.stats



def plot_xpcs(data,fit=None,label=None,no=None):
    
    plt.figure(figsize=(6,5))
    cmap = mpl.cm.jet
    
    labels = r"%.3f"
    
    if label is not None:
        names=label
    else:
        names = (' ',)*len(data)
        
    if no is not None:
        if isinstance(no,int):
            no1=no
            no2=no
        elif isinstance(no,tuple):
            no1=no[0]
            no2=no[1]
    
    if isinstance(data,str):
        print('blubb')
    else:
        if isinstance(data,tuple):

            for i in range(len(data)):
                cc = data[i]
                
                if no is not None:
                    cc = np.concatenate((cc[:,0:1],cc[:,no1+1:no2+2]),axis=1)
                    
                if fit is not None:
                    cc_fit = fit[i]
                    if no is not None:
                        cc_fit = np.concatenate((cc_fit[:,0:1],cc_fit[:,no1+1:no2+2]),axis=1)
                    
                marker = 'o'
                for j in range(np.size(cc[1,:])-1):
                    if fit is not None:
                        plt.plot(cc[1:,0],cc[1:,j+1],label=names[i]+labels % (cc[0,j+1]), ms=4, marker=marker, lw=.8, color=cmap(j / float(np.size(cc[1,:])-1)))
                        plt.plot(cc_fit[1:,0],cc_fit[1:,j+1], color=cmap(j / float(np.size(cc[1,:])-1)))
                    else:
                        plt.plot(cc[1:,0],cc[1:,j+1],label=names[i]+labels % (cc[0,j+1]),ms=4, marker=marker, c=cmap(j / float(np.size(cc[1,:])-1)))
  
        
        
        elif isinstance(data,np.ndarray):
            cc = data
            
            if no is not None:
                    cc = np.concatenate((cc[:,0:1],cc[:,no1+1:no2+2]),axis=1)
                    
            if fit is not None:
                cc_fit = fit
                if no is not None:
                    cc_fit = np.concatenate((cc_fit[:,0:1],cc_fit[:,no1+1:no2+2]),axis=1)
            
            for i in range(np.size(cc[1,:])-1):
                if fit is not None:
                    plt.plot(cc[1:,0],cc[1:,i+1], label=names[0]+labels % (cc[0,i+1]), ms=6,marker='o', 
                                lw=.8, c=cmap(i / float(np.size(cc[1,:])-1)),
                                mec='k', mew=.5,)
                    plt.plot(cc_fit[1:,0],cc_fit[1:,i+1], color=cmap(i / float(np.size(cc[1,:])-1)))
                else:
                    plt.plot(cc[1:,0],cc[1:,i+1],label=names[0]+labels % (cc[0,i+1]), ms=6,marker='o',
                                lw=.8, c=cmap(i / float(np.size(cc[1,:])-1)),
                                mec='k', mew=.5, alpha=.8)

    
    plt.xscale('log')
    plt.xlim(0.85*min(cc[1:,0]),1.15*max(cc[1:,0]))
    plt.xlabel('$t, s$',fontsize='x-large')
    plt.ylabel('$g{}_2(q,t)$',fontsize='x-large')
    plt.legend(loc=1, fontsize = 'small', frameon=False, title='q ($\AA^{-1}$)', bbox_to_anchor=[1.2,1])
    plt.grid(ls=':', c='gray', alpha=0.5)
    
    
def fit_xpcs(cc,model = 'exp',beta = 'local', dr=False, plot=False):
        
        
    q = cc[0,1:]
    t = cc[1:,0]
    cc_exp = cc[1:,1:]
    
    
    tt = np.tile(t,np.size(q))
    
        
    if model=='exp' and beta=='local':
        
        def fit_func(tt,*param): # g = beta * exp(-2*t*Gamma)+baseline, single exponential 
            _param = param[0]
            _param=np.reshape(param,(3,np.size(param)//3))
            Gamma = _param[0,:]
            baseline = _param[1,:]
            beta =_param[2,:]
            
            t = tt[0:np.size(tt)//np.size(Gamma)]
        
            core = np.outer(-2*t,Gamma)
            cc = beta[None,:]*np.exp(core)+baseline[None,:]
            return cc.ravel()
        
        baseline = np.ones(np.size(q))
        Gamma = 1/np.linspace(max(t)/10,min(t)*10,np.size(q))
        beta = 0.1*np.ones(np.size(q))
        
        p0 = np.append(Gamma,baseline)
        p0 = np.append(p0,beta) 
             
        para_fit, pcov = scipy.optimize.curve_fit(lambda tt, *p0 : fit_func(tt,p0),tt,cc_exp.ravel(),p0=p0)
        #para_fit, pcov = scipy.optimize.curve_fit(fit_func,tt,cc_exp.ravel(),p0=p0)
        cc_fit = fit_func(tt,para_fit)
        
        para_fit=np.reshape(para_fit,(3,np.size(para_fit)//3))
        Gamma = para_fit[0,:]
        baseline = para_fit[1,:]
        beta = para_fit[2,:]
        
        perr = np.sqrt(np.diag(pcov))
        perr=np.reshape(perr,(3,np.size(perr)//3))
        err_Gamma = perr[0,:]
        err_baseline = perr[1,:]
        err_beta = perr[2,:]
        
        #make dictionary of fit values
        para_fit = {'Gamma':Gamma,'baseline':baseline,'beta':beta}
        error_fit = {'Gamma':err_Gamma,'baseline':err_baseline,'beta':err_beta}
        
    elif model=='exp' and beta=='global':

        def fit_func(tt,*param): # g = beta * exp(-2*t*Gamma)+baseline, single exponential 
            param=param[0]
            beta = param[-1]
            _param=np.reshape(param[0:-1],(2,(np.size(param)-1)//2))
            Gamma = _param[0,:]
            baseline = _param[1,:]
            
            
            t = tt[0:np.size(tt)//np.size(Gamma)]
        
            core = np.outer(-2*t,Gamma)
            cc = beta[None,None]*np.exp(core)+baseline[None,:]
            return cc.ravel()
        
        baseline = np.ones(np.size(q))
        Gamma = 1/np.linspace(max(t)/10,min(t)*10,np.size(q))
        beta = 0.1
        
        p0 = np.append(Gamma,baseline)
        p0 = np.append(p0,beta)  
               
        para_fit, pcov = scipy.optimize.curve_fit(lambda tt, *p0 : fit_func(tt,p0),tt,cc_exp.ravel(),p0=p0)
        #para_fit, pcov = scipy.optimize.curve_fit(fit_func,tt,cc_exp.ravel(),p0=p0)
        cc_fit = fit_func(tt,para_fit)
        
        beta = para_fit[-1]
        para_fit=np.reshape(para_fit[0:-1],(2,(np.size(para_fit)-1)//2))
        Gamma = para_fit[0,:]
        baseline = para_fit[1,:]
        
        perr = np.sqrt(np.diag(pcov))
        err_beta = perr[-1]
        perr=np.reshape(perr[0:-1],(2,(np.size(perr)-1)//2))
        err_Gamma = perr[0,:]
        err_baseline = perr[1,:]
        
        #make dictionary of fit values
        para_fit= {'Gamma':Gamma,'baseline':baseline,'beta':beta}
        error_fit = {'Gamma':err_Gamma,'baseline':err_baseline,'beta':err_beta}
        
    elif model=='str' and beta=='local':
    
        print('to be done')
        
    elif model=='str' and beta=='global':
    
        print('to be done')
        
    else:
        
        print('requested model not found')
        
        
    #generate output correlation function
    #add t and q values in first line and column    
    cc_fit = cc_fit.reshape(np.size(t),np.size(q))
    cc_fit = np.vstack([q,cc_fit])
    tt = cc[:,0] 
    cc_fit = np.hstack([tt[np.newaxis, :].T,cc_fit])
    
    if plot==True:
        plot_xpcs(cc,fit=cc_fit)
        
        plt.figure()
        plt.errorbar(q,para_fit['beta'],error_fit['beta'],linestyle='None', marker='o', color='k')
        plt.ylabel(r'$\beta $',fontsize=13)
        plt.xlabel(r'$q / \mathrm{\AA}^{-1}$',fontsize=13)

        
    if dr == True:
        fit_dr(q,para_fit,error_fit,plot=plot)
        
 

    
    return cc_fit, para_fit, error_fit

def fit_dr(q,para_fit,error_fit,plot=False,fit='weightedaverage'):
    
    Gamma = para_fit["Gamma"]
    err_Gamma = error_fit["Gamma"]
    
    if fit=='curve_fit':
    
        # Calculate dispersion relation
        def fit_lin(x,a): 
            print(a*x*x)
            return a*x*x
        
        
        a_rough = (Gamma[1]-Gamma[0])/(q[1]**2-q[0]**2)
        popt,pcov=scipy.optimize.curve_fit(fit_lin,q,Gamma,p0=2*a_rough,sigma=err_Gamma,bounds=(0,100*a_rough)) 
        a=popt[0]; err_a=np.sqrt(pcov[0,0])
        
    elif fit=='weightedaverage':
        
        a = np.average(Gamma/q/q,weights=err_Gamma/q/q)
        err_a = np.std(Gamma/q/q)
    
        
    kB = 1.381e-23;           #J/K
    absT = 273.15 + 25;       #K
    XVisc = 1*1e-23; #1*1e-21;          #N*s*A^-2
    RH = kB*absT/6/np.pi/XVisc/a*1e10; #nm
    #    
    rr = np.arange(0.9*np.min(q*q),1.1*np.max(q*q),(1.1*np.max(q*q)-0.9*np.min(q*q))/100)
     
    para_fit['dr_a']=a
    para_fit['RH']=RH

  
    error_fit['dr_a']=err_a
    error_fit['RH']=RH*err_a/a
      #   
    plt.figure()
    #
    #left, bottom, width, height = [0.25, 0.6, 0.2, 0.2]
    #ax2 = fig.add_axes([left, bottom, width, height])
    #    
    plt.errorbar(q*q,Gamma,yerr=err_Gamma,linestyle='None', marker='o', color='k')
    plt.plot(rr,rr*a,label=r"$R_H = %.1f +- %.1f \ \mathrm{\AA}$" % (RH,RH*err_a/a))
    plt.legend(loc=4,fontsize=13,frameon=False)
    #  
   # ax2.plot(q,Gamma/((q**2)*a), 'or')
    #ax2.set_ylim([0.5,1.5])
    #    
    plt.ylabel(r'$\Gamma / \mathrm{s}^{-1}$',fontsize=13)
    plt.xlabel(r'$q^2 / \mathrm{\AA}^{-2}$',fontsize=13)
    plt.tight_layout()
    plt.show()

    
