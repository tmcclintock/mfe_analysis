"""
Here I built an emulator and then fit for cosmology
on one of the test boxes. I can do this both with
and without using realizations of the MF,
demonstrating the effect of having a perfect prediction
or not.
"""
import numpy as np
import tinker_mass_function as TMF
import sys, os, emulator
import cosmocalc as cc
from setup_routines import *
import emcee
usegeorge = True

scale_factors, redshifts = get_sf_and_redshifts()
volume = get_volume()
N_z = len(scale_factors)
N_cosmos = 39
colors = get_colors()
building_cosmos = get_building_cosmos()
testbox_cosmos = get_testbox_cosmos()
name = 'dfg'
mean_models, err_models, R = get_rotated_fits(name)

#First train the emulators
emu_list = train(building_cosmos, mean_models, err_models, use_george=usegeorge)

def get_cosmo_dict(cosmo):
    """
    Return a cosmo dictionary for use in cosmocalc
    """
    ombh2, omch2, w0, ns, H0, Neff, sigma8 = cosmo
    h = H0/100.
    Ob,Om = ombh2/(h**2), ombh2/(h**2)+omch2/(h**2)
    cosmo_dict = {"om":Om, "ob":Ob, "ol":1-Om, "ok":0.0, "h":h, 
                  "s8":sigma8, "ns":ns, "w0":w0, "wa":0.0}
    return cosmo_dict

def lnprior(params):
    #Ombh2 Omch2 w0 ns H0 Neff sigma8
    ombh2, omch2, w0, ns, H0, Neff, sigma8 = params
    if ombh2 < 0 or omch2 < 0 or w0 > 0 or ns < 0 or H0 < 0 or Neff < 0 or sigma8 < 0: return -np.inf #enforce these signs
    return 0

def lnlike(params, data, emu_list, zs, sfs):
    cosmo_dict = get_cosmo_dict(params)
    emu_model = predict_parameters(params, emu_list, mean_models, R=R, use_george=usegeorge)
    lM_bins_all, N_data, icovs = data
    LL = 0
    TMF_model = TMF.tinker_mass_function(cosmo_dict, zs[0])
    for j in range(len(zs)):
        lM_bins = lM_bins_all[j]
        N = N_data[j]
        icov = icovs[j]
        TMF_model.redshift = zs[j]
        TMF_model.scale_factor = sfs[j]
        TMF_model.build_splines()
        d,e,f,g,B = get_params(emu_model, sfs[j])
        TMF_model.set_parameters(d,e,f,g,B)
        N_emu = volume * TMF_model.n_in_bins(lM_bins)
        X = N - N_emu
        LL += np.dot(X, np.dot(icov, X))
    return LL

def lnprob(params, data, emu_list, zs, sfs):
    lpr = lnprior(params)
    if not np.isfinite(lpr): return -np.inf
    ret = lpr + lnlike(params, data, emu_list, zs, sfs)
    if np.isnan(ret): return -np.inf
    return ret

def do_minimize(params, data, emu_list, zs, sfs, truth):
    import scipy.optimize as op
    lnprob_args = (data, emu_list, zs, sfs)
    nll = lambda *args: -lnprob(*args)
    print "Finding best fit"    
    result = op.minimize(nll, params, args=lnprob_args, tol=1e-4)
    np.savetxt("txt_files/bf_cosmo.txt", result['x'])
    print result
    print "truth = ", truth
    return

def do_mcmc(params, data, emu_list, zs, sfs, truth):
    import emcee
    lnprob_args = (data, emu_list, zs, sfs)
    ndim = len(params)
    nwalkers = ndim*2+2
    nsteps = 2000
    nburn = 200
    pos = [params + 1e-3*np.random.randn(ndim) for k in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=lnprob_args, threads=4)
    sampler.run_mcmc(pos, nsteps)
    likes = sampler.flatlnprobability
    fullchain = sampler.flatchain
    np.savetxt("txt_files/chains/emucosmo_chain.txt", fullchain)
    chain = fullchain[nburn*nwalkers:]
    out = np.array([np.means(chain, 0), np.std(chain, 0)]).T
    np.savetxt("analysis_output.txt", out)
    return

def fit_box(box):
    #Fit this particular box to find the cosomology
    #Grab the data
    lM_bins = []
    N_data = []
    icovs = []
    for i in range(N_z):
        lM_bins_i, lM, N_i, err, cov_i = get_testbox_data(box,i)
        lM_bins.append(lM_bins_i)
        N_data.append(N_i)
        icovs.append(np.linalg.inv(cov_i))
    #create the sampler
    #set the initial walker locations to be scattered around the truth
    truth = testbox_cosmos[box]
    #do_minimize(truth, [lM_bins, N_data, icovs], emu_list, redshifts, scale_factors, truth)
    do_mcmc(truth, [lM_bins, N_data, icovs], emu_list, redshifts, scale_factors, truth)

if __name__ == "__main__":
    fit_box(0)
