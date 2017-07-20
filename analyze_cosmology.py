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

def lnlike(params, data, emulist):
    lM_bins, N_Data, covs = data
    print "todo"
def lnprob(params, data, emulist):
    print "todo"

def fit_box(box):
    #Fit this particular box to find the cosomology
    #Grab the data
    lM_bins = []
    N_data = []
    covs = []
    for i in range(N_z):
        lM_bins_i, lM, N_i, err, cov_i = get_testbox_data(box,i)
        lM_bins.append(lM_bins_i)
        N_data.append(N_i)
        covs.append(cov_i)
    #create the sampler
    #set the initial walker locations to be scattered around the truth
    truth = testbox_cosmos[box]
    print truth
    print lnprior(truth)
    print get_cosmo_dict(truth)

if __name__ == "__main__":
    fit_box(0)
