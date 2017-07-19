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

def lnprior(params):
    print "todo"
def lnlike(params, data, emulist):
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

if __name__ == "__main__":
    fit_box(0)
