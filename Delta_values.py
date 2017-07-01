"""
Here we run the emulator on each LOO box and get the Delta
values not at every mass but at nu values.
"""
import numpy as np
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', size=20)
import tinker_mass_function as TMF
import sys, os, emulator
import cosmocalc as cc
from setup_routines import *

usegeorge = False

xlabel = r"$\nu$"
y0label = r"$\Delta=\frac{N}{N_{emu}bG}-1-\frac{1}{bG}$"
y0label = ''
#y1label = r"$\%\ {\rm Diff}=\frac{N}{N_{emu}}$"
y1label = r"$\Delta N/N_{emu}$"

scale_factors, redshifts = get_sf_and_redshifts()
volume = get_volume()
N_z = len(scale_factors)
N_cosmos = 39
colors = get_colors()
building_cosmos = get_building_cosmos()
name = 'dfg'
mean_models, err_models, R = get_rotated_fits(name)

def get_bG(a, Masses):
    return cc.growth_function(a)*np.array([cc.tinker2010_bias(Mi, a, 200) for Mi in Masses])

def get_nu(a, Masses):
    return 1.686/np.array([cc.sigmaMtophat(Mi, a) for Mi in Masses])

for i in range(0,1):
    fig, axarr = plt.subplots(2, sharex=True)
    cosmo_dict = get_cosmo_dict(i)

    test_cosmo = building_cosmos[i]
    test_data  = mean_models[i]
    test_err   = err_models[i]
    training_cosmos = np.delete(building_cosmos, i, 0)
    training_data   = np.delete(mean_models, i, 0)
    training_errs   = np.delete(err_models, i, 0)

    #Train the emulators
    emu_list = train(training_cosmos, training_data, training_errs)
    emu_model = predict_parameters(test_cosmo, emu_list, training_data, R=R, use_george=usegeorge)

    for j in range(N_z):
        lM_bins, lM, N, err, cov = get_sim_data(i,j)

        #Get emulated curves
        TMF_model = TMF.tinker_mass_function(cosmo_dict, redshifts[j])
        d,e,f,g,B = get_params(emu_model, scale_factors[j])
        TMF_model.set_parameters(d,e,f,g,B)
        N_bf = volume * TMF_model.n_in_bins(lM_bins)
        bG = get_bG(scale_factors[j], 10**lM)
        pd = (N-N_bf)/N_bf
        pde  = err/N_bf
        Delta = pd/bG
        Deltae = pde/bG
        nu = get_nu(scale_factors[j], 10**lM)        
        
        axarr[0].errorbar(nu, Delta, Deltae, c=colors[j], marker='.',ls='')
        axarr[1].errorbar(nu, pd, pde, c=colors[j], marker='.',ls='')
    axarr[0].axhline(0, c='k', ls='-', zorder=-1)
    axarr[1].axhline(0, c='k', ls='-', zorder=-1)

    axarr[1].set_xlabel(xlabel)
    axarr[0].set_ylabel(y0label)
    axarr[1].set_ylabel(y1label)
    axarr[1].set_ylim(-.18, .18)
    plt.subplots_adjust(bottom=0.15, left=0.19, hspace=0.3)
    plt.show()
