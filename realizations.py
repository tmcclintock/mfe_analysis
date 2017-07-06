"""
Make a regular old testbox plot, but have many realizations.

NOTE: THIS DOESN'T WORK BECAUSE THE CONDITIONAL HAS SUPER LARGE ERRORS.
"""
import numpy as np
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', size=20)
import tinker_mass_function as TMF
import sys, os, emulator
import cosmocalc as cc
from setup_routines import *

usegeorge = True

xlabel  = r"$\log_{10}M\ [{\rm M_\odot}/h]$"
y0label = r"$N/[{\rm Gpc}^3\  \log_{10}{\rm M_\odot}/h]$"
y0label = r"$N/[{\rm Gpc}^3\  \log{\rm M}]$"
y1label = r"$\%\ {\rm Diff}$"
#y1label = r"$\frac{N-N_{emu}}{N_{emu}bG}$"

scale_factors, redshifts = get_sf_and_redshifts()
volume = get_volume()
N_z = len(scale_factors)
N_cosmos = 39
colors = get_colors()
building_cosmos = get_building_cosmos()
testbox_cosmos = get_testbox_cosmos()
name = 'dfg'
mean_models, err_models, R = get_rotated_fits(name)
N_realizations = 1

#First train the emulators
emu_list = train(building_cosmos, mean_models, err_models, use_george=usegeorge)

#Loop over test boxes and do everything
for i in range(0,1):
    fig, axarr = plt.subplots(2, sharex=True)
    test_cosmo = testbox_cosmos[i]
    cosmo_dict = get_testbox_cosmo_dict(i)

    #First plot the data
    for j in range(0, N_z):
        lM_bins, lM, N, err, cov = get_testbox_data(i,j)
        axarr[0].errorbar(lM, N, err, marker='.', ls='', c=colors[j], alpha=1.0, label=r"$z=%.1f$"%redshifts[j])

    for real in range(N_realizations):
        emu_model = get_realization(test_cosmo, emu_list, mean_models, R=R, use_george=usegeorge)
        #emu_model = predict_parameters(test_cosmo, emu_list, mean_models, R=R, use_george=usegeorge)
        print emu_model
        for j in range(0, N_z):
            lM_bins, lM, N, err, cov = get_testbox_data(i,j)
            #Get emulated curves
            TMF_model = TMF.tinker_mass_function(cosmo_dict, redshifts[j])
            d,e,f,g,B = get_params(emu_model, scale_factors[j])
            TMF_model.set_parameters(d,e,f,g,B)

            N_bf = volume * TMF_model.n_in_bins(lM_bins)
            axarr[0].plot(lM, N_bf, ls='--', c=colors[j], alpha=0.3)
            dN_N = (N-N_bf)/N_bf
            pd  = 100.*dN_N
            pde = 100.*err/N_bf
            axarr[1].errorbar(lM+0.02*j, pd, pde, marker='.', ls='', c=colors[j], alpha=1.0)
            #axarr[1].scatter(lM+0.02*j, pd, marker='.', c=colors[j], alpha=0.3)

    axarr[1].axhline(0, c='k', ls='-', zorder=-1)
    axarr[1].axhline(1, c='k', ls='--', lw=0.5, zorder=-1)
    axarr[1].axhline(-1, c='k', ls='--', lw=0.5, zorder=-1)

    axarr[1].set_xlabel(xlabel)
    axarr[0].set_ylabel(y0label)
    axarr[1].set_ylabel(y1label)
    axarr[0].set_yscale('log')
    axarr[0].set_ylim(1, axarr[0].get_ylim()[1])
    axarr[1].set_ylim(-15, 15)
    #axarr[1].set_xlim(12.9, 15)
    leg = axarr[0].legend(loc=0, fontsize=6, numpoints=1, frameon=False)
    leg.get_frame().set_alpha(0.5)
    plt.subplots_adjust(bottom=0.15, left=0.19, hspace=0.0)
    #plt.gcf().savefig("with_george_testbox%03d.png"%i)
    plt.show()
    plt.clf()
