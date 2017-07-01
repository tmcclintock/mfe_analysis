"""
Here I create an emulator where the training data is 'rotated' and degeneracies
are broken, so that the GP can predict each parameter independently of
the others.
"""
import numpy as np
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', size=20)
import tinker_mass_function as TMF
import sys, os, emulator
import cosmocalc as cc
from setup_routines import *

Xisnu = True
if Xisnu: xlabel = r"$\nu$"
else: xlabel  = r"$\log_{10}M\ [{\rm M_\odot}/h]$"

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
name = 'dfg'
mean_models, err_models, R = get_rotated_fits(name)

def get_bG(cosmo_dict, a, Masses):
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
    emu_model = predict_parameters(test_cosmo, emu_list, R)

    for j in range(0,N_z):
        lM_bins, lM, N, err, cov = get_sim_data(i,j)
        nu = get_nu(scale_factors[j], 10**lM)
        if Xisnu: domain = nu
        else: domain = lM

        #Get emulated curves
        TMF_model = TMF.tinker_mass_function(cosmo_dict, redshifts[j])
        d,e,f,g,B = get_params(emu_model, scale_factors[j])
        TMF_model.set_parameters(d,e,f,g,B)
        N_bf = volume * TMF_model.n_in_bins(lM_bins)
        bG = get_bG(cosmo_dict, scale_factors[j], 10**lM)


        dN_N = (N-N_bf)/N_bf
        dN_NbG = dN_N/bG
        edN_NbG = err/N_bf/bG
        pd  = 100.*dN_N
        pde = 100.*err/N_bf
        axarr[0].errorbar(domain, N, err, marker='.', ls='', c=colors[j], alpha=1.0, label=r"$z=%.1f$"%redshifts[j])
        axarr[0].plot(domain, N_bf, ls='--', c=colors[j], alpha=1.0)
        axarr[1].errorbar(domain, pd, pde, marker='.', ls='', c=colors[j], alpha=1.0)
        #axarr[1].errorbar(domain, dN_NbG, edN_NbG, marker='.', ls='', c=colors[j], alpha=1.0)
    axarr[1].axhline(0, c='k', ls='-', zorder=-1)

    axarr[1].set_xlabel(xlabel)
    axarr[0].set_ylabel(y0label)
    axarr[1].set_ylabel(y1label)
    axarr[0].set_yscale('log')
    axarr[0].set_ylim(1, axarr[0].get_ylim()[1])
    axarr[1].set_ylim(-18, 18)
    #axarr[1].set_xlim(12.9, 15)
    leg = axarr[0].legend(loc=0, fontsize=6, numpoints=1, frameon=False)
    leg.get_frame().set_alpha(0.5)
    plt.subplots_adjust(bottom=0.15, left=0.19, hspace=0.0)
    #fig.savefig("fig_emurot.pdf")
    plt.show()
