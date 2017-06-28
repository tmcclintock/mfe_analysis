"""
Here I will emulate the test boxes.
"""
import numpy as np
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', size=20)
import tinker_mass_function as TMF
import sys, os, emulator
import cosmocalc as cc
from setup_routines import *

def train(training_cosmos, training_data, training_errs):
    N_cosmos = len(training_cosmos)
    N_emulators = training_data.shape[1]
    emulator_list = []
    for i in range(N_emulators):
        y = training_data[:, i]
        yerr = training_errs[:, i]
        emu = emulator.Emulator(name="emu%d"%i, xdata=training_cosmos, ydata=y, yerr=yerr)
        emu.train()
        emulator_list.append(emu)
    return emulator_list

def predict_parameters(cosmology, emu_list):
    params = np.array([emu.predict_one_point(cosmology)[0] for emu in emu_list])
    return np.dot(R, params)

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
name = 'dfg'
mean_models, err_models, R = get_rotated_fits(name)

#First train the emulators
emu_list = train(building_cosmos, mean_models, err_models)

#Loop over test boxes and do everything
for i in range(0,1):
    fig, axarr = plt.subplots(2, sharex=True)
    #test_cosmo = get_testbox_cosmo_dict(i)
    #emu_model = predict_parameters(test_cosmo, emu_list)
    print "working on it..."

    for j in range(0, N_z):
        lM_bins, lM, N, err, cov = get_testbox_data(i,j)
        axarr[0].errorbar(lM, N, err, marker='.', ls='', c=colors[j], alpha=1.0, label=r"$z=%.1f$"%redshifts[j])

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
    plt.show()
