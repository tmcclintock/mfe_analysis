"""
Here I create the 'best fit' example, for one cosmology.

This is Figure 2.
"""
import numpy as np
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', size=20)
import tinker_mass_function as TMF
import cosmocalc as cc
from setup_routines import *

xlabel  = r"$\log_{10}M\ [{\rm M_\odot}/h]$"
y0label = r"$N/[{\rm Gpc}^3\  \log_{10}{\rm M_\odot}/h]$"
y1label = r"$\%\ {\rm Diff}$"
y2label = r"$\frac{N-N_{bf}}{N_{bf}bG}$"

scale_factors, redshifts = get_sf_and_redshifts()
volume = get_volume()
N_z = len(scale_factors)
N_cosmos = 39
colors = get_colors()

#This contains our parameterization
name = 'dfg'
best_fit_models, mean_models, err_models = get_all_fits(name)

def get_bG(cosmo_dict, a, Masses):
    return cc.growth_function(a)*np.array([cc.tinker2010_bias(Mi, a, 200) for Mi in Masses])

fig, axarr = plt.subplots(2, sharex=True)

BOX=2
for i in range(BOX,BOX+1):
    cosmo_dict = get_cosmo_dict(i)

    for j in range(N_z):
        if j <2: continue
        lM_bins, lM, N, err, cov = get_sim_data(i,j)
        axarr[0].errorbar(lM, N, err, marker='.', ls='', c=colors[j], alpha=1.0, label=r"$z=%.1f$"%redshifts[j])

        #Now get the BF and plot it.
        TMF_model = TMF.tinker_mass_function(cosmo_dict, redshifts[j])
        d,e,f,g,B = get_params(best_fit_models[i], scale_factors[j])
        TMF_model.set_parameters(d,e,f,g,B)
        N_bf = volume * TMF_model.n_in_bins(lM_bins)
        axarr[0].plot(lM, N_bf, ls='--', c=colors[j], alpha=1.0)

        #Plot the %difference
        bG = get_bG(cosmo_dict, scale_factors[j], 10**lM)
        dN_N = (N-N_bf)/N_bf
        dN_NbG = dN_N/bG
        edN_NbG = err/N_bf/bG
        pd  = dN_N
        pde = err/N_bf
        axarr[1].errorbar(lM, pd, pde, marker='.', ls='', c=colors[j], alpha=1.0)
        #axarr[1].errorbar(lM, dN_NbG, edN_NbG, marker='.', ls='', c=colors[j], alpha=1.0)
    axarr[1].axhline(0, c='k', ls='-', zorder=-1)
    axarr[1].axhline(-.01, c='k', ls='--', zorder=-1)
    axarr[1].axhline(.01, c='k', ls='--', zorder=-1)


#Show
axarr[1].set_xlabel(xlabel)
axarr[0].set_ylabel(y0label)
axarr[1].set_ylabel(y1label)
axarr[0].set_yscale('log')
axarr[0].set_ylim(1, axarr[0].get_ylim()[1])
axarr[1].set_ylim(-.08, .08)
leg = axarr[0].legend(loc=0, fontsize=8, numpoints=1, frameon=False)
leg.get_frame().set_alpha(0.5)
plt.subplots_adjust(bottom=0.15, left=0.19, hspace=0.0)
#fig.savefig("fig_BF.pdf")
fig.savefig("fig_BF.png")
plt.show()
