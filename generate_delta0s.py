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

usegeorge = True

xlabel = r"$\nu$"
y0label = r"$\Delta=\frac{N}{N_{emu}bG}-1-\frac{1}{bG}$"
y0label = r"$\Delta N/N_{emu}bG$"
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

def make_delta0():
    for i in range(0,N_cosmos):
        delta0 = []
        edelta0 = []
        nus = []
        cosmo_dict = get_cosmo_dict(i)
        
        test_cosmo = building_cosmos[i]
        test_data  = mean_models[i]
        test_err   = err_models[i]
        training_cosmos = np.delete(building_cosmos, i, 0)
        training_data   = np.delete(mean_models, i, 0)
        training_errs   = np.delete(err_models, i, 0)
        
        #Train the emulators
        emu_list = train(training_cosmos, training_data, training_errs, use_george=usegeorge)
        emu_model = predict_parameters(test_cosmo, emu_list, training_data, R=R, use_george=usegeorge)

        for j in range(N_z):
            lM_bins, lM, N, err, cov = get_sim_data(i,j)
            outcov = np.zeros_like(cov)

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
            delta0  = np.concatenate((delta0, Delta))
            edelta0 = np.concatenate((edelta0, Deltae))
            nus     = np.concatenate((nus, nu))
            for ii in range(len(cov)):
                for jj in range(len(cov[i])):
                    outcov[i,j] = cov[i,j]/(N_bf[ii]*bG[ii] * N_bf[jj]*bG[jj])
            np.savetxt("txt_files/bgcov_%03d_z%d.txt"%(i,j), outcov)
        out = np.array([nus,delta0,edelta0]).T
        np.savetxt("txt_files/delta_%03d.txt"%i, out)
        print "Saved deltas for %03d"%i
    return

def fit_delta0():
    for i in range(0,N_cosmos):
        nu, d, e = np.loadtxt("txt_files/delta_%03d.txt"%i, unpack=True)
        

if __name__ == "__main__":
    make_delta0()
    #fit_delta0()
