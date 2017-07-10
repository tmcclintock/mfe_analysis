"""
With the george emulator working, calculate the chi2 of each box.
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

scale_factors, redshifts = get_sf_and_redshifts()
volume = get_volume()
N_z = len(scale_factors)
N_cosmos = 39
colors = get_colors()
building_cosmos = get_building_cosmos()
name = 'dfg'
mean_models, err_models, R = get_rotated_fits(name)

def calc_chi2():
    chi2 = np.zeros((N_cosmos, N_z))
    dofs = np.zeros((N_cosmos, N_z)) #degrees of freedom
    for i in range(0,N_cosmos):
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
            icov = np.linalg.inv(cov)
            #Get emulated curves
            TMF_model = TMF.tinker_mass_function(cosmo_dict, redshifts[j])
            d,e,f,g,B = get_params(emu_model, scale_factors[j])
            TMF_model.set_parameters(d,e,f,g,B)
            N_bf = volume * TMF_model.n_in_bins(lM_bins)
            X = N - N_bf
            chi2[i, j] = np.dot(X, np.dot(icov, X))
            dofs[i, j] = len(X)
        print "chi2 calculated for box%03d"%i
    np.savetxt("txt_files/chi2.txt", chi2)
    np.savetxt("txt_files/dofs.txt",dofs)
    return

def plot_chi2():
    from scipy.stats import chi2
    chi2s = np.loadtxt("txt_files/chi2.txt")
    good = np.where(chi2s < 200)[0]
    chi2s = chi2s[good]
    dofs = np.loadtxt("txt_files/dofs.txt")
    df = np.mean(dofs)
    fchi2s = chi2s.flatten()
    mean, var = chi2.stats(df, moments="mv")
    x = np.linspace(chi2.ppf(0.01, df), chi2.ppf(0.99, df), 100)
    plt.plot(x, chi2.pdf(x, df))
    plt.hist(fchi2s, 40, normed=True)
    plt.xlabel(r"$\chi_2$", fontsize=24)
    plt.subplots_adjust(bottom=0.15)
    plt.show()

if __name__ == "__main__":
    #calc_chi2()
    plot_chi2()
