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
                for jj in range(len(cov[ii])):
                    outcov[ii,jj] = cov[ii,jj]/(N_bf[ii]*bG[ii] * N_bf[jj]*bG[jj])
            np.savetxt("txt_files/bgcov_%03d_z%d.txt"%(i,j), outcov)
        out = np.array([nus,delta0,edelta0]).T
        np.savetxt("txt_files/delta_%03d.txt"%i, out)
        print "Saved deltas for %03d"%i
    return

def fit_delta0():
    d0s = np.zeros(N_cosmos)
    ed0s= np.zeros(N_cosmos)
    for i in range(0,N_cosmos):
        data = np.loadtxt("txt_files/delta_%03d.txt"%i)
        start = 0
        d0j = np.zeros(N_z)
        vd0j = np.zeros(N_z)
        for j in range(0,N_z):
            cov = np.loadtxt("txt_files/bgcov_%03d_z%d.txt"%(i,j))
            icov = np.linalg.inv(cov)
            nu,d,_ = data[start:start+len(cov)].T
            A = np.ones_like(nu)
            var = np.dot(A, np.dot(icov, A))**-1
            rhs = np.dot(A, np.dot(icov, d))
            d0j[j] = np.dot(var, rhs)
            vd0j[j] = var
            start += len(cov)
        d0s[i] = np.average(d0j, weights=vd0j)
        ed0s[i] = np.sqrt(np.mean(vd0j))
        print i, d0s[i], ed0s[i]
    out = np.array([d0s, ed0s]).T
    np.savetxt("txt_files/delta0.txt", out)
    print "delta0s saved"
    plt.hist(d0s, 20)
    plt.show()

def get_bigdelta():
    Deltas = []
    eDeltas = []
    nus = []
    lMs = []
    zs = []
    siminds = []
    zinds = []
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
            #Get emulated curves
            TMF_model = TMF.tinker_mass_function(cosmo_dict, redshifts[j])
            d,e,f,g,B = get_params(emu_model, scale_factors[j])
            TMF_model.set_parameters(d,e,f,g,B)
            N_bf = volume * TMF_model.n_in_bins(lM_bins)
            bG = get_bG(scale_factors[j], 10**lM)
            Delta = (N-N_bf)/N_bf #- bG*delta0
            eDelta = err/N_bf
            nu = get_nu(scale_factors[j], 10**lM)
            Deltas = np.concatenate((Deltas, Delta))
            eDeltas = np.concatenate((eDeltas, eDelta))
            this_z = np.ones_like(nu)*redshifts[j]
            thissim = np.ones_like(nu)*i
            thesezs = np.ones_like(nu)*j
            nus = np.concatenate((nus, nu))
            lMs = np.concatenate((lMs, lM))
            zs = np.concatenate((zs, this_z))
            siminds = np.concatenate((siminds, thissim))
            zinds = np.concatenate((zinds, thesezs))
        print "Got bigDeltas for box%03d"%i
    out = np.array([zs, lMs, nus, Deltas, eDeltas, siminds, zinds]).T
    np.savetxt("txt_files/bigDeltas.txt", out)
    return

def plot_Delta_scatter():
    colors = get_colors()
    sf, zs = get_sf_and_redshifts()
    data = np.genfromtxt("txt_files/bigDeltas.txt")
    z, lM, nu, Delta, eDelta, thei, thej = data.T
    #good = np.where(np.fabs(Delta) < 0.5)
    #data = data[good]
    z, lM, nu, Delta, eDelta, thei, thej = data.T
    use_nu = True
    if use_nu: x = nu
    else: x = lM
    for i in range(len(colors)):
        inds = np.where(z == zs[i])[0]
        plt.scatter(x[inds], Delta[inds], alpha=0.2, marker='.', c=colors[i])
    plt.axhline(-0.01, c='k', ls='--')
    plt.axhline(0.01, c='k', ls='--')
    plt.axhline(0.0, c='k', ls='-')
    plt.ylim(-0.1, 0.1)
    if use_nu:
        plt.xlim(0.8,6)
        plt.xlabel(r"$\nu$", fontsize=24)
    else:
        plt.xlabel(r"$\log_{10}M\ [{\rm M_\odot}/h]$")
    plt.ylabel(r"$\Delta N/N_{\rm emu}$", fontsize=24)
    plt.subplots_adjust(bottom=0.15, left=0.2)
    plt.savefig("Delta_scatter.png")
    plt.savefig("fig_Delta_scatter.pdf")
    plt.show()

def plot_bigDelta():
    colors = get_colors()
    sf, zs = get_sf_and_redshifts()
    np.random.seed(12345666)
    data = np.genfromtxt("txt_files/bigDeltas.txt")
    L = len(data)/3
    newdata = np.random.permutation(data)[:L]
    nu = newdata[:,2]
    Delta = newdata[:,3]
    #inds = np.where(np.fabs(Delta) < 10)[0]
    #newdata = newdata[inds]
    z, lM, nu, Delta, eDelta, thei, thej = data.T
    x = np.array([nu,z]).T
    x0 = nu
    print x.shape
    ZEROS = np.zeros_like(Delta)
    aDelta = np.fabs(Delta)
    print np.mean(Delta), np.mean(eDelta), max(nu), min(nu)
    import george
    k,l = 1e-2, 3.5
    metric = l*np.ones_like(x[0])
    metric = np.array([0.537, 0.81318])
    kernel = k*george.kernels.ExpSquaredKernel(metric=metric, ndim=2)
    gp = george.GP(kernel)#, mean=np.mean(Delta))
    print "computing with george"
    gp.compute(x=x, yerr=eDelta)
    print "One compute done"
    #gp.optimize(x=x, y=Delta, yerr=eDelta)
    print "george optimized"
    print gp.kernel
    for i in range(len(colors)):
        inds = np.where(z == zs[i])[0]
        #plt.errorbar(x0[inds], Delta[inds], eDelta[inds], alpha=0.1, ls='', marker='.', zorder=-1, c=colors[i])
        t0 = np.linspace(min(x0)-1, max(x0)+1, 100)
        t = np.array([t0, zs[i]*np.ones_like(t0)]).T
        mu, cov = gp.predict(ZEROS, t)
        err = np.sqrt(np.diag(cov))
        #plt.plot(t, mu, c=colors[i])
        #plt.fill_between(t0, mu-err, mu+err, color=colors[i], alpha=0.1,zorder=-(i-10))
        for j in range(3):
            plt.plot(t0, gp.sample_conditional(ZEROS, t), c=colors[i], ls='-', zorder=-(i-10), alpha=0.2)
    plt.axhline(-0.01, c='k', ls='--')
    plt.axhline(0.01, c='k', ls='--')
    plt.axhline(0.0, c='k', ls='-')
    plt.axvline(max(x0), c='g', ls='-')
    plt.axvline(min(x0), c='g', ls='-')
    plt.ylim(-0.1, 0.1)
    plt.xlim(0,7)
    plt.xlabel(r"$\nu$", fontsize=24)
    plt.ylabel(r"$\Delta N/N_{\rm emu}$", fontsize=24)
    plt.subplots_adjust(left=0.2, bottom=0.15)
    plt.gcf().savefig("Delta_emu.png")
    plt.show()

def stats_on_Delta():
    sf, zs = get_sf_and_redshifts()
    data = np.genfromtxt("txt_files/bigDeltas.txt")
    print data.shape
    z, lM, nu, Delta, eDelta, thei, thej = data.T
    print np.max(Delta), np.min(Delta)
    print data.shape
    cut = 2
    good = np.where(np.fabs(Delta) < cut)
    bad = np.where(np.fabs(Delta) > cut)[0]
    bads = np.array([data[bad]]).T[5:]
    print len(data) - len(good[0])
    data = data[good]
    print data.shape
    z, lM, nu, Delta, eDelta, thei_new, thej_new = data.T
    weights = eDelta**-2
    print np.mean(Delta), np.mean(eDelta)
    print np.average(Delta, weights=weights)
    print np.average(eDelta, weights=weights)
    #print bads.shape
    return

if __name__ == "__main__":
    #make_delta0()
    #fit_delta0()
    #get_bigdelta()
    stats_on_Delta()
    plot_Delta_scatter()
    #plot_bigDelta()
