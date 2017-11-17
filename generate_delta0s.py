"""
Here we run the emulator on each LOO box and get the Delta
values not at every mass but at nu values.
"""
import numpy as np
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', size=20)
import tinker_mass_function as TMF
import sys, os
import cosmocalc as cc
from setup_routines import *
import scipy.optimize as op
import george
from george.kernels import *
from george import Metric

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

def plot_bigDelta():
    colors = get_colors()
    sf, zs = get_sf_and_redshifts()
    np.random.seed(12345666)
    data = np.genfromtxt("R_T08.txt")
    L = len(data)
    newdata = np.random.permutation(data)[:L]
    nu = newdata[:,2]
    Delta = newdata[:,3]
    z, lM, nu, Delta, eDelta, thei, thej = data.T
    x = np.array([nu,z]).T
    x0 = nu
    ZEROS = np.zeros_like(Delta)
    c = -7.4627695322
    w = 1./eDelta**2
    mean = np.sum(w*Delta)/sum(w)
    err  = np.sqrt(np.sum(w*eDelta**2)/sum(w)) #Weighted stddev
    metric =[ 0.45806604,1.2785944] #Found via optimization
    kernel = ConstantKernel(log_constant= -7.4627695322, ndim=2)*ExpSquaredKernel(metric=[ 0.45806604,1.2785944], ndim=2) #found via optimization
    #kernel = ConstantKernel(log_constant= -7.4627695322, ndim=2)*ExpSquaredKernel(metric=[ 0.45806604e-3,1.2785944e0], ndim=2) #found via optimization

    #kernel = ConstantKernel(log_constant= c, ndim=2)*ExpSquaredKernel(metric=metric, ndim=2) + ConstantKernel(log_constant= c, ndim=2)*Matern52Kernel(metric=metric, ndim=2)
    #kernel = ConstantKernel(log_constant=-5.71365221669, ndim=2) * ExpSquaredKernel(metric=[  1.89259881e+00,   1.47400726e+10], ndim=2) + ConstantKernel(log_constant=-9.81035792396, ndim=2) * Matern52Kernel(metric=[ 0.56007193,  1.02191441], ndim=2)
    
    gp = george.GP(kernel)#, fit_mean=True)#, fit_white_noise=True)
    print "computing with george"
    yerr = eDelta
    #yerr = np.sqrt(4*Delta**2*eDelta**2) #work with residuals squared
    gp.compute(x=x, yerr=yerr)
    print "One compute done"
    y = np.fabs(Delta)
    y = Delta**2
    y = Delta
    def nll(p):
        gp.set_parameter_vector(p)
        ll = gp.lnlikelihood(y=y, quiet=True)
        return -ll if np.isfinite(ll) else 1e25
    def grad_nll(p):
        gp.set_parameter_vector(p)
        return -gp.grad_lnlikelihood(y=y, quiet=True)
    p0 = gp.get_parameter_vector()
    results = op.minimize(nll, p0, jac=grad_nll)
    p0 = results.x
    gp.set_parameter_vector(p0)
    print "george optimized"
    print gp.kernel
    
    for i in range(len(colors)):
        inds = (z==zs[i])
        if i ==9:
            plt.scatter(x0[inds], Delta[inds], alpha=0.2, marker='.', c=colors[i], s=2)
            #plt.errorbar(x0[inds], Delta[inds], eDelta[inds], alpha=0.1, ls='', marker='.', markersize=1, zorder=-1, c=colors[i])
        inds = (z==zs[i])*(nu > 1)*(nu<2)
        w = 1./eDelta[inds]**2
        print i, np.sum(w*Delta[inds])/sum(w), np.sqrt(np.sum(w*eDelta[inds]**2)/sum(w))
        t0 = np.linspace(min(x0)-1, max(x0)+1, 100)
        t = np.array([t0, zs[i]*np.ones_like(t0)]).T
        Y_PRED = y
        #Y_PRED = Delta
        #Y_PRED = np.sqrt(Delta**2)
        Y_PRED = ZEROS
        mu, cov = gp.predict(Y_PRED, t)
        err = np.sqrt(np.diag(cov))
        #plt.plot(t, mu, c=colors[i])
        if i==3 or i==9:
            plt.fill_between(t0, mu-err, mu+err, color=colors[i], alpha=0.1,zorder=-(i-10))

            for j in range(10):
                plt.plot(t0, gp.sample_conditional(Y_PRED, t), c=colors[i], ls='-', zorder=-(i-10), alpha=0.2)

    plt.axhline(-0.01, c='k', ls='--')
    plt.axhline(0.01, c='k', ls='--')
    plt.axhline(0.0, c='k', ls='-')
    plt.ylim(-0.1, 0.1)
    plt.xlim(min(x0),max(x0))
    plt.xlabel(r"$\nu$", fontsize=24)
    #plt.ylabel(r"$\Delta N/N_{\rm emu}$", fontsize=24)
    plt.ylabel(r"$R_{\rm T08}$", fontsize=24)
    plt.subplots_adjust(left=0.2, bottom=0.15)
    plt.gcf().savefig("Delta_emu.png")
    plt.show()

def stats_on_Delta():
    sf, zs = get_sf_and_redshifts()
    data = np.genfromtxt("R_T08.txt")
    #data = np.genfromtxt("txt_files/bigDeltas.txt")
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
    #stats_on_Delta()
    #plot_Delta_scatter()
    plot_bigDelta()
