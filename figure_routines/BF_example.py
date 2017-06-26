"""
Here I create the 'best fit' example, for one cosmology.

This is Figure 2.
"""
import numpy as np
import matplotlib.pyplot as plt
plt.rc('text', usetex=True, fontsize=20)
import tinker_mass_function as TMF
import cosmocalc as cc

xlabel  = r"$\log_{10}M\ [{\rm M_\odot}/h]$"
y0label = r"$N/[{\rm Gpc}^3\  \log_{10}{\rm M_\odot}/h]$"
y1label = r"$\%\ {\rm Diff}$"
y2label = r"$\frac{N-N_{bf}}{N_{bf}bG}$"

base = "../Mass-Function-Emulator/test_data/"
datapath = base+"N_data/Box%03d_full/Box%03d_full_Z%d.txt"
covpath  = base+"covariances/Box%03d_cov/Box%03d_cov_Z%d.txt"

N_cosmos = 39
N_z      = 10
c = np.linspace(1.0, 0.0, N_z) #Colors
cmap = plt.get_cmap('seismic') 
#cmap = plt.get_cmap('RdBu') 

scale_factors = np.array([0.25, 0.333333, 0.5, 0.540541, 0.588235, 
                          0.645161, 0.714286, 0.8, 0.909091, 1.0])
redshifts = 1./scale_factors - 1.0
volume = 1050.**3 #[Mpc/h]^3

cosmologies = np.genfromtxt("cosmos.txt")

#This contains our parameterization
name = 'dfg'
base_dir = "../fit_mass_functions/output/%s/"%name
base_save = base_dir+"%s_"%name
best_fit_models = np.loadtxt(base_save+"bests.txt")

def get_params(model, sf):
    Tinker_defaults = {'d':1.97, 'e':1.0, "f": 0.51, 'g':1.228}
    B=None
    if name is 'dfgB':
        d0,d1,f0,f1,g0,g1,B = model
        e0 = Tinker_defaults['e']
        e1 = 0.0
    if name is 'defg':
        d0,d1,e0,e1,f0,f1,g0,g1 = model
    if name is 'dfg':
        d0,d1,f0,f1,g0,g1 = model
        e0 = Tinker_defaults['e']
        e1 = 0.0
    if name is 'efg':
        e0,e1,f0,f1,g0,g1 = model
        d0 = Tinker_defaults['d']
        d1 = 0.0
    if name is 'fg':
        f0,f1,g0,g1 = model
        d0 = Tinker_defaults['d']
        d1 = 0.0
        e0 = Tinker_defaults['e']
        e1 = 0.0
    k = sf - 0.5
    d = d0 + k*d1
    e = e0 + k*e1
    f = f0 + k*f1
    g = g0 + k*g1
    return d,e,f,g,B

def get_bG(cosmo_dict, a, Masses):
    return cc.growth_function(a)*np.array([cc.tinker2010_bias(Mi, a, 200) for Mi in Masses])

fig, axarr = plt.subplots(2, sharex=True)

for i in range(0,1):
    #Get in the cosmology and create a cosmo_dict
    num,ombh2,omch2,w0,ns,ln10As,H0,Neff,sigma8 = cosmologies[i]
    h = H0/100.
    Ob,Om = ombh2/(h**2), ombh2/(h**2)+omch2/(h**2)
    cosmo_dict = {"om":Om, "ob":Ob, "ol":1-Om, "ok":0.0, "h":h, 
                  "s8":sigma8, "ns":ns, "w0":w0, "wa":0.0}

    for j in range(N_z):
        #First plot the data.
        data = np.loadtxt(datapath%(i, i, j))
        lM_bins = data[:,:2]
        lM = np.mean(data[:, :2], 1)
        N = data[:,2]
        cov = np.loadtxt(covpath%(i, i, j))
        err = np.sqrt(np.diagonal(cov))
        axarr[0].errorbar(lM, N, err, marker='.', ls='', c=cmap(c[j]), alpha=1.0, label=r"$z=%.1f$"%redshifts[j])

        #Now get the BF and plot it.
        TMF_model = TMF.tinker_mass_function(cosmo_dict, redshifts[j])
        d,e,f,g,B = get_params(best_fit_models[i], scale_factors[j])
        TMF_model.set_parameters(d,e,f,g,B)
        N_bf = volume * TMF_model.n_in_bins(lM_bins)
        axarr[0].plot(lM, N_bf, ls='--', c=cmap(c[j]), alpha=1.0)

        #Plot the %difference
        bG = get_bG(cosmo_dict, scale_factors[j], 10**lM)
        dN_N = (N-N_bf)/N_bf
        dN_NbG = dN_N/bG
        edN_NbG = err/N_bf/bG
        pd  = 100.*dN_N
        pde = 100.*err/N_bf
        axarr[1].errorbar(lM, pd, pde, marker='.', ls='', c=cmap(c[j]), alpha=1.0)
        #axarr[1].errorbar(lM, dN_NbG, edN_NbG, marker='.', ls='', c=cmap(c[j]), alpha=1.0)
    axarr[1].axhline(0, c='k', ls='-', zorder=-1)

#Show
axarr[1].set_xlabel(xlabel)
axarr[0].set_ylabel(y0label)
axarr[1].set_ylabel(y1label)
axarr[0].set_yscale('log')
axarr[0].set_ylim(1, axarr[0].get_ylim()[1])
axarr[1].set_ylim(-18, 18)
leg = axarr[0].legend(loc=0, fontsize=8, numpoints=1, frameon=False)
leg.get_frame().set_alpha(0.5)
plt.subplots_adjust(bottom=0.15, left=0.15, hspace=0.0)
fig.savefig("fig_BF.pdf")
plt.show()
