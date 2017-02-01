"""
A quick script to look at the analytic covariance matrices
I have in comparison to the jackknife ones.
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rc("text",usetex=True,fontsize=24)

box = 0
zind = 9
MF_data = np.genfromtxt("../../all_MF_data/building_MF_data/full_mf_data/Box%03d_full/Box%03d_full_Z%d.txt"%(box,box,zind))
lM_bins = MF_data[:,:2]
lM = np.mean(lM_bins,1)
labels = ["%.1f"%lmi for lmi in lM]

cov_data = np.genfromtxt("../../all_MF_data/building_MF_data/covariances/Box%03d_cov/Box%03d_cov_Z%d.txt"%(box,box,zind))

cov_emu = np.loadtxt("emu_covs/cov_emu_%03d_Z%d.txt"%(box,zind))

seediag = False
if seediag:
    ddiag = np.sqrt(np.diagonal(cov_data))
    ediag = np.sqrt(np.diagonal(cov_emu))
    plt.plot(lM,ddiag,label="JK")
    plt.plot(lM,ediag,label="analytic")
    plt.yscale('log')
    plt.legend()
    plt.xlabel(r"$\log_{10}M\ [{\rm M_\odot}/h]$")
    plt.ylabel(r"$\sqrt{C_{\rm N_i,N_i}}$")
    plt.subplots_adjust(bottom=0.15,left=0.15)
    plt.show()

see_covs = True
if see_covs:
    def plot_cov(cov,title=None,log=False):
        if log:plt.pcolor(np.log10(np.fabs(cov)))
        else: plt.pcolor(cov)
        if title is not None: plt.title(title)
        ax = plt.gca()
        ax.set_xticks(np.arange(len(cov))+0.5,minor=False)
        ax.set_xticklabels(labels,minor=False,fontsize=12)
        ax.set_yticks(np.arange(len(cov))+0.5,minor=False)
        ax.set_yticklabels(labels,minor=False,fontsize=12)
        plt.xlabel(r"$\log_{10}M\ [{\rm M_\odot}/h]$")
        plt.ylabel(r"$\log_{10}M\ [{\rm M_\odot}/h]$")
        plt.colorbar()
        plt.show()
        return
    #plot_cov(cov_data,r"$C_{\rm JK}$")
    #plot_cov(cov_data,r"$\log_{10}|C_{\rm JK}|$",log=True)
    plot_cov(cov_emu,r"$C_{\rm emu}$")
    plot_cov(cov_emu,r"$\log_{10}|C_{\rm emu}|$",log=True)

see_corr = True
if see_corr:
    def plot_corr(cov,title=None):
        corr = np.ones_like(cov)
        for i in range(len(corr)):
            for j in range(len(corr)):
                corr[i,j] = cov[i,j]/np.sqrt(cov[i,i]*cov[j,j])
        print corr[0]
        print np.diagonal(corr)
        plt.pcolor(corr,vmin=-1.0,vmax=+1.0)
        if title is not None: plt.title(title)
        ax = plt.gca()
        ax.set_xticks(np.arange(len(corr))+0.5,minor=False)
        ax.set_xticklabels(labels,minor=False,fontsize=12)
        ax.set_yticks(np.arange(len(corr))+0.5,minor=False)
        ax.set_yticklabels(labels,minor=False,fontsize=12)
        plt.xlabel(r"$\log_{10}M\ [{\rm M_\odot}/h]$")
        plt.ylabel(r"$\log_{10}M\ [{\rm M_\odot}/h]$")
        plt.colorbar()
        plt.show()
        return
    #plot_corr(cov_data,r"$R_{\rm JK}$")
    plot_corr(cov_emu,r"$R_{\rm JK}$")


