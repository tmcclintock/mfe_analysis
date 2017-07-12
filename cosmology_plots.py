"""
Make scatter plots relating to the cosmologies we have.
This will one day include the super nice scatter plot
on top of the planck contours.
"""
import sys, os
import numpy as np
import matplotlib.pyplot as plt
plt.rc("text", usetex=True)
plt.rc("font", size=20)
from setup_routines import *

cosmos = get_building_cosmos(drop39=False)
ombh2,omch2,w0,ns,H0,Neff,sigma8 = cosmos.T
h = H0/100.
Ob,Om = ombh2/(h**2), ombh2/(h**2)+omch2/(h**2)

def Om_s8_scatterplot():
    plt.scatter(Om, sigma8, c='k', marker='o')
    plt.xlabel(r"$\Omega_m$", fontsize=24)
    plt.ylabel(r"$\sigma_8$", fontsize=24)
    plt.subplots_adjust(bottom=0.15)
    plt.show()

def H0_w_scatterplot():
    plt.scatter(H0, w0, c='k', marker='o')
    plt.xlabel(r"$H0\ [{\rm km/s/Mpc}]$", fontsize=24)
    plt.ylabel(r"$w$", fontsize=24)
    plt.subplots_adjust(bottom=0.16, left=0.15)
    plt.show()



if __name__=="__main__":
    #Om_s8_scatterplot()
    H0_w_scatterplot()
