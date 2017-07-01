"""
This file contains the code needed to, for instance, get the correct parameters,
get the cosmology, get redshifts, etc.
"""
import numpy as np
import matplotlib.pyplot as plt
import sys, os, emulator

#Paths to the building boxes
base = "../Mass-Function-Emulator/test_data/"
datapath = base+"N_data/Box%03d_full/Box%03d_full_Z%d.txt"
covpath  = base+"covariances/Box%03d_cov/Box%03d_cov_Z%d.txt"
def get_basepaths():
    return [base, datapath, covpath]

#Paths to the test boxes
base2 = "../../all_MF_data/Test_NM_data/averaged_mf_data/"
datapath2 = base2+"full_mf_data/TestBox%03d/TestBox%03d_mean_Z%d.txt"
covpath2  = base2+"covariances/TestBox%03d_cov/TestBox%03d_cov_Z%d.txt"
def get_testbox_paths():
    return [base2, datapath2, covpath2]

#Scale factors and redshifts of the sim
scale_factors = np.array([0.25, 0.333333, 0.5, 0.540541, 0.588235, 
                          0.645161, 0.714286, 0.8, 0.909091, 1.0])
redshifts = 1./scale_factors - 1.0
volume = 1050.**3 #[Mpc/h]^3
N_z = len(scale_factors)

def get_colors(cmapstring="seismic"):
    cmap = plt.get_cmap(cmapstring)
    return [cmap(ci) for ci in np.linspace(1.0, 0.0, N_z)]

def get_sf_and_redshifts():
    return [scale_factors, redshifts]

def get_volume():
    return volume

testbox_cosmos = np.genfromtxt("testbox_cosmos.txt")
cosmologies = np.genfromtxt("cosmos.txt")

def get_cosmo_dict(index):
    num,ombh2,omch2,w0,ns,ln10As,H0,Neff,sigma8 = cosmologies[index]
    h = H0/100.
    Ob,Om = ombh2/(h**2), ombh2/(h**2)+omch2/(h**2)
    cosmo_dict = {"om":Om, "ob":Ob, "ol":1-Om, "ok":0.0, "h":h, 
                  "s8":sigma8, "ns":ns, "w0":w0, "wa":0.0}
    return cosmo_dict

def get_testbox_cosmo_dict(index):
    ombh2,omch2,w0,ns,ln10As,H0,Neff,sigma8 = testbox_cosmos[index]
    h = H0/100.
    Ob,Om = ombh2/(h**2), ombh2/(h**2)+omch2/(h**2)
    cosmo_dict = {"om":Om, "ob":Ob, "ol":1-Om, "ok":0.0, "h":h, 
                  "s8":sigma8, "ns":ns, "w0":w0, "wa":0.0}
    return cosmo_dict

def get_building_cosmos(remove_As=True):
    building_cosmos = np.delete(cosmologies, 0, 1) #Delete boxnum
    if remove_As:
        building_cosmos = np.delete(building_cosmos, 4, 1) #Delete ln10As
    building_cosmos = np.delete(building_cosmos, -1, 0)#39 is broken
    return building_cosmos

def get_testbox_cosmos():
    return np.delete(testbox_cosmos, 4, 1) #Delete ln10As

"""
This gets the parameters. the default name is 'dfg', which is the
model from which we will write the paper. Other routines
can instead specify other names at call time.
"""
def get_params(model, sf, name='dfg'):
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

def get_all_fits(name='dfg'):
    base_dir = "../fit_mass_functions/output/%s/"%name
    base_save = base_dir+"%s_"%name
    best_fit_models = np.loadtxt(base_save+"bests.txt")
    mean_models = np.loadtxt(base_save+"means.txt")
    err_models = np.sqrt(np.loadtxt(base_save+"vars.txt"))
    return [best_fit_models, mean_models, err_models]

def get_rotated_fits(name='dfg'):
    base_dir = "../fit_mass_functions/output/%s_rotated/"%name
    base_save = base_dir+"rotated_%s_"%name
    mean_models = np.loadtxt(base_save+"means.txt")
    err_models = np.sqrt(np.loadtxt(base_save+"vars.txt"))
    R = np.genfromtxt(base_dir+"R_matrix.txt")
    return [mean_models, err_models, R]

#Routines for getting sim data
def get_sim_data(sim_index, z_index):
    base, datapath, covpath = get_basepaths()
    data = np.loadtxt(datapath%(sim_index, sim_index, z_index))
    lM_bins = data[:,:2]
    lM = np.mean(lM_bins, 1)
    N = data[:,2]
    cov = np.loadtxt(covpath%(sim_index, sim_index, z_index))
    err = np.sqrt(np.diagonal(cov))
    return lM_bins, lM, N, err, cov

def get_testbox_data(sim_index, z_index):
    base, datapath, covpath = get_testbox_paths()
    data = np.loadtxt(datapath%(sim_index, sim_index, z_index))
    N = data[:,2]
    goodinds = N>0
    data = data[goodinds]
    lM_bins = data[:,:2]
    lM = np.mean(lM_bins, 1)
    N = data[:,2]
    cov = np.loadtxt(covpath%(sim_index, sim_index, z_index))
    cov = cov[goodinds]
    cov = cov[:,goodinds]
    err = np.sqrt(np.diagonal(cov))
    return lM_bins, lM, N, err, cov

#################################
# Routines for getting emulator data
#################################
def train(training_cosmos, training_data, training_errs, use_george=False):
    N_cosmos = len(training_cosmos)
    N_emulators = training_data.shape[1]
    emulator_list = []
    for i in range(N_emulators):
        y = training_data[:, i]
        yerr = training_errs[:, i]
        emu = emulator.Emulator(name="emu%d"%i, xdata=training_cosmos, 
                                ydata=y, yerr=yerr)
        emu.train()
        emulator_list.append(emu)
    return emulator_list

def predict_parameters(cosmology, emu_list, R=None):
    params = np.array([emu.predict_one_point(cosmology)[0] for emu in emu_list])
    if R is None: return params
    else: return np.dot(R, params)

