"""
Create the emulator covariance matrices that will
replace the JK covariance matrix.
"""
import sys,os
import cosmocalc as cc
import numpy as np
import tinker_mass_function as TMF
sys.path.insert(0,"../Mass-Function-Emulator/")
import mf_emulator as mfe

#Scale factors and redshifts
scale_factors = np.array([0.25,0.333333,0.5,0.540541,0.588235,0.645161,0.714286,0.8,0.909091,1.0])
redshifts = 1./scale_factors - 1.0
volume = 1050.**3 #snapshot volume; [Mpc/h]^3

#Read in the input cosmologies
all_cosmologies = np.genfromtxt("./test_data/building_cosmos_all_params.txt")
#all_cosmologies = np.delete(all_cosmologies,5,1) #Delete ln10As
all_cosmologies = np.delete(all_cosmologies,0,1) #Delete boxnum
all_cosmologies = np.delete(all_cosmologies,-1,0)#39 is broken
N_cosmologies = len(all_cosmologies)
N_z = 10

#Read in the input data
means = np.loadtxt("./test_data/mean_models.txt")
variances = np.loadtxt("./test_data/var_models.txt")
data = np.ones((N_cosmologies,len(means[0]),2)) #Last column is for mean/erros
data[:,:,0] = means
data[:,:,1] = np.sqrt(variances)

#Pick out the training/testing data
box_index, z_index = 0, 9
test_cosmo = all_cosmologies[box_index]
test_data = data[box_index]
training_cosmologies = np.delete(all_cosmologies,box_index,0)
training_data = np.delete(data,box_index,0)

#Train
mf_emulator = mfe.mf_emulator("test")
mf_emulator.train(training_cosmologies,training_data)

#Predict the TMF parameters
predicted = mf_emulator.predict_parameters(test_cosmo)

#Read in the data
MF_data = np.genfromtxt("../../all_MF_data/building_MF_data/full_mf_data/Box%03d_full/Box%03d_full_Z%d.txt"%(box_index,box_index,z_index))
lM_bins = MF_data[:,:2]
lM = np.mean(lM_bins,1)
print lM
N_data = MF_data[:,2]
cov_data = np.genfromtxt("../../all_MF_data/building_MF_data/covariances/Box%03d_cov/Box%03d_cov_Z%d.txt"%(box_index,box_index,z_index))
N_err = np.sqrt(np.diagonal(cov_data))

#Predict the TMF
n = mf_emulator.predict_mass_function(test_cosmo,redshift=redshifts[z_index],lM_bins=lM_bins)
N_emu = n*volume

#Make the analytic covariance matrix
cov_an = np.diag(N_emu)

#Set up cosmocalc
ombh2,omch2,w0,ns,ln10As,H0,Neff,sigma8 = test_cosmo
h = H0/100.
Ob = ombh2/h**2
Om = Ob + omch2/h**2
cosmo_dict = {"om":Om,"ob":Ob,"ol":1-Om,\
                  "ok":0.0,"h":h,"s8":sigma8,\
                  "ns":ns,"w0":w0,"wa":0.0}
cc.set_cosmology(cosmo_dict) #Used to create the splines in cosmocalc

#Get the biases
bias = np.zeros_like(lM)
a = 1./(1+redshifts[z_index])
for i in xrange(0,len(lM)):
    M = 10**lM[i]
    bias[i] = cc.tinker2010_bias(M,a,200)

#Get sigmaR
R = (3./4./np.pi*volume)**(1./3.)
sigmaR = cc.sigmaRtophat_exact(R,a)
print sigmaR**2

for i in xrange(0,len(bias)):
    for j in xrange(0,len(bias)):
        if i==j:continue
        cov_an[i,j] = bias[i]*bias[j]*N_emu[i]*N_emu[j]*sigmaR**2
        continue
    continue
print cov_an[0]
import matplotlib.pyplot as plt
plt.imshow(np.log(cov_an))
plt.show()
