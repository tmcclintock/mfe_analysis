import sys,os
import cosmocalc as cc
import numpy as np
import tinker_mass_function as TMF
sys.path.insert(0,"../Mass-Function-Emulator/")
import mf_emulator as mfe
import matplotlib.pyplot as plt
plt.rc('text',usetex=True,fontsize=24)

#Scale factors and redshifts
scale_factors = np.array([0.25,0.333333,0.5,0.540541,0.588235,0.645161,0.714286,0.8,0.909091,1.0])
k = (scale_factors - 0.5)
redshifts = 1./scale_factors - 1.0
volume = 1050.**3 #snapshot volume; [Mpc/h]^3

#Read in the input cosmologies
all_cosmologies = np.genfromtxt("./test_data/building_cosmos_all_params.txt")
#all_cosmologies = np.delete(all_cosmologies,5,1) #Delete ln10As
all_cosmologies = np.delete(all_cosmologies,0,1) #Delete boxnum
all_cosmologies = np.delete(all_cosmologies,-1,0)#39 is broken
N_cosmologies = len(all_cosmologies)
N_z = 10

#Read in the input data; data is f0, f1, g0, g1
means = np.loadtxt("./test_data/mean_models.txt")
variances = np.loadtxt("./test_data/var_models.txt")
data = np.ones((N_cosmologies,len(means[0]),2)) #Last column is for mean/erros
data[:,:,0] = means
data[:,:,1] = np.sqrt(variances)

#Pull out the data point for box 0
f0,f1,g0,g1 = data[0,:,0]
ef0,ef1,eg0,eg1 = data[0,:,1]
f = f0 + k*f1
ef = ef0 + k*ef1
g = g0 + k*g1
eg = eg0 + k*eg1

#Pick out the training/testing data
box_index = 0
test_cosmo = all_cosmologies[box_index]
test_data = data[box_index]
training_cosmologies = np.delete(all_cosmologies,box_index,0)
training_data = np.delete(data,box_index,0)

#Train
mf_emulator = mfe.mf_emulator("test")
mf_emulator.train(training_cosmologies,training_data)

#Predict the TMF parameters
predicted = mf_emulator.predict_parameters(test_cosmo)
f0e,f1e,g0e,g1e = predicted[:,0]
fem = f0e + k*f1e
gem = g0e + k*g1e

plt.plot(scale_factors,fem,c='b',label="f")
plt.plot(scale_factors,gem,c='g',label="g")
plt.errorbar(scale_factors,f,ef,c='b',marker='.',ls='')
plt.errorbar(scale_factors,g,eg,c='g',marker='.',ls='')
plt.ylim(0.3,1.3)
plt.legend(loc=0)
plt.xlabel("scale factor",fontsize=24)
plt.ylabel("tinker paramater",fontsize=24)
plt.subplots_adjust(bottom=0.15,left=0.15)
plt.show()
