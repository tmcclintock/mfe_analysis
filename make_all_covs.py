"""
Create all the emulator covariance matrices that will
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
N_z = len(redshifts)
volume = 1050.**3 #snapshot volume; [Mpc/h]^3
R = (3./4./np.pi*volume)**(1./3.)

#Read in the input cosmologies
all_cosmologies = np.genfromtxt("./test_data/building_cosmos_all_params.txt")
#all_cosmologies = np.delete(all_cosmologies,5,1) #Delete ln10As
all_cosmologies = np.delete(all_cosmologies,0,1) #Delete boxnum
all_cosmologies = np.delete(all_cosmologies,-1,0)#39 is broken
N_cosmologies = len(all_cosmologies)

#Read in the input data
means = np.loadtxt("./test_data/mean_models.txt")
variances = np.loadtxt("./test_data/var_models.txt")
data = np.ones((N_cosmologies,len(means[0]),2)) #Last column is for mean/erros
data[:,:,0] = means
data[:,:,1] = np.sqrt(variances)

N_data_array = []
N_emu_array = []
logN_data_array = []
logN_emu_array = []
cov_emu_array = []
logcov_emu_array = []

#Loop over boxes and redshifts to create the matrices
for box in xrange(0,N_cosmologies):
    #Split the data
    test_cosmo = all_cosmologies[box]
    test_data = data[box]
    training_cosmologies = np.delete(all_cosmologies,box,0)
    training_data = np.delete(data,box,0)

    #Train
    mf_emulator = mfe.mf_emulator("test")
    mf_emulator.train(training_cosmologies,training_data)

    #Set up cosmocalc
    ombh2,omch2,w0,ns,ln10As,H0,Neff,sigma8 = test_cosmo
    h = H0/100.
    Ob = ombh2/h**2
    Om = Ob + omch2/h**2
    cosmo_dict = {"om":Om,"ob":Ob,"ol":1-Om,"ok":0.0,"h":h,"s8":sigma8,"ns":ns,"w0":w0,"wa":0.0}
    cc.set_cosmology(cosmo_dict)

    #Now loop over each redshift bin
    for zind in xrange(0,N_z):
        z = redshifts[zind]
        a = scale_factors[zind]
        sigmaR = cc.sigmaRtophat_exact(R,a)
        
        #Read in the data
        MF_data = np.genfromtxt("../../all_MF_data/building_MF_data/full_mf_data/Box%03d_full/Box%03d_full_Z%d.txt"%(box,box,zind))
        lM_bins = MF_data[:,:2]
        lM = np.mean(lM_bins,1)
        N_data_array.append(MF_data[:,2])
        logN_data_array.append(np.log(MF_data[:,2]))

        #Get the biases
        bias = np.zeros_like(lM)
        for i in xrange(0,len(bias)):
            bias[i] = cc.tinker2010_bias(10**lM[i],a,200)

        #Predict the TMF
        n = mf_emulator.predict_mass_function(test_cosmo,redshift=z,lM_bins=lM_bins)
        N_emu = n*volume
        N_emu_array.append(N_emu)
        logN_emu_array.append(np.log(N_emu))

        #Make the covariance matrix
        cov_emu = np.diag(N_emu)
        logcov_emu = np.diag(1./N_emu)
        for i in xrange(0,len(lM)):
            for j in xrange(0,len(lM)):
                if i==j: continue
                cov_emu[i,j] = bias[i]*bias[j]*N_emu[i]*N_emu[j]*sigmaR**2
                logcov_emu[i,j] = bias[i]*bias[j]*sigmaR**2
                continue #end i
            continue #end j
        cov_emu_array.append(cov_emu)
        logcov_emu_array.append(logcov_emu)
        np.savetxt("emu_covs/cov_emu_%03d_Z%d.txt"%(box,zind),cov_emu)
        print "Done with %d, %d"%(box,zind)
        continue #end zind
    continue #end box
import pickle
pickle.dump(cov_emu_array,open("test_data/cov_emu_array.p","wb"))
pickle.dump(N_data_array,open("test_data/N_data_array.p","wb"))
pickle.dump(N_emu_array,open("test_data/N_emu_array.p","wb"))
pickle.dump(logcov_emu_array,open("test_data/logcov_emu_array.p","wb"))
pickle.dump(logN_data_array,open("test_data/logN_data_array.p","wb"))
pickle.dump(logN_emu_array,open("test_data/logN_emu_array.p","wb"))
