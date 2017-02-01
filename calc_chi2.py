import numpy as np
import pickle, sys, os, copy
import matplotlib.pyplot as plt
from scipy.stats import chi2

N_cosmos = 39#Number of data files
N_z = 10#Number of redshifts

N_data_array = pickle.load(open("test_data/N_data_array.p","rb"))
N_emu_array = pickle.load(open("test_data/N_emu_array.p","rb"))
cov_data_array = pickle.load(open("test_data/cov_data_array.p","rb"))
cov_emu_array = pickle.load(open("test_data/cov_emu_array.p","rb"))
logN_data_array = pickle.load(open("test_data/logN_data_array.p","rb"))
logN_emu_array = pickle.load(open("test_data/logN_emu_array.p","rb"))
logcov_emu_array = pickle.load(open("test_data/logcov_emu_array.p","rb"))

#chi2s = np.zeros((N_cosmos*N_z))
N_fp = np.zeros((N_cosmos*N_z)) #Number of free parameters
chi2s = []
N_fp = []
for i in xrange(0,N_cosmos):
    for j in xrange(0,N_z):
        index = i*N_z + j
        N_data    = N_data_array[index]
        N_emu     = N_emu_array[index]
        cov_emu  = cov_emu_array[index]
        cov_data  = cov_data_array[index]
        logN_data    = logN_data_array[index]
        logN_emu     = logN_emu_array[index]
        logcov_emu  = logcov_emu_array[index]
        #cov = cov_emu
        cov = logcov_emu
        w,v = np.linalg.eig(cov)
        icov = np.linalg.inv(cov)
        #X = N_data - N_emu
        X = logN_data - logN_emu
        #print X
        #print icov
        thischi2 = np.dot(X,np.dot(icov,X))
        #chi2s[i*N_z + j] = thischi2
        #N_fp[i*N_z + j] = len(N_data)
        if np.isnan(thischi2): 
            print "fuck up on %d %d"%(i,j)
            continue
        chi2s.append(thischi2)
        N_fp.append(len(N_data))
        #print "%.2f"%thischi2,":", "%.2e"%min(w),"%.2e"%max(w)
        continue #end j
    continue #end i

plt.hist(chi2s,20,normed=True) #Make the histogram
df = np.mean(N_fp)
mean,var,skew,kurt = chi2.stats(df,moments='mvsk')
x = np.linspace(chi2.ppf(0.01,df),chi2.ppf(0.99,df),100)
plt.plot(x,chi2.pdf(x,df))
#plt.title(r"$\chi^2$ for Box000",fontsize=24)
plt.xlabel(r"$\chi^2$",fontsize=24)
plt.subplots_adjust(bottom=0.15)
plt.show()
