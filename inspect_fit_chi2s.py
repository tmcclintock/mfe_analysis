import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

all_chi2s = np.genfromtxt("test_data/full_chi2_models.txt")
print all_chi2s.shape
chi2s = all_chi2s.flatten()
print chi2s.shape

plt.hist(chi2s,30,normed=True) #Make the histogram
df = 10 #ish, sometimes 9
mean,var,skew,kurt = chi2.stats(df,moments='mvsk')
x = np.linspace(chi2.ppf(0.01,df),chi2.ppf(0.99,df),100)
plt.plot(x,chi2.pdf(x,df))
#plt.title(r"$\chi^2$ for Box000",fontsize=24)
plt.xlabel(r"$\chi^2$",fontsize=24)
plt.subplots_adjust(bottom=0.15)
plt.show()
