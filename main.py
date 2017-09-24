###Main code for the HW1

print("UBitName:"+ "anantram")
print("personNumber:" + "50249127")


import pandas as pd
import numpy as np
import statistics as st
import matplotlib.pyplot as plt

file = 'university data.xlsx'
pfile = pd.ExcelFile(file)
sheetname = pfile.sheet_names
dataframe = pfile.parse('university_data')

X1 = np.array([dataframe['CS Score (USNews)'][0:49]])
X2 = np.array([dataframe['Research Overhead %'][0:49]])
X3 = np.array([dataframe['Admin Base Pay$'][0:49]])
X4 = np.array([dataframe['Tuition(out-state)$'][0:49]])


mu1 = np.mean(X1)
mu2 = np.mean(X2)
mu3 = np.mean(X3)
mu4 = np.mean(X4)

var1 = np.var(X1)
var2 = np.var(X2)
var3 = np.var(X3)
var4 = np.var(X4)

sigma1 = np.std(X1)
sigma2 = np.std(X2)
sigma3 = np.std(X3)
sigma4 = np.std(X4)

##############################################################################2
covls = []
corls = []
for i in [X1,X2,X3,X4]:
    for j in [X1,X2,X3,X4]:
	    covls.append(np.cov(i,j)[0][1])
		corls.append(np.corrcoef(i,j)[0][1])
		plt.plot(i,j)
		plt.show()
		

arr = np.array(covls)
arr1 = np.array(corls)
a,b,c,d = np.split(arr,4)
a1,b1,c1,d1 = np.split(arr1,4)
covarianceMat = np.matrix([a,b,c,d])
correlationMat = np.matrix([a1,b1,c1,d1])


#################################################################################3
Assuming that each variable is normally distributed and that they are independent of each
other, determine the log-likelihood of the data (Use the means and variances computed
earlier to determine the likelihood of each data value.)

from scipy.stats import norm,multivariate_normal
vector = []
logLikelihood = 0
multlogLikelihood = 0
for i in range(0,49):
	l=[]
	v2=[]
	matl = [X1.transpose(),X2.transpose(),X3.transpose(),X4.transpose()]
	meanm = [mu1,mu2,mu3,mu4]
	stdm = [sigma1,sigma2,sigma3,sigma4]
    for j,k,m in zip(matl,meanm,stdm):
		l.append(norm(k,m).pdf(j[i,0]))
		v2.append(j[i,0])
	multlogLikelihood = multlogLikelihood + np.log(multivariate_normal.pdf(v2,meanm,covarianceMat,allow_singular=True))
    vector.append(l)
for i in vector:
	a = 1
    for j in i:
		a = a * j
    logLikelihood = logLikelihood + np.log(a)
print(logLikelihood)#-1315.09879256
print(multlogLikelihood)

#####For multivariate 


##################################################################################4
Using the correlation values construct a Bayesian network which results in a higher log-
likelihood than in 3.

BNgraph:
A 4-by-4 matrix representing the acyclic directed graph showing the connections
of the Bayesian network. Each entry of the matrix takes value 0 or 1.

BNlogLikelihood:
A scalar showing log-likelihood produced by your Bayesian network.
The higher it is, the better score you get. Of course, it must match with the structure of
of network.





########################################################5
Using the Bayesian network to determine some interesting conditional probabilities