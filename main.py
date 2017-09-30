###########################################################################
## This code is to explore some probability distribution concepts of machine learning using a data sets of US colleges from usnews.com
## Author: Anant Gupta
###########################################################################


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy  as scipy
import math
import seaborn as sns


data=pd.read_excel("/university data.xlsx")

df= pd.DataFrame(data)
df=df.drop(df.index[49])  ###to delete last Nan row

X= df.iloc[:,2:6]
Y=df.iloc[:,7]

mu	= X.apply(np.mean)
sigma = X.apply(np.std)
var = X.apply(np.var)

mu1= mu[0];
mu2= mu[1];
mu3= mu[2];
mu4= mu[3];

sigma1= sigma[0];
sigma2= sigma[1];
sigma3= sigma[2];
sigma4= sigma[3];

var1= var[0];
var2= var[1];
var3= var[2];
var4= var[3];

covarianceMat= pd.DataFrame.cov(X)

correlationMat= pd.DataFrame.corr(X)

##Correlation matrix Heat plot for analysing the correlation
multiGraph= sns.heatmap(correlationMat, vmax=1., square=False).xaxis.tick_top()


###All-in-One scatter plot
pd.tools.plotting.scatter_matrix(X, diagonal="kde")


### Separate Pairwise scatter Plots
sns.lmplot('CS Score (USNews)', 'Research Overhead %', X, hue='CS Score (USNews)', fit_reg=False);
sns.lmplot('CS Score (USNews)', 'Admin Base Pay$', X, hue='CS Score (USNews)', fit_reg=False);
sns.lmplot('CS Score (USNews)', 'Tuition(out-state)$', X, hue='CS Score (USNews)', fit_reg=False);

sns.lmplot('Research Overhead %', 'Admin Base Pay$', X, hue='Research Overhead %', fit_reg=False);
sns.lmplot('Research Overhead %', 'Tuition(out-state)$', X, hue='Research Overhead %', fit_reg=False);

sns.lmplot('Admin Base Pay$', 'Tuition(out-state)$', X, hue='Admin Base Pay$', fit_reg=False);



###Univariate PDF, Using scipy.stats.norm.pdf()

uni_log=0
logLikelihood_univariate=0
for i in range(0,4):
     pdf= scipy.stats.norm.pdf(X.iloc[:,i], mu[i], sigma[i])
     uni_log = uni_log + np.sum(np.log(pdf))


logLikelihood_univariate = uni_log


###Multivariate PDF, using inbuilt function scipy.stats.multivariate_normal.pdf()

multiVar_log=0
logLikelihood_multivariate1=0
for i in range(0,49):
     multiVar_log= multiVar_log + math.log(scipy.stats.multivariate_normal.pdf(X.iloc[i,:],mu,covarianceMat, allow_singular=True))

logLikelihood_multivariate1 = multiVar_log


covarianceMat= np.matrix(covarianceMat)
#### Multivariate PDF Formula Implemented
def multivariate(v2,meanm,covarianceMat):
	v2 = np.array(v2)
	meanm = np.array(meanm)
	size = len(v2)
	if size == len(meanm) and (size, size) == covarianceMat.shape:
		det = np.linalg.det(covarianceMat)
		if det == 0:
			raise NameError("The covariance matrix can't be singular")
		norm_const = 1.0/ ( math.pow((2*math.pi),float(size)/2) * math.pow(det,1.0/2) )
		x_mu = np.matrix(v2 - meanm)
		inv = covarianceMat.I
		result = math.pow(math.e, -0.5 * (x_mu * inv * x_mu.T))
		return norm_const * result
	else:
		raise NameError("The dimensions of the input don't match")

######Multivariate_PDF using implemented formula for Multivariate pdf
multiVar_log=0
logLikelihood_multivariate2=0
for i in range(0,49):
     multiVar_log= multiVar_log + math.log(multivariate(X.iloc[i,:],mu,covarianceMat))

logLikelihood_multivariate2 = multiVar_log


###########Printing outputs after rounding off to 3 decimals
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
var = np.array(var)

print('\nmu1: ',mu1,'\nmu2: ',mu2,'\nmu3: ',mu3,'\nmu4: ',mu4)
print('\nvar1: ',np.around(var[0], decimals=3),'\nvar2: ',np.around(var[1], decimals=3),'\nvar3: ',np.around(var[2], decimals=3),'\nvar4: ',np.around(var[3], decimals=3))
print('\nsigma1: ',sigma1,'\nsigma2: ',sigma2,'\nsigma3: ',sigma3,'\nsigma4: ',sigma4)


print('\ncovarianceMat: ', covarianceMat)
print('\ncorrelationMat: ', correlationMat)

print('\nlogLikelihood_univariate: ', logLikelihood_univariate)
print('\nlogLikelihood_multivariate1: ', logLikelihood_multivariate1)
print('\nlogLikelihood_multivariate2: ', logLikelihood_multivariate2)
