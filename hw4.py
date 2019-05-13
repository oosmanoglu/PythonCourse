import numpy as np
import pandas as pd
import os
os.chdir('desktop/pythoncourse/Homework')
data = pd.read_csv('immSurvey.csv')
data.head()

alphas = data.stanMeansNewSysPooled
sample = data.textToSend

from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer()
X = vec.fit_transform(sample)
X

pd.DataFrame(X.toarray(), columns=vec.get_feature_names())

bigram = CountVectorizer(ngram_range=(1,2),token_pattern = r'\b\w+\b', min_df=1)
X_2 = bigram.fit_transform(sample)
X_2

from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X_2, alphas,
random_state=1)

from sklearn.gaussian_process.kernels import ConstantKernel
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessRegressor

rbf = ConstantKernel(1.0) * RBF(length_scale=1.0)
gpr = GaussianProcessRegressor(kernel=rbf, alpha=1e-8)

gpr.fit(Xtrain.toarray(), ytrain)

mu_s, cov_s = gpr.predict(Xtest.toarray(), return_cov=True)

np.corrcoef(ytest, mu_s)

# for an attempt to increase the correlation coefficient, I used Matern Kernel 
from sklearn.gaussian_process.kernels import Matern 

rbf = Matern(1.0) * RBF(length_scale=1.0)
gpr = GaussianProcessRegressor(kernel=rbf, alpha = 1e-8)

gpr.fit(Xtrain.toarray(), ytrain)

mu_s, cov_s = gpr.predict(Xtest.toarray(), return_cov=True)

np.corrcoef(ytest, mu_s)

# as an alternative, I used TfidfVectorizer as well
from sklearn.feature_extraction.text import TfidfVectorizer
vec = TfidfVectorizer()
X = vec.fit_transform(sample)
X

pd.DataFrame(X.toarray(), columns = vec.get_feature_names())

bigram = TfidfVectorizer(ngram_range=(1,2),
token_pattern = r'\b\w+\b', min_df=1)

X_2= bigram.fit_transform(sample)

rbf = ConstantKernel(1.0) * RBF(length_scale=1.0)
gpr = GaussianProcessRegressor(kernel=rbf, alpha=1e-8)

gpr.fit(Xtrain.toarray(), ytrain)

mu_s, cov_s = gpr.predict(Xtest.toarray(), return_cov=True)

np.corrcoef(ytest, mu_s)

from sklearn.gaussian_process.kernels import Matern

rbf = Matern(1.0) * RBF(length_scale=1.0)
gpr = GaussianProcessRegressor(kernel=rbf, alpha = 1e-8)

gpr.fit(Xtrain.toarray(), ytrain)

mu_s, cov_s = gpr.predict(Xtest.toarray(), return_cov=True)

np.corrcoef(ytest, mu_s)