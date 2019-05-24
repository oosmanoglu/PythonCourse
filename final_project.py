import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

os.chdir('desktop/python_project')

housing = pd.read_stata('housing.dta', columns = ["price", "lotsize", "bedrooms", "bathrms", 
"stories", "driveway", "recroom", "fullbase", "gashw", "airco", "garagepl", "prefarea"])

housing.head()

housing.describe().transpose()

housing.shape

X = housing.drop('price', axis = 1) 

y = housing.price

 
import statsmodels.api as sm

x = sm.add_constant(X) # adding intercept to the linear regression model

model = sm.OLS(y,x)

LR = model.fit() 

yhat = LR.predict(x)  

LR.summary()

LR_robust = LR.get_robustcov_results() # Linear Regression with Robust Standard Errors
LR_robust.summary()

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.2)

from sklearn.linear_model import LinearRegression
model = LinearRegression(fit_intercept = True)

LR_model = model.fit(Xtrain, ytrain)
predicted = model.predict(Xtest)

plt.scatter(ytest, predicted)
plt.xlabel('TrueValues')
plt.ylabel('PredictedValues')
plt.show()

np.corrcoef(predicted, ytest)

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.5) # alternative, the one with the test size of 0.5

from sklearn.linear_model import LinearRegression
linearmodel = LinearRegression(fit_intercept = True)

LR_model = linearmodel.fit(Xtrain, ytrain)
predicted = linearmodel.predict(Xtest)

plt.scatter(ytest, predicted)
plt.xlabel('TrueValues')
plt.ylabel('PredictedValues')
plt.show()

np.corrcoef(predicted, ytest)

from sklearn.svm import SVR

svr_1 = svr = SVR(C=1000)

X_new = np.asarray(X)

y_new = np.asarray(y)

X_sliced = X_new[:,0]

X_sliced.shape

a = X_sliced[:, None]
 
a.shape

xfit = np.linspace(0, 10000, num = 546)

yfit = svr.fit(a, y_new).predict(xfit[:,None])

plt.figure(figsize = (12,10))
plt.errorbar(X_sliced, y_new, 0.3, fmt='o')
plt.plot(xfit, yfit, '-r', label = 'predicted', zorder = 10)
plt.plot(xfit, y_new, '-k', alpha=0.5, label = 'true model', zorder = 10)
plt.legend()

from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_new, yfit)
print(mse)

from sklearn.svm import SVR

svr_2 = SVR(kernel='linear', C=1000, epsilon = 1.0) # alternative SVR

X_new = np.asarray(X)

y_new = np.asarray(y)

X_sliced = X_new[:,0]

X_sliced.shape

a = X_sliced[:, None]
 
a.shape

xfit = np.linspace(0, 10000, num = 546)

yfit = svr_2.fit(a, y_new).predict(xfit[:,None])

plt.figure(figsize = (12,10))
plt.errorbar(X_sliced, y_new, 0.3, fmt='o')
plt.plot(xfit, yfit, '-r', label = 'predicted', zorder = 10)
plt.plot(xfit, y_new, '-k', alpha=0.5, label = 'true model', zorder = 10)
plt.legend()

from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_new, yfit)
print(mse)