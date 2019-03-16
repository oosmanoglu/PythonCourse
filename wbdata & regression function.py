import wbdata
import numpy as np
import pandas as pd
from numpy.linalg import inv
from matplotlib import pyplot
import matplotlib.pyplot as plt
import seaborn; seaborn.set()

wbdata.get_source()
wbdata.get_indicator(source = 25)

wbdata.get_data('NY.GDP.PCAP.PP.KD', country = 'USA')
wbdata.get_data('SL.TLF.BASC.ZS', country = 'USA')

country = [i['id'] for i in wbdata.get_country('USA', display=False)]
indicators = {"NY.GDP.PCAP.PP.KD": "gdppc_ppp", "SL.TLF.BASC.ZS": "laborforce_basic_educ"}
# indicators are "GDP per capita, PPP (constant 2011 international $)"
# and "Labor force with basic education (% of total working-age population with basic education)"

df = wbdata.get_dataframe(indicators, country, convert_date = False)

df.to_csv('hw2.csv')
df.describe()

dataset = pd.read_csv('hw2.csv')

print(dataset)
data=dataset.dropna()
print(data)

X = data.iloc[:, 2].copy()
print(X)

y = data.iloc[:, 1].copy()
print(y)

X= np.array(X)
print(X)

y = np.array(y)
print(y)

X.shape
X.shape = ((6,4))

y.shape
y.shape = ((6,4))

b = np.linalg.inv(X.T @ X)@ X.T@ y

print(b)

yhat = np.array(X @ b)

e = yhat - y

sigmasq = e.T.dot(e) // (22) # (e'e)/(n-k-1) formula

varb = np.linalg.inv(X.T @ X) @ sigma-sq # sig-sq * (X'X)^-1 formula

plt.scatter(X,yhat, color = 'black')
plt.xlabel('laborforce_basic_educ')
plt.ylabel ('gdppc_ppp')
plt.show()




