import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pystan

trend2 = pd.read_csv('trend2.csv')
dataset = trend2.dropna()
dataset.head()


dataset.country = dataset.country.str.strip()
dataset_country = dataset.country.unique()
dataset_year = dataset.year.unique()
countries = len(dataset_country)
years = len(dataset_year)
country_search = dict(zip(dataset_country, range(len(dataset_country))))
year_search = dict (zip(dataset_year, range(len(dataset_year))))
country = dataset['country_code'] = dataset.country.replace(country_search).values
year = dataset['year_code'] = dataset.year.replace(year_search).values


church = dataset.church2
gini = dataset.gini_net
rgdpl = dataset.rgdpl

# beta with uniform distribution
dataset_model_1 = """ data {
  int<lower=0> J; 
  int<lower=0> N; 
  int<lower=1,upper=J> country[N];
  vector[N] x;
  vector[N] y;
} 
parameters {
  vector[J] a;
  real b;
  real mu_a;
  real<lower=0,upper=100> sigma_a;
  real<lower=0,upper=100> sigma_y;
} 
transformed parameters {

  vector[N] yhat;

  for (i in 1:N)
    yhat[i] = a[country[i]] + x[i] * b;
}
model {
  sigma_a ~ uniform(0, 100);
  a ~ normal (mu_a, sigma_a);

  b ~ uniform (0, 100);

  sigma_y ~ uniform(0, 100);
  y ~ normal(yhat, sigma_y);
}
"""
dataset_data = {'N': len(church),
                          'J': len(dataset_year),
                          'country': country+1, 
                          'x': gini,
                          'y': church}
						  

dataset_fit = pystan.stan(model_code= dataset_model_1, data= dataset_data, iter=1000, chains=2)
a_sample = pd.DataFrame(dataset_fit['a'])

import seaborn as sns
sns.set(style="darkgrid", palette="colorblind", color_codes=True)

plt.figure(figsize=(16, 8))
sns.boxplot(data=a_sample, whis=np.inf, color="blue")

dataset_fit.plot(pars=['sigma_a', 'b']);

dataset_fit['b'].mean()

xvals = np.arange(1)
bp = dataset_fit['a'].mean(axis=0)
mp = dataset_fit['b'].mean()
for bi in bp:
    plt.plot(xvals, mp*xvals + bi, 'bo-', alpha=0.4)
plt.xlim(-0.2,1.2)
plt.show()

# beta with normal distribution
dataset_model_2 = """ data {
  int<lower=0> J; 
  int<lower=0> N; 
  int<lower=1,upper=J> country[N];
  vector[N] x;
  vector[N] y;
} 
parameters {
  vector[J] a;
  real b;
  real mu_a;
  real<lower=0,upper=100> sigma_a;
  real<lower=0,upper=100> sigma_y;
} 
transformed parameters {

  vector[N] yhat;

  for (i in 1:N)
    yhat[i] = a[country[i]] + x[i] * b;
}
model {
  sigma_a ~ uniform(0, 100);
  a ~ normal (mu_a, sigma_a);

  b ~ normal (0, 1);

  sigma_y ~ uniform(0, 100);
  y ~ normal(yhat, sigma_y);
}
"""

dataset_data = {'N': len(church),
                          'J': len(dataset_year),
                          'country': country+1, 
                          'x': gini,
                          'y': church}
						  
dataset_fit = pystan.stan(model_code= dataset_model_2, data= dataset_data, iter=1000, chains=2)
a_sample = pd.DataFrame(dataset_fit['a'])

import seaborn as sns
sns.set(style="darkgrid", palette="colorblind", color_codes=True)

plt.figure(figsize=(16, 8))
sns.boxplot(data=a_sample, whis=np.inf, color="blue")

dataset_fit.plot(pars=['sigma_a', 'b']);

dataset_fit['b'].mean()

xvals = np.arange(1)
bp = dataset_fit['a'].mean(axis=0)
mp = dataset_fit['b'].mean()
for bi in bp:
    plt.plot(xvals, mp*xvals + bi, 'bo-', alpha=0.4)
plt.xlim(-0.2,1.2)
plt.show()