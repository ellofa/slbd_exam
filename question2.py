from load_data import load_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
#from sklearn.ensemble import RandomForestRegressor
#from sklearn.impute import IterativeImputer


X,y,data = load_data()

# Identify missing values
data[data == -1000] = np.nan
data_X = data.loc[:, data.columns != '0'] 
data_y = data.loc[:,'0'] 

missing_data_per_obs = data_X.isna().sum(axis=1)
#print(missing_data_per_obs)
missing_data_per_feature = data_X.isna().sum(axis=0)
missing_data_distribution = data.isnull().mean(axis=1)
print(missing_data_distribution)

#plot the distribution of missing data per feature
plt.hist(missing_data_per_obs, bins=range(1, np.max(missing_data_per_obs) + 2), edgecolor='black')
plt.xlabel('Number of Missing Features')
plt.ylabel('Frequency')
plt.title('Distribution of missing data per observation')
plt.savefig("figures/missing_data_per_obs")
plt.show()


#plot the distribution of missing data per observation
plt.hist(missing_data_per_feature, bins=range(300, np.max(missing_data_per_feature) + 2), edgecolor='black')
plt.xlabel('Number of Missing Features')
plt.ylabel('Frequency')
plt.title('Distribution of missing data per feature')
plt.savefig("figures/missing_data_per_feature")
plt.show()

data_X.dropna(inplace=True)
print("shape clean", data_X.shape)

np.random.seed(42)

for i in range(data_X.shape[0]):
    # missingness distribution for feature 
    bernoulli_pick = np.random.binomial(1, 0.6, 1)
    print(bernoulli_pick)
    obs_missing_data_ratio = missing_data_distribution.iloc[i]
    print(obs_missing_data_ratio)
    pick = np.random.rand(1)
    if (pick < obs_missing_data_ratio) and (bernoulli_pick[0] == 1):
        data_X.iloc[i, :] = np.nan

data_X.dropna(inplace=True)
print("shape after mimicing missing vals", data_X.shape)