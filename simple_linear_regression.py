#%%
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#%%
data = pd.read_csv('canada_per_capita_income.csv')
data.head(3)
#%%
sns.scatterplot(data=data, x='year', y='per capita income (US$)')
#%%
reg = LinearRegression()
#%%
year = data[['year']]
income = data['per capita income (US$)']
#%%
reg.fit(year, income)
#%%
reg.predict([[2020]])