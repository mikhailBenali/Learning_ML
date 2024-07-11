#%%
import numpy as np
import pandas as pd
from sklearn import linear_model
#%%
df = pd.read_csv('homeprices.csv')
df.head()
#%%
import math
df.bedrooms = df.bedrooms.fillna(math.floor(df.bedrooms.median()))
df.head()
#%%
reg = linear_model.LinearRegression()
reg.fit(df[['area', 'bedrooms', 'age']], df.price)
# %%
reg.coef_
# %%
reg.intercept_
# %%
reg.predict([[3000, 3, 40]])