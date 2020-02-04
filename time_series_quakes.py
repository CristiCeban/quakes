#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from pandas import read_csv
from matplotlib import pyplot
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn.metrics import mean_squared_error
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv('quakes_toTrain.csv')
df.dtypes


# In[3]:


temp=[]
for i in range(len(df.period)):
    temp += [pd.to_datetime(df.period[i][:4] , format = '%Y')]
df['period'] = temp;
data = df.drop(['period'], axis=1)
data.index = df.period
data.head()


# In[4]:


plt.rcParams["figure.figsize"] = (20,3)
data.mean_mw.plot()
data.max_mw.plot()
pyplot.show()


# In[5]:


df.hist(figsize=(20,10))


# In[6]:


autocorrelation_plot(data.mean_mw)
autocorrelation_plot(data.max_mw)
pyplot.show()


# In[7]:


johan_test_temp = data
coint_johansen(johan_test_temp,-1,1).eig


# In[8]:


train = data[:int(0.8*(len(data)))]
valid = data[int(0.8*(len(data))):]
model = VAR(endog=train)
model_fit = model.fit()
prediction = model_fit.forecast(model_fit.y, steps=len(valid))


# In[9]:


model = VAR(endog=data)
model_fit = model.fit()
yhat = model_fit.forecast(model_fit.y, steps=3)
print(yhat)


# In[10]:


temp_list=[]
for i in range(len(yhat)):
    yhat[i][2] = abs(yhat[i][2])
    yhat[i][0] = yhat[i][0].astype(int)
    df_temp=pd.DataFrame(yhat, columns =['T','mean_mw','Speed','b','niu','delta_M','max_mw'])
    years=pd.to_datetime('2016-1-1')
    temp_list += [years.replace(year = years.year+(i+1)*5)]
df_temp['period'] = temp_list
df_temp.index = df_temp.period
df_temp = df_temp.drop(['period'], axis=1)
data=data.append(df_temp)
data.head()


# In[11]:


data.mean_mw.plot()
data.max_mw.plot()
pyplot.show()


# In[12]:


data


# In[ ]:




