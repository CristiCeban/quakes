#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#a


# In[3]:


df=pd.read_csv('quakes.csv',sep='\s*,\s*',engine='python')
df.head()


# In[4]:


len(df.index)


# In[5]:


df.nunique()


# In[6]:


df.drop_duplicates(subset ="datetime", 
                     keep = False, inplace = True) 


# In[7]:


len(df.index)


# In[8]:


(df['md']!=0).value_counts()


# In[9]:


df=df.drop(columns=['md'])
df['datetime'] = pd.to_datetime(df['datetime'])
df.head()


# In[10]:


sns.pairplot(df,vars=df.columns[1:],)


# In[11]:


sns.violinplot(df['mw'],orient='v')


# In[12]:


df[df['mw']==df['mw'].max()]


# In[13]:


# Deoarece majoritatea datelor observabile sunt apropiate de 2,
# si ele sunt nesemnificate,nu sunt simtite de om,le vom sterge.
# Vom lua in considerare doar celea care au in impact pentru om sau natura,
# cu mw >= 3.5.
(df['mw']>=3.5).value_counts()


# In[14]:


new_df=df.loc[df['mw'] >= 3.5].reset_index().drop(columns=['index'])
len(new_df.index)


# In[15]:


new_df.head()


# In[16]:


diction={}


# In[17]:


sns.pairplot(new_df,vars=new_df.columns[1:])


# In[18]:


sns.violinplot(new_df['mw'],orient='v')


# In[19]:


new_df.plot(x='datetime',y='mw',figsize=(20,10))


# In[20]:


new_df.loc[new_df['datetime'] >= '1979-1-1'].plot(x='datetime',y='mw',figsize=(20,10))


# In[21]:


for i in range(1900,2020,5):
    mask = (new_df['datetime'] > str(i+1)+'-1-1') & (new_df['datetime'] <= str(i+5)+'-12-31')
    diction.update({(str(i+1)+'-'+str(i+5)):new_df.loc[mask].reset_index().drop(columns=['index'])})


# In[22]:


diction['2016-2020'].head()


# In[23]:


# Cutremurile intr-un interval de timp(cel analizat e de 5 ani),
# au 6 caracteristici importante care le caracterizeaza.
# T - intervalul dintre primul cutremur si ultimul in perioada de 5 ani.
# mean_mw - media a earthquake's magnitude waves(mw) in intervalul de 5 ani.
# sqrt_dE(speed) - viteza de expulsare a energiei dE^(1/2) = sum(E^(1/2)/T)
# Coeficientii a si b din relatia ToDO sa atasez relatia
# Delta_M
# Max_mw magnitudinea maxima in intervalul de timp T


# In[24]:


mean_mw=[]
datetime=[]
interval=[]
sqrt_dE=[] #viteza de expulsare a energiei
b=[]
a=[]
niu=[]
delta_M=[]
max_mw=[]
for key in diction:
    interval+=[key]
    datetime+=[diction[key].datetime.max()-diction[key].datetime.min()]
    mean_mw+=[diction[key].mw.mean()]
    sqrt_dE+=[sum((10**(11.8+1.5*diction[key].mw))**0.5)]
    n=len(diction[key])
    Ni=[]
    for i in range(0,len(diction[key]),1):
        Ni+=[(diction[key].mw[i]<=diction[key].mw).sum()]
    sum_1=0
    for i in range(0,len(diction[key]),1):
        sum_1+=diction[key].mw[i]*np.log10(Ni[i])
    sum_mi=sum(diction[key].mw)
    sum_ni=0
    for i in range(0,len(diction[key]),1):
        sum_ni+=np.log10(Ni[i])
    sum_mi2=sum(diction[key].mw**2)
    b_temp=0
    b_temp=(n*sum_1-sum_mi*sum_ni)/(sum_mi**2-n*sum_mi2)
    b+=[b_temp]
    a_temp=0
    for i in range(0,len(diction[key]),1):
        a_temp+=(np.log(Ni[i])+b_temp*diction[key].mw[i])/n
    a+=[a_temp]
    niu_temp=0
    for i in range(0,len(diction[key]),1):
        niu_temp+=((np.log(Ni[i])-(a_temp-b_temp*diction[key].mw[i]))**2/(n-1))
    niu+=[niu_temp]
    delta_M+=[abs(diction[key].mw.max()-a_temp/b_temp)/10]
    max_mw+=[diction[key].mw.max()]
datetime=pd.to_timedelta(datetime, errors='coerce').days
sqrt_dE = [i / j for i, j in zip(sqrt_dE , datetime)]
df=pd.DataFrame({'period':interval,
                'T':datetime,
                'mean_mw':mean_mw,
                'Speed':sqrt_dE,
                'b':b,
                'niu':niu,
                'delta_M':delta_M,
                'max_mw': max_mw,
                #'Ptest':[0]*24,
                #'Ytest':[0]*24,
                })
df.head()


# In[25]:


df.plot(x='period',y='mean_mw',figsize=(20,10))
#deoarece in ultima perioada sunt cu mult mai multe date despre cutremure,pe 5 an cate 200 cu mw>3.5,
#fata de 1940 cand erau fixate doar cate 15-20 si doar cele mai puternice


# In[26]:


df.plot(x='period',y='max_mw',figsize=(20,10))


# In[27]:


df.to_csv(r'quakes_toTrain.csv', index = None, header=True)

