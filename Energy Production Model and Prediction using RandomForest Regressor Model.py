#!/usr/bin/env python
# coding: utf-8

# ### Random Forest Regressor Model 

# In[2]:


import matplotlib.pylab as plt
import pprint
import numpy as np
import pandas as pd
import seaborn as sns


# ### Exploratory Data Analysis 

# In[3]:


file_path = 'C:\Dilip\Business Analytics\MP\Model\energy_daily.csv'
df = pd.read_csv(file_path)

df.rename(columns = {'Energy MW':'Energy'}, inplace=True)

#changing to datetime
from datetime import datetime
df['date'] = pd.to_datetime(df['date'])

#in time series set index to date time column
df = df.set_index('date')

df


# In[4]:


#train and test data
train = df.loc[df.index <  '7/1/2020']
test = df.loc[df.index > '7/1/2020']

fig, ax = plt.subplots(figsize=(15,5))
train.plot(ax=ax, label='Training set', title='Train/Test split of Energy Production')
test.plot(ax=ax, label='Test set')
ax.legend(['Training set','Test set'])
plt.show()


# In[5]:


#Creating Feature Function
def create_feature(df):
    df = df.copy()
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    return df

df = create_feature(df)
df


# In[6]:


train = create_feature(train)
test = create_feature(test)


# In[7]:


FEATURES = ['dayofweek', 'quarter', 'month', 'year', 'dayofyear']
TARGET = 'Energy'


# In[8]:


x_train = train[FEATURES]
y_train = train[TARGET]

x_test = test[FEATURES]
y_test = test[TARGET]

print(x_test)
print(y_test)


# ### Building Random Forest Model 

# In[9]:


from sklearn.ensemble import RandomForestRegressor


# In[10]:


RF = RandomForestRegressor(max_depth=40,random_state=0)
RF.fit(x_train,y_train)
RF_predict = RF.predict(x_train)
plt.scatter(RF_predict,y_train)


# In[11]:


#Importing libraries
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn import metrics

#R2 square error
rf_r2_value = metrics.r2_score(y_train,RF_predict)
print("r-sqaure: ",rf_r2_value)


# In[13]:


train['prediction'] = RF.predict(x_train)
data = df.merge(train['prediction'], how='left', left_index=True, right_index = True)
data


# In[14]:


#plot predcition with actual data
plt.figure(figsize = (15,5))
plt.plot(data['Energy'],label = 'original')
plt.plot(data['prediction'], label = 'prediction')
plt.legend(loc='best')
plt.title("Energy Production using Random Forest")


# In[37]:


#Plotting one week data to analyse correlation
data.loc[(data.index > '12/30/2019') & (data.index <'1/30/2020')]['Energy'].plot(figsize=(15,5),title='One week data')
data.loc[(data.index > '12/30/2019') & (data.index <'1/30/2020')]['prediction'].plot(style='-')
plt.legend(['Original','Prediction'])
plt.show()


# ### Predicting future trends 

# In[21]:


future = pd.date_range('2024-01-01','2025-12-31', freq='1m')
future_df = pd.DataFrame(index = future)

future_df['isFuture'] = True
data['isFuture'] = False
future_df = create_feature(future_df)

final = pd.concat([data,future_df])

final = create_feature(final)
final_df = final.drop('prediction', axis=1)


# In[17]:


future_final = final_df.query('isFuture').copy()
future_final 


# In[18]:


future_final['Energy'] = RF.predict(future_final[FEATURES])
future_final.head()


# In[24]:


data


# In[19]:


future_final['Energy'].plot(figsize=(10,5), title='Future Energy Production Prediction')


# In[20]:


#2024 future trends
future_final.loc[(future_final.index > '01/01/2024') 
                 & (future_final.index <'12/30/2024')]['Energy'].plot(figsize=(15,5),title='2024 prediction')
plt.show()


# In[ ]:




