#!/usr/bin/env python
# coding: utf-8

# ### XGBoosting Regression Model

# In[71]:


#importing libraries
import matplotlib.pylab as plt
import pprint
import numpy as np
import pandas as pd
import seaborn as sns
color_pal = sns.color_palette()
from sklearn.metrics import mean_squared_error


# In[72]:


#importing energy data as CSV file
file_path = 'C:\Dilip\Business Analytics\MP\Model\energy_daily.csv'
df = pd.read_csv(file_path)


# ### Exploratory Data Analysis

# In[73]:


print("The First 5 records in the dataset--->\n",df.head())
print("The Last 5 records in the dataset--->\n",df.tail())
df.shape


# In[74]:


#Renaming column name
df.rename(columns = {'Energy MW':'Energy'}, inplace = True)
df.columns


# In[75]:


#Setting date as index column
df = df.set_index('date')
df.index


# In[76]:


#Converting index from object to datetime
df.index=pd.to_datetime(df.index)
print("MAX date: ",df.index.max())
print("MIN date: ", df.index.min())


# In[77]:


#Energy Production graph from 2015 - 2022
df.plot(figsize=(15,5),
        color = color_pal[0],
        title='Energy Production 2015 - 2022')


# In[78]:


#Creating train and test data for model building
train = df.loc[df.index <  '7/1/2020']
test = df.loc[df.index > '7/1/2020']
print("Training Data:", train.shape)
print("Testing Data: ",test.shape)


# In[79]:


#Plotting train and test data (80:20 precent)
fig, ax = plt.subplots(figsize=(15,5))
train.plot(ax=ax, label='Training set', title='Train/Test split')
test.plot(ax=ax, label='Test set')
ax.legend(['Training set','Test set'])
plt.show()


# In[80]:


#Plotting one week data to check the model prediction accuracy
df.loc[(df.index > '01/05/2019') & (df.index <'01/19/2019')]['Energy'].plot(figsize=(15,5),title='January 2 weeks of Data')


# In[81]:


#Function for Feature Creation 
def create_feature(df):
    df = df.copy()
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    return df

df = create_feature(df)

df.head()


# In[82]:


#Applying the feature creation fuction to train and test data
train = create_feature(train)
test = create_feature(test)

test_final = create_feature(test)
print("Train: ", train.shape)
print("Test: ", test.shape)


# In[83]:


#Column names
features = ['dayofweek', 'quarter', 'month', 'year', 'dayofyear']
target = ['Energy']


# In[84]:


#Creating x_train, y_train, x_test, y_test for source and target respectively
x_train = train[features]
y_train = train[target]

x_test = test[features]
y_test = test[target]


# In[86]:


print("Y_train",y_train.head())


# ### XGBooster Regressor Model Building 

# In[29]:


import xgboost as xgb
from xgboost import XGBRegressor 

#XGBoost Regressor model
#n_estimators = number of boosting rounds
#early_stoppings = avoid model performance degradation
#learning_rate = Avoid overfitting
#verbose = Print validation scores every 100 trees

reg = xgb.XGBRegressor(n_estimators = 1000,early_stopping_rounds=50,
                      learning_rate=0.01)
reg.fit(x_train,y_train,
       eval_set=[(x_train,y_train), (x_test,y_test)],
        verbose = 100) 


# In[30]:


#Prediction model
xgb_predict = reg.predict(x_train)


# In[31]:


#Creating dataframe for plotting
xgb_predict = pd.DataFrame(xgb_predict,columns=['Energy'])


# In[32]:


#Plotting the predicted values
xgb_predict.plot(figsize=(15,5),
        color = color_pal[0],
        title='Energy distrbution')


# In[33]:


#Plotting the y_train values
y_train.plot(figsize=(15,5),
        color = color_pal[0],
        title='Energy distrbution')


# In[34]:


train['prediction'] = reg.predict(x_train)


# In[35]:


#Merging two data frame with energy and prediction
data = df.merge(train['prediction'], how='left', left_index=True, right_index = True)


# In[36]:


data


# In[90]:


#plotting actual data and predicted data
plt.figure(figsize = (15,5))
plt.plot(data['Energy'],label = 'original')
plt.plot(data['prediction'], label = 'prediction')
plt.legend(loc='best')
plt.title("Orginal Vs Predcited Energy value")


# In[93]:


#Comparing one week of actual data and predicted data
data.loc[(data.index > '12/30/2019') & (data.index <'1/30/2020')]['Energy'].plot(figsize=(15,5),title='2019 One week data')
data.loc[(data.index > '12/30/2019') & (data.index <'1/30/2020')]['prediction'].plot(style='-')
plt.legend(['Original','Prediction'])
plt.show()


# ### Model tuning using lag features

# In[39]:


file_path = 'C:\Dilip\Business Analytics\MP\Model\energy_daily.csv'
df = pd.read_csv(file_path)
df.rename(columns = {'Energy MW':'Energy'}, inplace = True)
df = df.set_index('date')
df.index=pd.to_datetime(df.index)


# In[40]:


train = df.loc[df.index <  '7/1/2020']
test = df.loc[df.index > '7/1/2020']
print(train.shape)
print(test.shape)
df


# In[41]:


#Time series cross validation
from sklearn.model_selection import TimeSeriesSplit


# In[42]:


tss = TimeSeriesSplit(n_splits = 5)
df = df.sort_index()

for train_idx,val_idx in tss.split(df):
    break

df = create_feature(df)


# In[43]:


for train_idx,val_idx in tss.split(df):
    break


# In[44]:


df = create_feature(df)


# In[45]:


#Lag features
target_map = df['Energy'].to_dict()


# In[46]:


def add_lags(df):
    df['lag1'] = (df.index -pd.Timedelta('365 days')).map(target_map)
    df['lag2'] = (df.index -pd.Timedelta('728 days')).map(target_map)
    df['lag3'] = (df.index -pd.Timedelta('1092 days')).map(target_map)
    return df

df = add_lags(df)
df.tail()


# In[47]:


#Train model with lags
tss = TimeSeriesSplit(n_splits = 5)
df = df.sort_index()

fold = 0
preds = []
scores = []

for train_idx, val_idx in tss.split(df):
    train = df.iloc[train_idx]
    test = df.iloc[val_idx]
    
    train = create_feature(train)
    test = create_feature(test)
    
    features = ['dayofweek', 'quarter', 'month', 'year', 'dayofyear','lag1',
               'lag2','lag3']
    target = ['Energy']
    
    x_train = train[features]
    y_train = train[target]
    
    x_test = test[features]
    y_test = test[target]
    
    reg = xgb.XGBRegressor(base_score = 0.5,
                           booster='gbtree',
                           n_estimators = 1000,
                           early_stopping_rounds=50,
                           objective='reg:linear',
                           max_depth=3,
                           learning_rate = 0.01)
    reg.fit(x_train,y_train,
            eval_set=[(x_train,y_train), (x_test,y_test)],
            verbose = 100) 
    
    y_pred = reg.predict(x_train)
    preds.append(y_pred)
    score = np.sqrt(mean_squared_error(y_train,y_pred))
    scores.append(score)


# In[48]:


prod_df = pd.DataFrame(y_pred)


# In[49]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn import metrics

#R2 square error
rf_r2_value = metrics.r2_score(y_train,prod_df)
#calculating mean square error
rf_mean_square_value = metrics.mean_squared_error(y_train,prod_df)
#calculating root mean square error
rf_root_mean_square_value = np.sqrt(rf_mean_square_value)

print("r-sqaure: ",rf_r2_value)
print("RMSE: ",rf_root_mean_square_value)


# In[50]:


#Converting the y_pred values into a dataframe
y_pred = pd.DataFrame(y_pred,columns=['Energy'])


# In[51]:


y_pred


# In[94]:


#PLotting y_pred values
y_pred.plot(figsize=(15,5),
        color = color_pal[0],
        title='Energy Production using parameter tuning')


# In[ ]:




