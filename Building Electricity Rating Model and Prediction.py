#!/usr/bin/env python
# coding: utf-8

#### Building Electricity Rating Prediction Model 


#Importing Libraries for evaluation and visualization
import matplotlib.pylab as plt
import pprint
import numpy as np
import pandas as pd
import seaborn as sns
color_pal = sns.color_palette()


#### Exploratory Data Analysis 

#Importing file as csv
file_path = 'C:\Dilip\Business Analytics\MP\Model\L_BER_rating.csv'
df = pd.read_csv(file_path)
df.shape
df.columns
df.head()


#Checking for NULL values and cleaning dataset
df.isnull().sum()
df = df.dropna()

#Renaming column name and setting date column as index
from datetime import datetime
df.rename(columns = {'Year_of_Construction':'Year'}, inplace = True)
df.head()

df['Year']=pd.to_datetime(df['Year'])
df = df.set_index('Year')
df


#Checking min and max data in the dataset
print(df.index.min())
print(df.index.max())
#df.tail()
df.isnull().sum()


#Train and Test Data split
train = df.loc[df.index <  '2011-01-01']
test = df.loc[df.index > '2011-01-01']
print("Train: ",train.count())
print("Test: ",test.count())



#Plotting Train and Test data
plt.figure(figsize = (10,7))
plt.plot(train['BerRating'],label = 'Train')
plt.plot(test['BerRating'], label = 'Test')
plt.legend(loc='best')



#Feature creation 
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

#Adding the features to the train and test data
train = create_feature(train)
test = create_feature(test)

test_final = create_feature(test)
print("Train: ", train.shape)
print("Test: ", test.shape)


features = ['dayofweek', 'quarter', 'month', 'year', 'dayofyear']
target = ['BerRating']


#Splitting train and test based on feature and target
x_train = train[features]
y_train = train[target]

x_test = test[features]
y_test = test[target]
x_train


#### Random Forest Regressor Model
from sklearn.ensemble import RandomForestRegressor

#Train and fit the model for prediction
RF = RandomForestRegressor(max_depth=40,random_state=0)
RF.fit(x_train,y_train)
RF_predict = RF.predict(x_train)
plt.scatter(RF_predict,y_train)

#Importing libraries
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn import metrics
#R2 square error
rf_r2_value = metrics.r2_score(y_train,RF_predict)
print("r-sqaure: ",rf_r2_value)


#Merging preditcted and actual data
train['prediction'] = RF.predict(x_train)
data = df.merge(train['prediction'], how='left', left_index=True, right_index = True)
data

#Plot predcition with actual data
plt.figure(figsize = (15,5))
plt.plot(data['BerRating'],label = 'original')
plt.plot(data['prediction'], label = 'prediction')
plt.legend(loc='best')


#### Future Prediction 

#Preciting future value from 2024 to 2025
future = pd.date_range('2024-01-01','2025-12-31', freq='1m')
future_df = pd.DataFrame(index = future)

future_df['isFuture'] = True
data['isFuture'] = False
future_df = create_feature(future_df)

final = pd.concat([data,future_df])
final = create_feature(final)

final_df = final.drop('prediction',axis=1)
final_df

future_value = final_df.query('isFuture').copy()
future_value 

features = ['dayofweek', 'quarter', 'month', 'year', 'dayofyear']
target = ['BerRating']
future_value['BerRating'] = RF.predict(future_final[features])
future_value.head()


#Plotting the future results
future_value['BerRating'].plot(figsize=(10,5), title='Future Energy Production Prediction')


#2024 BER prediction
future_value.loc[(future_value.index > '01/01/2024') 
                 & (future_value.index <'12/30/2024')]['BerRating'].plot(figsize=(15,5),title='BER 2024 prediction')
plt.show()

#########################################################################################################################


