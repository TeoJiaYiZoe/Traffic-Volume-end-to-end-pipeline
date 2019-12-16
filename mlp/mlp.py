#!/usr/bin/env python
# coding: utf-8
#Date preprocessing

#importing necessery libraries for analysis of the dataset
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import datetime  
from datetime import date 
import calendar 
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_log_error
from math import sqrt
import time

#read csv file
traffic_data=pd.read_csv('https://aisgaiap.blob.core.windows.net/aiap5-assessment-data/traffic_data.csv')
#examing head of traffic data csv file 
traffic_data.head()

#Understanding the amount of of rows and columns we will be working with
traffic_data.shape

# Convert the type of 'date_time' column to a more appropriate type which is datetime 
traffic_data["date_time"] = pd.to_datetime(traffic_data["date_time"])
#examine the changes.
traffic_data.dtypes


#Understanding the information of the columns 
traffic_data.info()


#Double Checking to find out which columns have null values
traffic_data.isnull().sum()

#correlation between teh features
traffic_data.describe()


#Observe the value counts for each unique value in rain_1h.
traffic_data.rain_1h.value_counts().sort_index()


# Display the rain that has occurred in that 1h
traffic_data[traffic_data['rain_1h']>0]


#Calculating the percentage of data in 'rain_1h' that has value more than 0.0
(632/8572)*100


#To find out and confirm the unique value in 'snow_1h' is 0
traffic_data.snow_1h.unique()



# Extract just the month from the date_time column
traffic_data["start_date_month"] = traffic_data["date_time"].apply(lambda row: row.month)
# Extract just the hour from the date_time column
traffic_data["start_date_hour"] = traffic_data["date_time"].apply(lambda row: row.hour)
# Extract just the day of the week from the date_time column
traffic_data['dayofweek']=traffic_data['date_time'].dt.weekday_name
# Take a look at the date_time column and new month and hour columns
traffic_data[['date_time', 'start_date_month','start_date_hour','dayofweek']].head()



#Let the value be 1 if the day of the week is Sunday or Saturday else 0
traffic_data['is_weekend'] = np.where(traffic_data['dayofweek'].isin(['Sunday','Saturday']),1,0)
traffic_data[['date_time','is_weekend']].head()



#let the value to be 0 if the holiday is None else 1
traffic_data["holiday"]=traffic_data["holiday"].apply(lambda val: 0 if val=='None' else 1)
traffic_data[['holiday']].head()


#Examing some interesting categorical unique values
traffic_data.weather_main.unique()


#Encoding weather_main
data = [traffic_data]
weather_main_mapping = {"Clouds":0, "Snow":1, "Clear":2, "Mist":3, "Haze":4, "Fog":5, "Rain":6,"Drizzle":7, "Thunderstorm":8, "Squall":9}
for dataset in data:
    dataset['weather_main'] = dataset['weather_main'].map(weather_main_mapping)



#Encoding dayofweek
data = [traffic_data]
dayofweek_mapping = {"Monday":0, "Tuesday":1, "Wednesday":2, "Thursday":3, "Friday":4, "Saturday":5, "Sunday":6}
for dataset in data:
    dataset['dayofweek'] = dataset['dayofweek'].map(dayofweek_mapping)


#Observe the value counts for each unique value in weather main.
traffic_data.weather_main.value_counts().sort_index()


#Examing some interesting categorical unique values
traffic_data.weather_description.unique()



#Observe the value counts for each unique value in weather description.
traffic_data.weather_description.value_counts().sort_index()


# - Data is unbalanced. There are 1494 data for 'sky is clear' compared to 1 data for 'thunderstorm with light drizzle'. 
# - There is a need to standardize the words for 'Sky is Clear' and 'sky is clear'


#standardize words
traffic_data["weather_description"]= traffic_data["weather_description"].replace('Sky is Clear', "sky is clear")
#examing the changes
traffic_data.weather_description.unique()



#plt.figure(figsize=(20,10))
#x = traffic_data['weather_main']
#y = traffic_data['weather_description']
#plt.scatter(x, y, marker='o')
#plt.show()


# - 'weather_description' column is the subset of 'weather_main'. We should drop one of the columns when we train the dataset since there is a high association between these two features.



#Overall view of correlation between different features
#f,ax=plt.subplots(figsize=(10,10))
#sns.heatmap(traffic_data.corr(),annot=True,linewidths=0.5,linecolor="red",fmt=".3f",ax=ax)
#plt.show()



#dropping columns that have no values and significant for our future data exploration and predictions
traffic_data.drop(['snow_1h','rain_1h','date_time','weather_description'], axis=1, inplace=True)


#examining changes
traffic_data.head()



# Segregate traffic volume into 0,1,2 representing low,medium and high of traffic volume according to the dataset distribution
data = [traffic_data]

for dataset in data:
    dataset.loc[ dataset['traffic_volume'] <= 1193, 'traffic_volume'] = 0,
    dataset.loc[(dataset['traffic_volume'] > 1193) & (dataset['traffic_volume'] < 5001), 'traffic_volume'] = 1,
    dataset.loc[ dataset['traffic_volume'] >= 5100, 'traffic_volume'] = 2
    dataset['traffic_volume'] = dataset['traffic_volume'].astype(int)



#examining changes
traffic_data.head()


#Splitting of training and test data
X = traffic_data.loc[:, traffic_data.columns != 'traffic_volume']
y = traffic_data.loc[:, traffic_data.columns == 'traffic_volume'].values.ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


#Train random forest model
rf = RandomForestClassifier(n_estimators=150, random_state=42)
rf.fit(X_train, y_train)



#Print the report
expected = y_test
predicted = rf.predict(X_test)
acc = metrics.accuracy_score(expected, predicted)
print("Evaulation metrics:")
print("accuracy is "+ str(acc)) 
print("RMSLE value = ",np.sqrt(mean_squared_log_error(expected,predicted)))


# ##  Feature Selection


importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(rf.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
importances.head(8)

#importances.plot.bar()


# ##  Feature Reduction


#Splitting of training and test data
X2  = traffic_data.drop(['holiday','weather_main'], axis=1, inplace=True)
X2 = traffic_data.loc[:, traffic_data.columns != 'traffic_volume']
y = traffic_data.loc[:, traffic_data.columns == 'traffic_volume'].values.ravel()
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y, random_state=42)

# Random Forest

#Train random forest model
rf = RandomForestClassifier(n_estimators=150, random_state=42)
rf.fit(X_train2, y_train2)

#Print the report
expected2 = y_test
predicted2 = rf.predict(X_test2)
acc = metrics.accuracy_score(expected2, predicted2)
print("After feature reduction")
print("accuracy is "+ str(acc)) 
print("RMSLE value = ",np.sqrt(mean_squared_log_error(expected2,predicted2)))


# ### Model engineering



# Exploring the number of estimators in the random forest
#score = []
#est = []
#n_trees = np.arange(1, 260, 10) # build a array contains values form 1 to 100 with step of 4
#n_trees = [1, 10, 50, 100, 150, 200, 300, 400, 500]
#for n in n_trees:
    #rfc1 = RandomForestClassifier(n_estimators=n, random_state=42)
    #pred1 = rfc1.fit(X_train2, y_train2).predict(X_test2)
    #accuracy = metrics.accuracy_score(y_test2, pred1)
    #score.append(accuracy)
    #est.append(n)
#plt.figure(figsize=(20,5))
#plot = sns.pointplot(x=est, y=score)
#plot.set(xlabel='Number of estimators', ylabel='Accuracy', 
         #title='Accuracy score of RFC per # of estimators')
#plt.show()


# 61 estimators are enough for the dataset, and the accuracy is one of the highest.



# Exploring minimum leaf samples
#score = []
#leaf = []
#leaf_options = np.arange(1, 20, 1)
#for l in leaf_options:
    #rfc2 = RandomForestClassifier(random_state=42, min_samples_leaf=l, n_estimators=61)
    #pred2 = rfc2.fit(X_train2, y_train2).predict(X_test2)
    #accuracy = metrics.accuracy_score(y_test2, pred2)
    #score.append(accuracy)
    #leaf.append(l)
#plt.figure(figsize=(20,5))
#plot = sns.pointplot(x=leaf, y=score)
#plot.set(xlabel='Number of minimum leaf samples', ylabel='Accuracy', 
         #title='Accuracy score of RFC per # of minimum leaf samples')
#plt.show()


# Accuracy is highest when minimum leaf samples is 1.

# Exploring maximun depth
#score = []
#leaf = []
#max_depth = np.arange(1, 20, 1)
#for d in max_depth:
    #rfc2 = RandomForestClassifier(n_estimators=61, random_state=42, min_samples_leaf=1, max_depth=d)
    #pred2 = rfc2.fit(X_train2, y_train2).predict(X_test2)
    #accuracy = metrics.accuracy_score(y_test2, pred2)
    #score.append(accuracy)
    #leaf.append(d)
#plt.figure(figsize=(20,5))
#plot = sns.pointplot(x=leaf, y=score)
#plot.set(xlabel='Number of maximum depth', ylabel='Accuracy', 
         #title='Accuracy score of RFC per # of maximum depth')
#plt.show()


# Apparently, the accuracy is the highest when number of maximum depth is 19 or 18.
# However, overfit might occur if the the tree is too deep. 
# As a result, depth in 14 seems enough for genneral situation.

rfc = RandomForestClassifier(n_estimators=61, random_state=42, min_samples_leaf=1, max_depth=14)
rfc.fit(X_train2, y_train2)


#Print the report
expected3 = y_test2
predicted3 = rfc.predict(X_test2)

print("After Model Enginnering:")
acc2 = metrics.accuracy_score(expected3, predicted3)
print("accuracy is "+ str(acc2)) 
print("RMSLE value = ",np.sqrt(mean_squared_log_error(expected3,predicted3)))


# Compared to the previous accuracy 0.9277, the accuracy has increased after model engineering (adjusting the number of estimators, minimum leaf samples and maximum depth).

# ## K-Fold Cross Validation


from sklearn.model_selection import cross_val_score
scores = cross_val_score(rfc, X_train2, y_train2, cv=5, scoring = "accuracy")
print("K-Fold Cross Validation:")
print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())


# K-Fold Cross Validation has a average accuracy of 92.9% with a standard deviation of 1.67 %.
