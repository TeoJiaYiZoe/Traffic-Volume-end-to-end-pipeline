## End to End Machine Learning Pipeline

### Define the problem

What affect the traffic volume in the year of 2013?

### Data Preparation

After loading the dataset, we can observe the data types of the column.

As noticed that the 'date_time' is an object, thus we will convert it to the type 'datetime'.

We will also check if there are null values.

We can also get a statistical view of the data to observe is there any anomalies. We observed that the there maybe anomalies in some features. For example, for the 'snow_1h' column, the minimum, maximum, mean value is 0.0. In addition, under the' rain_1h' column, there is a mean of 0.161 but the maximum value is 55.63. Hence we will do a further analysis by observing the count of each unique value and when 'rain_1h' is more than 0.

We observed that the 'date_time 'column include both the date and the time. Hence we can split it into 3 column, namely ' start_date_month', 'start_date_hour' and 'dayofweek' for better analysis. 

As the column 'holiday 'consist of none and the national or regional holidays name, we will change it to binary number as the name of holidays are not an importance feature but whether is it a holiday is of concern to us. Therefore, binary number is used in this case.

We can observed that there are 2 categorical column, namely 'weather_main', 'dayofweek', thus we will encode it using mapping to prepare for training it later.

By observing the unique value of 'weather_description', there is a duplicate values 'sky is clear' and 'Sky is Clear'. Thus we will standardize the words to 'sky is clear'.

We will drop the feature 'snow_1h', 'rain_1h', 'date_time', 'weather_description'.

We observed that the 'traffic_volume ' has a large range and the numbers are bigger than the rest thus we can segregate it to low, medium and high. 25% of dataset is value has a value of 1193 or less, thus it will be bin to a value of 0. Above 75% of dataset will have a value of 2 while the in between range value is 1.

### Model Training and Prediction

We will split the data into X, y. Feature 'traffic_volume' is our target (y). The other features, namely, 'holiday', 'temp', 'clouds_all', 'weather_main', 'start_date_month', 'start_date_hour', 'dayofweek', 'is_weekend' will be x variables that affects y.

We will train our model using random forest because the bias of random forest  algorithm is reduced due to the power of "the crowd" as there are multiple trees and each tree is trained on a subset of data. In addition, random forest works well when you have both categorical and numerical features.  

 Another useful feature of the Random Forest method is its estimation of relative predictor importance. The method is based on measuring the effect of the classifier if one of the predictors was removed thus this helps us to do feature reduction at a later timing. 

The evaluation metrics we will be using is accuracy and RMSLE ( Root Mean Squared Logarithmic Error ). Accuracy refer to the correctness of our model. The RMSLE is measuring the precise of the estimate. Hence, the lower the RMSLE and the higher the accuracy will our model be better.

After training, evaluation metric of our model: 

```
accuracy is 0.9188432835820896
RMSLE value =  1.1166235040423258
```

## Feature Selection

By using feature importance, we can understand which feature is the most important to the training of the dataset.  We observe that holiday has 0.0.  This mean that it is does not affect traffic volume at all. We will be dropping weather_main and holiday since there are the lowest in this feature selection.

## Feature Reduction

After feature reduction, evaluation metric of our model: 

```
accuracy is 0.9277052238805971
RMSLE value =  1.0554852276233961
```

we noticed that the accuracy increased and RMSLE value has fallen. This mean our model has improved. 

## Model Engineering

To further enhance our model, we can do model engineering to determine the optimal number of estimators,minimum leaf samples and maximum depth. We noticed that when optimal number of estimators=61 ,minimum leaf samples=1.  As overfitting may occur, we will not take the highest accuracy for maximum depth (19) instead we will use 14.

After model engineering, evaluation metric of our model: 

```
accuracy is 0.9305037313432836
RMSLE value =  1.1139001601543737
```

We observe that the accuracy has improved and RMSLE has fallen a little. it makes sense as we did not use the most accurate maximum depth since we want to prevent overfitting.

## K-Fold Cross Validation

Cross-validation is a resampling procedure used to evaluate machine learning models on a limited data sample.

We will be using this to estimate the skill of a machine learning model on unseen data. That is, to use a limited sample in order to estimate how the model is expected to perform in general when used to make predictions on data not used during the training of the model.

```
Scores: [0.89841986 0.92807425 0.9372549  0.94632991 0.93913043]
Mean: 0.9298418720847259
Standard Deviation: 0.016754420116516695
```

We observed K-Fold Cross Validation has a average accuracy of 92.9% with a standard deviation of 1.67 %. In conclusion, our model has performed quite good.







