# credit-risk-classification

## Overview of the Analysis
Determine whether a loan would be healthy or high-risk depending on the user's credit information

### Financial information data used

The information used for creating this analysis is the following:

- loan_size 
- interest_rate
- borrower_income
- debt_to_income
- num_of_accounts
- derogatory_marks
- total_debt
- loan_status

A value of 0 in the “loan_status” column means a healthy loan whereas a 1 in the "loan_status" means a high-risk loan. The analysis is based on this variable to predict the model.

Data includes 75036 samples of healthy status and 2500 of high-risk.


### Process stages for the machine learning algorithms 

- The data was loaded from a CSV file 
- The data was split between features which consisted of all the input variables (X) and the the loan_status (y).
- The dataset was further split into train and test set using sklearn's train_test_split function.


### Methods used

2 models that used `LogisticRegression`from sklearn were used for the analysis. The first model used the original dataset split into training and testing datasets.

In the second model a `LogisticRegression` was also used, but the training dataset was enlarged with a RandomOverSampler module from imbalanced-learn. This attempted to address the imbalance between Healthy and High-Risk samples as described above (75036 healthy status samples vs 2500 high-risk).

## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

### Machine Learning Model 1:

`LogisticRegression` with originally split dataset

#### Confusion matrix model 1:

|18663|   102|
|--- | ---|
|56|563|


#### Metrics Learning model 1:

              precision    recall  f1-score   support

           0       1.00      0.99      1.00     18765
           1       0.85      0.91      0.88       619

    accuracy                           0.99     19384
   macro avg       0.92      0.95      0.94     19384
weighted avg       0.99      0.99      0.99     19384

Both Healthy and High-Risk loans were predicted with over 85% of accuracy. 15% if the time the High-risk loans were predicted as healthy (which was wrong).
The total model accuracy is 95.20%. 


### Machine Learning Model 2:

`LogisticRegression` with added samples using the `RandomOverSampler` module from the imbalanced-learn library to resample the data. The healthy and high-risk loans have an equal number of data points. 

#### Confusion matrix model 2:

|18649|   116|
|--- | ---|
|4|   615|

#### Metrics Learning model 2:
              precision    recall  f1-score   support

           0       1.00      0.99      1.00     18765
           1       0.84      0.99      0.91       619

    accuracy                           0.99     19384
   macro avg       0.92      0.99      0.95     19384
weighted avg       0.99      0.99      0.99     19384

For this model, the accuracy is higher eventhough the high risk loan was mislabeled 16% of the times.

## Summary

The performance of the two models is similar, with an accuracy of 95.20% for the first model and 99.38% for the second.

The model with the RandomOverSampler improved overall the recognition of the High-Risk as High-Risk from 91% to 99%. The false negatives and false positives in both models are very similar. 

Based on the analysis, Model 2 outperforms Model 1 in predicting high-risk loans and has an overall higher accuracy. Specifically, Model 2 achieved a relatively high precision in predicting high-risk loans while correctly identifying all high-risk loans in the dataset, which is considered a relatively good performance in this context. Therefore, I would recommend using Model 2 in identifying high-risk loans and overall better accuracy in predicting labels.
