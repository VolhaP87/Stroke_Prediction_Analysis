![](images/brain.jpeg)

# Stroke Prediction Analysis
Authors: Volha Puzikava
***

## Overview
Every 40 seconds, someone in the United States has a stroke. Every 3.5 minutes, someone dies of stroke. Every year, more than 795,000 people in the United States have a stroke. According to the World Health Organization (WHO), stroke is the 2nd leading cause of death globally, responsible for approximately 11% of total deaths. Moreover, stroke is a leading cause of serious long-term disability.

This project tends to predict whether a patient is more likely to get a stroke during their lifetime and thus, helps to reduce stroke occurence and prevent the illness before it happens. 
***

## Business Problem
The World Health Organization asked to analyze the stroke dataset and provide information about what parameters likely increase the occurence of stroke in people, so the people with the higher chance of getting stroke can be monitored more often in order to prevent the illness before it strikes. The main purpose of the analysis was to build different machine learning algorithms and choose the one that has the highest performance rate predicting the stroke occurence.
***

## Data Understanding
The data for the analysis was taken from [kaggle website](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset?select=healthcare-dataset-stroke-data.csv). The data provided 11 clinical features for predicting stroke effect: gender and age of the patients, their marital status, work type, residence type, smoking status, the precense of hypertension and heart disease, average glucose level and body mass index. The dataset contained information about 5,110 patients.
***

## Data Preparation and Exploration
The first step was to check how imbalanced the dataset was. Since the column "stroke" served as the indicator of weather or not a patient had a stroke, the distribution of values in that column was calculated. It turned out that the dataset was imbalanced: 95% of people did not have a stroke, while only 5% had. 

The dataset was then checked for the presence of missing values. They were present only in "bmi" column and were replaced with the column's mean value.

The distribution of stroke occurence was then checked among the numerical features of the dataset. It was found out that the average age of people with stroke is around 67 years, those people are more likely to have hypertension and heart disease and their glucose level is much higher than in people without a stroke. Also, body mass index turned out to be not of a much influence. The pairplot was then used to visualize the relationship betweeen the variables.

The stroke incidences were also compared by gender, marital status, work type, residence type and smoking status. It turned out that people who are married, self-employed and formely smokers are more likely to develop stroke in their lifetime. Gender and residence type do not influence much on the occurence of the illness.

In order to be used in a model, all categorical variables ("gender", "ever_married", "Residence_type", "work_type" and "smoking_status") had to be transformed. The first three mentioned columns had the values replaced to numerical ones, while the remaining two columns used dummy variables.

It was also important to see if the predictive features would result in multicollinearity in the final model. With that in mind, pearson correlation coefficients of the predictive features were generated and visualized. According to the heatmap, the highest correlation belonged to "ever_married" and "age" (0.68). The most strongly correlated features with the target variable were "age", "hypertension", "heart_disease", "ever_married" and "avg_glucose_level".
***

## Data Modeling
In order to build any ML model, a train-test split should be performed. The prediction target for this analysis was the column "stroke", so the data was separated into a train set and test set accordingly. The "id" column was dropped since it represented a unique identifier, not an actual numeric feature.

The class imbalance in the training and test set was then found and SMOTE-NC class was used in order to improve the models' performance on the minority class. SMOTE-NC was applied because the dataset contained both categorical and numerical features.

Since the World Health Organization really cares about avoiding 'false negatives' more than avoiding 'false positives' (it is a crime to say that a person will not have a stroke, and then he/she will develop it, than predict that the person will have a stroke and he/she will not actually have it), higher recall score and lower number of false negatives will be the metrics the ML models will be evaluated upon. The model with the highest recall will be chosen for the prediction.

Different machine learning algorithms will be built in the following way:
- the baseline model will be build and evaluated;
- one or more hyperparameters will be tuned to find if the model will perform any better;
- the optimized model will be run and checked for any improvements in the performance;
- the model with the highest recall score for each algorithm type will be chosen for further analysis.

#### Logistic Regression Models
The logistic regression baseline model with default parameters had recall of 74%, meaning that if a person belongs to class 1 (having a stroke), there is about 74% chance that the model will correctly label this person as class 1. The number of false negatives was equal to 16. The accuracy of the model constituted 73%, meaning that the model correctly identifies if a person will have a stroke about 73% of the time.
The logistic regression tuned model (penalty='l2', solver='liblinear') performed a little better: the recall score got higher (76%), while the number of false negatives got a little less (15). The logistic regression tuned model was chosen for future analysis.

#### Decision Tree Models
The decision tree baseline model had a recall of 34%, which was less than the logistic regression models. The number of false negatives got higher and became equal to 41. The model used all the features except "work_type_Never_worked" and "work_type_children" with the most important being "age", "bmi" and "avg_glucose_level". The model was then optimized to check if better recall could be achieved.
The decision tree tuned model (max_depth=4, min_samples_split=18, min_samples_leaf=9, max_features=13) performed much better than the baseline model. It showed the recall of 74% with false negatives being equal to 16. The model considered "age", "work_type_Private", "bmi" and "avg_glucose_level" as the most important features. The model was chosen for future analysis.

#### Bagged Trees Models
The bagged tree baseline model (n_estimators=100) didn't perform well enough. The recall score was equal to 34%, while the number of false negatives became 41. The model got to be tuned to perform better.
The bagged tree tuned model (max_features=2) didn't perform differently: the recall score stayed the same 34%, while the number of false negatives remained equal to 41. But because the accuracy score of the bagged tree tuned model was higher, the model was chosen for further analysis.

#### Random Forest Models
The random forest baseline model didn't perform well enough: it had a low recall score (32%) with the high number of false negatives (42). The model gave the strongest importance to "age", "avg_glucose_level" and "bmi" columns. The model was then tuned to find if it could perform any better.
The recall score of random forest tuned model (max_features=7) got a little better (35%), and the number of false negatives got a little bit lower (40). The model considered "age", "bmi", and "avg_glucose_level" as the most important features and didn't take into consideration "work_type_Never_worked". The random forest tuned model was chosen for further analysis.