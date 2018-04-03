#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 14:37:55 2018

@author: erenzedeli
"""

# Kaggle Titanic Competetion 

import pandas as pd
import os
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns


# get the current working directory 
print (os.getcwd()) 

# Provide the path here
os.chdir('/Users/erenzedeli/Desktop/Kaggle - Titanic') 

# Read train and test dataset         
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Take a look at the dataset 
train.head()

#See how many observations and features in the dataset 
train.shape
test.shape

# Get info about features and their types 
train.info()
test.info()

#detect how many null records in columns
train.isnull().sum()
test.isnull().sum()

# Feature Engineering

# 1- Pclass
# No missing value on this feature and it is a numerical value
print (train[["Pclass", "Survived"]].groupby(['Pclass'], as_index=False).mean())


# 2- Sex
# Seems like gender has an effect on survival 
print (train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean())


# 3- SibSp and Parch 
# With the number of siblings/spouse and the number of children/parents
# we can create new feature called Family Size.

train['FamilySize'] = train['SibSp'] + train['Parch'] + 1

print (train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean())


# Now lets take a look if you are likely to survive if you are alone,
# which means you do not have family with you
train['IsAlone'] = 0
train.loc[train['FamilySize'] == 1, 'IsAlone'] = 1
               
# the impact on being alone is considerable     
print (train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())



# 4- Embarked
#the embarked feature has some missing value.
# I will fill those with the most occurred value ( 'S' )

# Find the most occurred value  
train['Embarked'].value_counts()
# Fill missing values with S 
train['Embarked'] = train['Embarked'].fillna('S')

# Determine the effect of Embarked on Survival 
print (train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean())

# 5- Fare 
#  I categorize Fare into 4 ranges
train['CategoricalFare'] = pd.qcut(train['Fare'], 4)
print (train[['CategoricalFare', 'Survived']].groupby(['CategoricalFare'], as_index=False).mean())

# 6- Age
# generate random numbers between (mean - std) and (mean + std)
# Then we categorize age into 5 range.

age_avg = train['Age'].mean()
age_std = train['Age'].std()
age_null_count = train['Age'].isnull().sum()
age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

train['Age'][np.isnan(train['Age'])] = age_null_random_list

train['CategoricalAge'] = pd.cut(train['Age'], 5)

print (train[['CategoricalAge', 'Survived']].groupby(['CategoricalAge'], as_index=False).mean())



# Data Cleaning
# Now let's clean our data and map our features into numerical values.

# Mapping Sex
train['Sex'] = train['Sex'].map( {'female' : 0 , 'male' : 1}). astype(int)

# Mapping Embarked
train['Embarked'] = train['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

# Mapping Fare
train.loc[ train['Fare'] <= 7.91, 'Fare'] = 0
train.loc[(train['Fare'] > 7.91) & (train['Fare'] <= 14.454), 'Fare'] = 1
train.loc[(train['Fare'] > 14.454) & (train['Fare'] <= 31), 'Fare'] = 2
train.loc[ train['Fare'] > 31, 'Fare'] = 3

train['Fare'] = train['Fare'].astype(int)


# Mapping Age
train.loc[ train['Age'] <= 16, 'Age'] = 0
train.loc[(train['Age'] > 16) & (train['Age'] <= 32), 'Age'] = 1
train.loc[(train['Age'] > 32) & (train['Age'] <= 48), 'Age'] = 2
train.loc[(train['Age'] > 48) & (train['Age'] <= 64), 'Age'] = 3
train.loc[train['Age'] > 64, 'Age'] = 4

train['Age'] = train['Age'].astype(int)
         
# Feature Selection

drop_elements_train = ['Name', 'Ticket', 'Cabin', 'SibSp',
                 'Parch', 'FamilySize','CategoricalFare','CategoricalAge',
                 'IsAlone','Embarked']  

drop_elements_test = ['Name', 'Ticket', 'Cabin', 'SibSp',
                 'Parch','Embarked']

train = train.drop(drop_elements_train, axis = 1)
test = test.drop(drop_elements_test, axis = 1)

# Clean the test set

# Fare

test['Fare'] = test['Fare'].fillna(test['Fare'].median())
 
test.loc[ test['Fare'] <= 7.91, 'Fare'] = 0
test.loc[(test['Fare'] > 7.91) & (test['Fare'] <= 14.454), 'Fare'] = 1
test.loc[(test['Fare'] > 14.454) & (test['Fare'] <= 31), 'Fare'] = 2
test.loc[ test['Fare'] > 31, 'Fare'] = 3
  
test['Fare'] = test['Fare'].astype(int)
        
         
# Sex
test['Sex'] = test['Sex'].map( {'female': 0 , 'male':1}).astype(int)

# Age 
age_avg = test['Age'].mean()
age_std = test['Age'].std()
age_null_count = test['Age'].isnull().sum()
age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

test['Age'][np.isnan(test['Age'])] = age_null_random_list


test.loc[ test['Age'] <= 16, 'Age'] = 0
test.loc[(test['Age'] > 16) & (test['Age'] <= 32), 'Age'] = 1
test.loc[(test['Age'] > 32) & (test['Age'] <= 48), 'Age'] = 2
test.loc[(test['Age'] > 48) & (test['Age'] <= 64), 'Age'] = 3
test.loc[test['Age'] > 64, 'Age'] = 4

test['Age'] = test['Age'].astype(int)

# Prediction

X_train = train.drop('Survived', axis=1).values
y_train = train['Survived'].values

      
X_test = test.values          

# train a logistic regression model on training set
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train,y_train)   

# make class predcition for the testing set
y_pred_class = logreg.predict(X_test)

pd.DataFrame({'PassengerId' : test.PassengerId, 'Survived' : y_pred_class}).to_csv('Titanic_Out_Put.csv')











              
              
