# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 22:15:03 2018
"""

# Core Libraries - Data manipulation and analysis
import pandas as pd
import numpy as np
import math
from math import sqrt
import matplotlib.pyplot as plt
import seaborn as sns
  
# Core Libraries - Machine Learning
import sklearn
import xgboost as xgb


# Importing Classifiers - Modelling
from sklearn.linear_model import LogisticRegression
from xgboost.sklearn import XGBClassifier

## Importing train_test_split,cross_val_score,GridSearchCV,KFold - Validation and Optimization
from sklearn.model_selection import  train_test_split, cross_val_score, GridSearchCV, KFold 


# Importing Metrics - Performance Evaluation
from sklearn import metrics


# Load Data
train_set = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data', header = None)
test_set =  pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test', skiprows = 1, 
                header = None)
col_labels = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation','relationship',
                  'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'wage_class']
train_set.columns = col_labels
test_set.columns = col_labels


# Understand the Dataset and Data


train_set.shape,test_set.shape

train_set.columns

train_set.head()

test_set.columns

test_set.head()

train_set.info()

test_set.info()

train_set.get_dtype_counts()

test_set.get_dtype_counts()


# Clean the data

#Clean Column Names

train_set.columns

test_set.columns


#The columns don't have any nonsensical values, therefore there is no need to clean or change column names

# Clean Numerical Columns

# Null values

num_cols = train_set.select_dtypes(include="int64").columns.values
# num_cols = test_set.select_dtypes(include="int64").columns.values can also be used because the columns are the same

train_set[num_cols].isna().sum()

test_set[num_cols].isna().sum()


# No null values in the numerical columns of both the train_set and test_set

# Zeros

#Check if there are any rows with all row values = zero that need our consideration so that we can decide to study those rows

train_set.loc[(train_set==0).all(axis=1),num_cols].shape

test_set.loc[(train_set==0).all(axis=1),num_cols].shape


#There are no rows which have all row values == 0

#Check if there are any rows with any row values = zero that need our consideration so that we can decide to study those rows

train_set.loc[(train_set==0).any(axis=1),num_cols].shape

train_set.loc[(train_set==0).any(axis=1),num_cols].head()

train_set.loc[(train_set.drop(["capital_gain", "capital_loss"],axis=1)==0).any(axis=1),num_cols].shape

test_set.loc[(train_set==0).any(axis=1),num_cols].shape

test_set.loc[(test_set.drop(["capital_gain", "capital_loss"],axis=1)==0).any(axis=1),num_cols].shape


#There are no rows which have any row values == 0, except in captital_gain, capital_loss columns(where 0 is a valid value)

# Nonsensical values

# Clean Categorical Columns

# Null values

cat_cols = train_set.select_dtypes(include="object").columns.values
cat_cols


train_set[cat_cols].isna().sum()


test_set[cat_cols].isna().sum()


# Empty Values

train_set.loc[(train_set=="").any(axis=1),cat_cols].shape


test_set.loc[(train_set=="").any(axis=1),cat_cols].shape
#There are no empty strings in any of the rows


#Nonsensical values 

train_set[cat_cols].nunique()

for col in cat_cols:
    print(train_set[col].unique(),"\n")


#The columns workclass, occupation and native_country have rows that have garbage values which need to be imputed or dropped in the train_set

test_set['workclass'].unique()

for col in cat_cols:
    print(test_set[col].unique(),"\n")


#The columns workclass, occupation and native_country have rows that have garbage values which need to be imputed or dropped in the test_set

plt.figure(figsize=(20,10))
plt.subplot(2,2,1) 
plt.title("Workclass Count Distribution")
train_set['workclass'].value_counts().plot.bar()

plt.subplot(2,2,2) 
plt.title("Occupation Count Distribution")
train_set['occupation'].value_counts().plot.bar()


plt.figure(figsize=(20,5))
plt.subplot(1,1,1) 
plt.title("Native Country Count Distribution")
train_set['native_country'].value_counts().plot.bar()


plt.figure(figsize=(20,10))
plt.subplot(2,2,1) 
plt.title("Workclass Count Distribution")
test_set['workclass'].value_counts().plot.bar()

plt.subplot(2,2,2) 
plt.title("Occupation Count Distribution")
test_set['occupation'].value_counts().plot.bar()


plt.figure(figsize=(20,5))
plt.subplot(1,1,1) 
plt.title("Native Country Count Distribution")
test_set['native_country'].value_counts().plot.bar()


train_set[train_set.workclass.str.contains("\?")].head()

test_set[test_set.workclass.str.contains("\?")].head()


(train_set.loc[(train_set==" ?").any(axis=1),cat_cols].shape[0]/train_set.shape[0])*100


(test_set.loc[(test_set==" ?").any(axis=1),cat_cols].shape[0]/test_set.shape[0])*100


#If we drop the rows containing ? values, we incur a data loss of approximately 7.5% data loss in the train_set and the test_set. Therefore we choose to drop it
train_set.drop(train_set.loc[(train_set==" ?").any(axis=1)].index, inplace= True)
train_set.shape[0]

test_set.drop(test_set.loc[(test_set==" ?").any(axis=1)].index, inplace= True)
test_set.shape[0]

test_set.loc[(test_set==" ?").any(axis=1),cat_cols].shape[0]/test_set.shape[0]


#Get Basic Statistical Information

train_set.describe()

train_set.describe(include='object')


test_set.describe()


test_set.describe(include='object')


train_set.corr()


test_set.corr()


# Explore Data more
# Uni-variate
train_set[num_cols].hist(bins=50, figsize=(20,20), layout=(4,2))
plt.show()

test_set[num_cols].hist(bins=50, figsize=(20,20), layout=(3,2))
plt.show()


#Categorical Columns
for i, col in enumerate(cat_cols):
    if(col!='native_country'):
        plt.figure(i,figsize = (20,5))
        sns.countplot(y=col, data=train_set,)
    else:
        plt.figure(i,figsize = (20,10))
        sns.countplot(y=col, data=train_set)
for i, col in enumerate(cat_cols):
    if(col!='native_country'):
        plt.figure(i,figsize = (20,5))
        sns.countplot(y=col, data=test_set)
    else:
        plt.figure(i,figsize = (20,10))
        sns.countplot(y=col, data=test_set)


#Bi-variate
sns.pairplot(train_set[num_cols],kind ='reg',diag_kind='kde')
sns.pairplot(test_set[num_cols],kind ='reg',diag_kind='kde')
#None of the numerical columns are strongly correlated with each other, either in train_set or test_set. However, it is interesting to note that education is more correlated with capital_gain than capital_loss


for i, col in enumerate(num_cols):
    plt.figure(i,figsize = (20,5))
    sns.violinplot(x=col,y='wage_class', data=train_set)

for i, col in enumerate(num_cols):
    plt.figure(i,figsize = (20,5))
    sns.violinplot(x=col,y='wage_class', data=test_set)


#Multi-variate
plt.figure(figsize=(10,10))
sns.heatmap(train_set.corr(), annot = True,cmap= "PRGn")
plt.figure(figsize=(10,10))
sns.heatmap(test_set.corr(), annot = True,cmap= "PRGn")


# Select Features

#Convert Categorical Columns

for col in train_set.columns: # Loop through all columns in the dataframe
    if train_set[col].dtype == 'object': # Only apply for columns with categorical strings
        train_set[col] = pd.Categorical(train_set[col]).codes # Replace strings with an integer

for col in test_set.columns: # Loop through all columns in the dataframe
    if test_set[col].dtype == 'object': # Only apply for columns with categorical strings
        test_set[col] = pd.Categorical(test_set[col]).codes # Replace strings with an integer



#Generate Input Vector X and Output Y, and Split the Data for Training and Testing

x_train = train_set.drop('wage_class', axis =1)
y_train = train_set['wage_class']
x_test = test_set.drop('wage_class', axis =1)
y_test = test_set['wage_class']

x_train.shape, y_train.shape, x_test.shape, y_test.shape
#Fit the Base Models and Collect the Metrics



#Logistic Regression
log_res = LogisticRegression()
model_lr = log_res.fit(x_train, y_train)
y_test_pred = model_lr.predict(x_test)

y_test_pred_prob = model_lr.predict_proba(x_test)
model_lr.score(x_test,y_test)



# Generate model evaluation metrics for the Logistic Regression
print("Performance metrics of the model for the Logistic Regression")
print("-"*100)
print("Accuracy: ", metrics.accuracy_score(y_test, y_test_pred))
print("Precision Score: ",metrics.precision_score(y_test, y_test_pred))
print("Recall Score: ",metrics.recall_score(y_test, y_test_pred))
print("AUROC Score: ",metrics.roc_auc_score(y_test,  y_test_pred_prob[:,1]))
print()
print("Confusion Matrix:  \n ",metrics.confusion_matrix(y_test, y_test_pred))
print()
print("Classification Report:\n ",metrics.classification_report(y_test, y_test_pred))



#XGBoost Base Model
params = {'learning_rate': 0.1, 'n_estimators': 1000, 'seed':0, 'subsample': 0.8, 'colsample_bytree': 0.8, 
             'objective': 'binary:logistic'}
XGB_base = XGBClassifier(**params)

XGB_base.fit(x_train, y_train)

y_test_pred = XGB_base.predict(x_test)
y_test_pred_prob = XGB_base.predict_proba(x_test)

XGB_base.score(x_test,y_test)

# Generate model evaluation metrics for the XGBOOST- Base Model
print("Performance metrics of the model for the XGBOOST- Base Model")
print("-"*100)
print("Accuracy: ", metrics.accuracy_score(y_test, y_test_pred))
print("Precision Score: ",metrics.precision_score(y_test, y_test_pred))
print("Recall Score: ",metrics.recall_score(y_test, y_test_pred))
print("AUROC Score: ",metrics.roc_auc_score(y_test,  y_test_pred_prob[:,1]))
print()
print("Confusion Matrix:  \n ",metrics.confusion_matrix(y_test, y_test_pred))
print()
print("Classification Report:\n ",metrics.classification_report(y_test, y_test_pred))



#Select Features
xgb.plot_importance(XGB_base)

importance = pd.DataFrame.from_dict({'cols':x_train.columns, 'importance': XGB_base.feature_importances_})
importance = importance.sort_values(by='importance', ascending=False)
plt.figure(figsize=(10,10))
sns.barplot(importance.cols, importance.importance)
plt.xticks(rotation=90)
imp_cols = importance[importance.importance >= 0.03].cols.values
imp_cols
params = {'learning_rate': 0.1, 'n_estimators': 1000, 'seed':0, 'subsample': 0.8, 'colsample_bytree': 0.8, 
             'objective': 'binary:logistic'}
XGB_feat_rem1 = XGBClassifier(**params)

XGB_feat_rem1.fit(x_train[imp_cols], y_train)

y_test_pred = XGB_feat_rem1.predict(x_test[imp_cols])
y_test_pred_prob = XGB_feat_rem1.predict_proba(x_test[imp_cols])
 
XGB_feat_rem1.score(x_test[imp_cols],y_test)



# Generate model evaluation metrics for the XGBOOST- Feature Importance Threshold = 0.03
print("Performance metrics of the model for the XGBOOST- Feature Importance Threshold = 0.03")
print("-"*100)
print("Accuracy: ", metrics.accuracy_score(y_test, y_test_pred))
print("Precision Score: ",metrics.precision_score(y_test, y_test_pred))
print("Recall Score: ",metrics.recall_score(y_test, y_test_pred))
print("AUROC Score: ",metrics.roc_auc_score(y_test,  y_test_pred_prob[:,1]))
print()
print("Confusion Matrix:  \n ",metrics.confusion_matrix(y_test, y_test_pred))
print()
print("Classification Report:\n ",metrics.classification_report(y_test, y_test_pred))


#Our base model with all the features performs better than the model for which features were removed with a feature importance threshold of 0.03.
# So we stick with the  model with all the features



#Validate Model
scoring = 'neg_mean_squared_error'

kfold = KFold(n_splits=10, random_state=7)
cv_results = cross_val_score(XGB_base, x_train,y_train, cv=kfold, scoring=scoring)
print("xGB_Base CV Scores:","\n\t CV-Mean:", cv_results.mean(),
                    "\n\t CV-Std. Dev:",  cv_results.std(),"\n")


#We have good CV mean and Std deviation score for XGB_base, however, we still need to optimize the hyper-parameters.



#Optimize or Tune Model for better Performance
XGBClassifier()
param_grid = {
              'colsample_bylevel':[0.8],
              'colsample_bytree':[0.8],
              'learning_rate':[0.1, 0,2, 0.3],
              'max_depth':[2, 4, 7],
              'min_child_weight':[1, 3], 
              'n_estimators':[200],
              'n_jobs':[-1], 
              'objective':['binary:logistic'],
              'random_state':[100],
              'reg_alpha':[0.1, 1, 10], 
              'scale_pos_weight':[1], 
              'silent':[True]}

XGB_grid = GridSearchCV(XGBClassifier(), param_grid=param_grid,cv = 5, verbose=1)
XGB_grid.fit(x_train, y_train)
XGB_grid.best_params_
model = XGB_grid.best_estimator_
model.fit(x_train, y_train)
y_test_pred = model.predict(x_test)
model.score(x_test, y_test)


# Generate model evaluation metrics for the XGBOOST - Hyperparameter tuned
print("Performance metrics of the model for the XGBOOST - Hyperparameter tuned")
print("-"*100)
print("Accuracy: ", metrics.accuracy_score(y_test, y_test_pred))
print("Precision Score: ",metrics.precision_score(y_test, y_test_pred))
print("Recall Score: ",metrics.recall_score(y_test, y_test_pred))
print("AUROC Score: ",metrics.roc_auc_score(y_test,  y_test_pred_prob[:,1]))
print()
print("Confusion Matrix:  \n ",metrics.confusion_matrix(y_test, y_test_pred))
print()
print("Classification Report:\n ",metrics.classification_report(y_test, y_test_pred))


