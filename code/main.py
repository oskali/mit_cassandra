# -*- coding: utf-8 -*-
"""
Created on Fri May 15 18:14:33 2020

@author: omars
"""
#############################################################################
############# Import libraries
from sklearn.model_selection import train_test_split
import pandas as pd
from utils import wrangle_clustering, second_stage
#############################################################################

#############################################################################
############# Define your dataset with all the columns you need
df = pd.read_csv(
        'C:/Users/omars/Desktop/covid19_georgia/data/output/predictions_over_7_days_new.csv') # TODO: change with correct path
df = wrangle_clustering(
        df, cols_to_keep=None).dropna() # don't run this command unless it's the clustering results dataset
#############################################################################

#############################################################################
############# Define your column names
## Column name of predicted value of first stage model
predicted = 'predicted_value_model' + str(6)
## Column name of true value
true = 'cases' 
## Column names for the features you want to use in the second stage model
features = list(df.columns[4:26])

############# Which models do you want to test for the second stage?
ml_models = ['lin', # linear model
             'elastic', # elastic net
             'cart',  # cart tree
             'rf', # random forest
             'xgb', # xgboost
             #'xst', # xstrees
             'linear_svm', # linear svm
             'kernel_svm'] # kernel svm
#############################################################################

#############################################################################
############# Split into training and testing
df_train, df_test = train_test_split(df, test_size=0.33, shuffle=False)
X_train, y_train, first_stage_train = df_train.loc[:, features], df_train[true] - df_train[predicted], df_train[predicted]
X_test, y_test, first_stage_test = df_test.loc[:, features], df_test[true] - df_test[predicted], df_test[predicted]
#############################################################################

#############################################################################
############# Automatically get results for the second stage (combined)
## 'results' is the table summarizing the results
## 'model_dict' is the dictionary containing all the trained models. e.g. model_dict['lin'] is the trained linear model
results, model_dict = second_stage(X_train, y_train, first_stage_train,
                                   X_test, y_test, first_stage_test,
                                   ml_models=ml_models)
#############################################################################