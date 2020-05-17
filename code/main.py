# -*- coding: utf-8 -*-
"""
Created on Fri May 15 18:14:33 2020

@author: omars
"""
#############################################################################
############# Import libraries
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from utils import (wrangle_clustering, second_stage,
                   matrix_agg_predict, mean_absolute_percentage_error)
#############################################################################

#############################################################################
############# Define your dataset with all the columns you need
df = pd.read_csv(
        'C:/Users/omars/Desktop/covid19_georgia/covid19_team2/data/output/predictions_over_7_days_new.csv') # TODO: change with correct path
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
############# Adapt for a per-state model
all_models = {}
all_states = sorted(list(set(df['state'])))
for state in all_states:
    df_st = df.query('state == @state')
    # Split into training and testing
    df_train, df_test = train_test_split(df_st, test_size=0.33, shuffle=False)
    X_train, y_train, first_stage_train = df_train.loc[:, features], df_train[true] - df_train[predicted], df_train[predicted]
    X_test, y_test, first_stage_test = df_test.loc[:, features], df_test[true] - df_test[predicted], df_test[predicted]
    
    X_train, y_train, first_stage_train = np.array(X_train), np.array(y_train), np.array(first_stage_train)
    X_test, y_test, first_stage_test = np.array(X_test), np.array(y_test), np.array(first_stage_test)
    
    # Automatically get results for the second stage (combined)
    ## 'results' is the table summarizing the results
    ## 'model_dict' is the dictionary containing all the trained models.
    ## e.g. model_dict['lin'] is the trained linear model
    results, model_dict = second_stage(X_train, y_train, first_stage_train,
                                       X_test, y_test, first_stage_test,
                                       ml_models=ml_models)
    results = results.iloc[1:, :]
    all_models[state] = model_dict[results.loc[
        results['Out-of-Sample MAPE'].idxmin(), 'Model']]
    print(state + ' done')
    
############# Aggregation and results
df_train, df_test = train_test_split(df, test_size=0.33, shuffle=False)
X_train, y_train, first_stage_train = df_train.loc[:, ['state'] + features], df_train[true] - df_train[predicted], df_train[predicted]
X_test, y_test, first_stage_test = df_test.loc[:, ['state'] + features], df_test[true] - df_test[predicted], df_test[predicted]
X_train, y_train, first_stage_train = np.array(X_train), np.array(y_train), np.array(first_stage_train)
X_test, y_test, first_stage_test = np.array(X_test), np.array(y_test), np.array(first_stage_test)
predictions_train = matrix_agg_predict(X_train, all_models)
predictions_test = matrix_agg_predict(X_test, all_models)
print('First stage In-Sample R2: ' + str(r2_score(
    first_stage_train + y_train, first_stage_train)))
print('First stage Out-of-Sample R2: ' + str(r2_score(
    first_stage_test + y_test, first_stage_test)))
print('First stage In-Sample MAPE: ' + str(mean_absolute_percentage_error(
    first_stage_train + y_train, first_stage_train)))
print('First stage Out-of-Sample MAPE: ' + str(mean_absolute_percentage_error(
    first_stage_test + y_test, first_stage_test)))

print('Aggregated two-stage state model In-Sample R2: ' + str(r2_score(
    first_stage_train + y_train, first_stage_train + predictions_train)))
print('Aggregated two-stage state model Out-of-Sample R2: ' + str(r2_score(
    first_stage_test + y_test, first_stage_test + predictions_test)))
print('Aggregated two-stage state model In-Sample MAPE: ' + str(mean_absolute_percentage_error(
    first_stage_train + y_train, first_stage_train + predictions_train)))
print('Aggregated two-stage state model Out-of-Sample MAPE: ' + str(mean_absolute_percentage_error(
    first_stage_test + y_test, first_stage_test + predictions_test)))
#############################################################################