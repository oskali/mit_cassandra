# -*- coding: utf-8 -*-
"""
Created on Fri May 15 18:14:33 2020

@author: omars
"""
#%%
#############################################################################
############# Import libraries
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from utils import (wrangle_clustering, second_stage, read_measures,
                   matrix_agg_predict, mean_absolute_percentage_error,
                   print_results)
from copy import deepcopy
from datetime import datetime
import matplotlib.pyplot as plt
#############################################################################

#%%
#############################################################################
############# Clustering second stage
df = pd.read_csv(
        'C:/Users/omars/Desktop/covid19_georgia/covid19_team2/data/output/predictions_over_7_days_20200508.csv') # TODO: change with correct path
df = wrangle_clustering(
        df, cols_to_keep=None).dropna()

measures, measures_names = read_measures('C:/Users/omars/Desktop/covid19_georgia/large_data/input/05-14_states_cases_measures_mobility.csv')
df = df.merge(measures, how='left', left_on=['state', 'Date'], right_on=['state', 'date'])
#############################################################################

#############################################################################
############# Define your column names
## Column name of predicted value of first stage model
predicted = 'predicted_value_model' + str(6)
## Column name of true value
true = 'cases' 
df = df[df[true] > 100]
## Column names for the features you want to use in the second stage model
features= measures_names

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

#%%
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

#%%
############# Aggregation and results
df_train, df_test = train_test_split(df, test_size=0.33, shuffle=False)
X_train, y_train, first_stage_train = df_train.loc[:, ['state'] + features], df_train[true] - df_train[predicted], df_train[predicted]
X_test, y_test, first_stage_test = df_test.loc[:, ['state'] + features], df_test[true] - df_test[predicted], df_test[predicted]
X_train, y_train, first_stage_train = np.array(X_train), np.array(y_train), np.array(first_stage_train)
X_test, y_test, first_stage_test = np.array(X_test), np.array(y_test), np.array(first_stage_test)
predictions_train = matrix_agg_predict(X_train, all_models)
predictions_test = matrix_agg_predict(X_test, all_models)

print_results(X_train, y_train, first_stage_train,
              X_test, y_test, first_stage_test,
              predictions_train, predictions_test)

output = deepcopy(df.loc[:, ['state', 'Date'] + features + [true, predicted]])
output['Clustering + ML'] =  output[predicted] + matrix_agg_predict(np.array(df.loc[:, ['state'] + features]), all_models)
output = output.rename(columns={"state": "State", true: "True", predicted: "Clustering"})
#############################################################################










#%%
#############################################################################
############# Epidemiological second stage
df = pd.read_csv(
        'C:/Users/omars/Desktop/covid19_georgia/large_data/output/USA_Cases_Predictions_May_Data_Akarsh.csv') # TODO: change with correct path

df['date'] = df['date'].apply(lambda x: datetime.strptime(x, '%d-%m-%Y'))
df = df.merge(measures, how='left', left_on=['state', 'date'], right_on=['state', 'date'])
#############################################################################

#############################################################################
############# Define your column names
## Column name of predicted value of first stage model
predicted = 'predictions'
## Column name of true value
true = 'cases' 
df = df[df[true] > 100]
df = df[df['date'] >= output.Date.min()]
df = df[df['date'] <= output.Date.max()]
## Column names for the features you want to use in the second stage model
features= measures_names

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

#%%
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

#%%
############# Aggregation and results
df_train, df_test = train_test_split(df, test_size=0.33, shuffle=False)
X_train, y_train, first_stage_train = df_train.loc[:, ['state'] + features], df_train[true] - df_train[predicted], df_train[predicted]
X_test, y_test, first_stage_test = df_test.loc[:, ['state'] + features], df_test[true] - df_test[predicted], df_test[predicted]
X_train, y_train, first_stage_train = np.array(X_train), np.array(y_train), np.array(first_stage_train)
X_test, y_test, first_stage_test = np.array(X_test), np.array(y_test), np.array(first_stage_test)
predictions_train = matrix_agg_predict(X_train, all_models)
predictions_test = matrix_agg_predict(X_test, all_models)

print_results(X_train, y_train, first_stage_train,
              X_test, y_test, first_stage_test,
              predictions_train, predictions_test)

df['Epidemiological + ML'] = df[predicted] + matrix_agg_predict(np.array(df.loc[:, ['state'] + features]), all_models)
df = df.loc[:, ['state', 'date', predicted, 'Epidemiological + ML']]
df = df.rename(columns={predicted: 'Epidemiological'})
output = output.merge(df, how='left', left_on=['State', 'Date'], right_on=['state', 'date'])
output = output.dropna().drop(columns=['state', 'date'])
output.to_csv('C:/Users/omars/Desktop/covid19_georgia/large_data/output/2stage_results.csv')
#############################################################################

#%%
#############################################################################
state = 'Texas'
model = 'Clustering'
train_date = df_train['date'].iloc[-1]
df_st = output.query('State == @state')
plt.plot(df_st['Date'], df_st['True'],  label= 'True value')
plt.plot(df_st['Date'], df_st[model],  label= '1-stage ' + model + ' model')
plt.plot(df_st['Date'], df_st[model + ' + ML'], label= '2-stage ' + model + ' model')
plt.axvline(x=train_date, linestyle='--', color='red')
plt.xticks(rotation=45)
plt.legend()
print('1-stage ' + model + ' model MAPE: ' + str(mean_absolute_percentage_error(df_st['True'], df_st[model])))
print('2-stage ' + model + ' MAPE: ' + str(mean_absolute_percentage_error(df_st['True'], df_st[model + ' + ML'])))
#############################################################################
