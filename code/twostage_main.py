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
from utils import (wrangle_clustering, read_measures, matrix_agg_predict,
                   print_results, plot_results, train_state_models)
from copy import deepcopy
from datetime import datetime
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
all_states = sorted(list(set(df['state'])))
all_models_cl = train_state_models(df, all_states,
                                   features, true, predicted, ml_models)

#%%
############# Aggregation and results
df = df.sort_values('date')
df_train, df_test = train_test_split(df, test_size=0.33, shuffle=False)
X_train, y_train, first_stage_train = df_train.loc[:, ['state'] + features], df_train[true] - df_train[predicted], df_train[predicted]
X_test, y_test, first_stage_test = df_test.loc[:, ['state'] + features], df_test[true] - df_test[predicted], df_test[predicted]
X_train, y_train, first_stage_train = np.array(X_train), np.array(y_train), np.array(first_stage_train)
X_test, y_test, first_stage_test = np.array(X_test), np.array(y_test), np.array(first_stage_test)
predictions_train = matrix_agg_predict(X_train, all_models_cl)
predictions_test = matrix_agg_predict(X_test, all_models_cl)

print_results(X_train, y_train, first_stage_train,
              X_test, y_test, first_stage_test,
              predictions_train, predictions_test)

output = deepcopy(df.loc[:, ['state', 'Date'] + features + [true, predicted]])
output['Clustering + ML'] =  output[predicted] + matrix_agg_predict(np.array(df.loc[:, ['state'] + features]), all_models_cl)
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
all_states = sorted(list(set(df['state'])))
all_models= train_state_models(df, all_states,
                               features, true, predicted, ml_models)

#%%
############# Aggregation and results
df = df.sort_values('date')
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
state = 'Massachusetts'
model = 'Epidemiological'
train_date = df_train['date'].iloc[-1]
plot_results(output, state, model, train_date)
#############################################################################

# #%%
# dic_results1 = {}
# dic_results2 = {}
# dic_results3 = {}
# dic_results4 = {}
# model = 'Epidemiological'
# train_date = df_train['Date'].iloc[-1]

# for state in all_states:
#     df_st = output.query('State == @state')
#     dic_results1[state] = mean_absolute_percentage_error(df_st[df_st['Date'] <= train_date]['True'], df_st[df_st['Date'] <= train_date][model])
#     dic_results2[state] = mean_absolute_percentage_error(df_st[df_st['Date'] <= train_date]['True'], df_st[df_st['Date'] <= train_date][model + ' + ML'])
#     dic_results3[state] = mean_absolute_percentage_error(df_st[df_st['Date'] > train_date]['True'], df_st[df_st['Date'] > train_date][model])
#     dic_results4[state] = mean_absolute_percentage_error(df_st[df_st['Date'] > train_date]['True'], df_st[df_st['Date'] > train_date][model + ' + ML'])


# df_st = output.query('State != "LOL"')
# dic_results1[state] = mean_absolute_percentage_error(df_st[df_st['Date'] <= train_date]['True'], df_st[df_st['Date'] <= train_date][model])
# dic_results2[state] = mean_absolute_percentage_error(df_st[df_st['Date'] <= train_date]['True'], df_st[df_st['Date'] <= train_date][model + ' + ML'])
# dic_results3[state] = mean_absolute_percentage_error(df_st[df_st['Date'] > train_date]['True'], df_st[df_st['Date'] > train_date][model])
# dic_results4[state] = mean_absolute_percentage_error(df_st[df_st['Date'] > train_date]['True'], df_st[df_st['Date'] > train_date][model + ' + ML'])


# results = {'Model': list(dic_results1.keys()),
#       #model + ' In-Sample MAPE': [100*dic_results1[i] for i in list(dic_results1.keys())],
#       #model + ' + ML In-Sample MAPE': [100*dic_results2[i] for i in list(dic_results1.keys())],
#       model + ' Out-of-Sample MAPE': [100*dic_results3[i] for i in list(dic_results1.keys())],
#       model + ' + ML Out-of-Sample MAPE': [100*dic_results4[i] for i in list(dic_results1.keys())]}

# pd.options.display.float_format = "{:.2f}".format
# dfp = pd.DataFrame(results)
# dfp['Out-of-Sample Improvement'] = dfp[model + ' Out-of-Sample MAPE'] - dfp[model + ' + ML Out-of-Sample MAPE']
# print(dfp)


# globalre = pd.DataFrame([['In-Sample R2', 100*0.986, 100*0.996],
# ['Out-of-Sample R2', 100*0.995, 100*0.998],
# ['In-Sample MAPE', 100*0.133, 100*0.111],
# ['Out-of-Sample MAPE', 100*0.047, 100*0.044]], columns= ['Metric', '1-stage', '2-stage'])


# globalre = pd.DataFrame([['In-Sample R2', 100*0.925, 100*0.928],
# ['Out-of-Sample R2', 100*0.186, 100*0.189],
# ['In-Sample MAPE', 100*0.346, 100* 0.265],
# ['Out-of-Sample MAPE', 100*0.919, 100*0.932]], columns= ['Metric', '1-stage', '2-stage'])
#############################################################################