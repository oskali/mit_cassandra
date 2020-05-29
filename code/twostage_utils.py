# -*- coding: utf-8 -*-
"""
Created on Fri May 15 18:14:33 2020

@author: omars
"""
#############################################################################
############# Import libraries
import pandas as pd
from copy import deepcopy
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV    
from sklearn.metrics import r2_score
from sklearn.svm import SVR, LinearSVR
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#############################################################################

#############################################################################
############# Wrangle clustering results
def wrangle_clustering_results_day_i(clt,
                                     i,
                                     var='cases'):
    
    target = 'pred_growth_for_next_' + str(i) + 'days' 

    output = deepcopy(clt)
    output['pred_value_for_next_' + str(i) + 'days'] = (1+output[target])* output[var]
    output['predicted_value_model' +str(i)] = output.groupby('state')[
            'pred_value_for_next_' + str(i) + 'days'].shift(i)
    del output[target]
    return(output)
    
def wrangle_clustering(clt,
                       n=None,
                       cols_to_keep=['state', 'Date', 'cases', 'deaths'],
                       var='cases'):
    
    if n is None:
        n = max([int(x[len('pred_growth_for_next_'): x.find('days')]) if x.find('pred_growth_for_next_') >= 0 else 0 for x in clt.columns])
    output = deepcopy(clt)
    ltarget = []
    for i in range(1, n+1):
        output = wrangle_clustering_results_day_i(output,
                                                  i,
                                                  var='cases')
        ltarget.append('predicted_value_model' +str(i))
    
    if cols_to_keep is not None:
        output = output.loc[:, cols_to_keep + ltarget]
    
    output['Date'] = output['Date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    return(output)
#############################################################################
    
#############################################################################
############# Helper functions
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))

def update_results(X_train, y_train, first_stage_train,  # y should be equal to true value - first-stage value
                   X_test, y_test, first_stage_test,
                   model,
                   model_name,
                   dic_results,
                   train_dic_results,
                   dic_results_mape,
                   train_dic_results_mape):
    
        dic_results[model_name]= r2_score(first_stage_test + y_test, first_stage_test + model.predict(X_test))
        train_dic_results[model_name] = r2_score(first_stage_train + y_train, first_stage_train + model.predict(X_train))
        dic_results_mape[model_name]= mean_absolute_percentage_error(first_stage_test + y_test, first_stage_test + model.predict(X_test))
        train_dic_results_mape[model_name] = mean_absolute_percentage_error(first_stage_train + y_train, first_stage_train + model.predict(X_train))
        
        return dic_results, train_dic_results, dic_results_mape, train_dic_results_mape

def read_measures(path):
    measures = pd.read_csv(path)
    measures_names =  ['EmergDec', 'SchoolClose', 'GathRestrict25', 'GathRestrictAny',
                   'OtherBusinessClose', 'RestaurantRestrict', 'GathRestrict10',
                   'CaseIsolation', 'StayAtHome', 'PublicMask', 'Quarantine',
                   'NEBusinessClose', 'TravelRestrictIntra', 'GathRestrict50',
                   'BusinessHealthSafety', 'GathRestrict250', 'GathRecomAny',
                   'GathRestrict1000', 'TravelRestrictExit', 'TravelRestrictEntry',
                   'GathRestrict100', 'GathRestrict5', 'GathRestrict500']

    measures = measures.loc[:, ['state', 'date'] + measures_names]
    measures['date'] = measures['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    measures = measures.groupby(['state', 'date']).last().reset_index()
    return(measures, measures_names)

def print_results(X_train, y_train, first_stage_train, X_test, y_test, first_stage_test,
                  predictions_train, predictions_test):
    
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

def plot_results(output, state, model, train_date):
    if state is not None:
        df_st = output.query('State == @state')
    else:
        df_st = output.groupby('Date')[['True', model, model + ' + ML']].sum().reset_index()
    plt.plot(df_st['Date'], df_st['True'],  label= 'True value')
    plt.plot(df_st['Date'], df_st[model],  label= '1-stage ' + model + ' model')
    plt.plot(df_st['Date'], df_st[model + ' + ML'], label= '2-stage ' + model + ' model')
    plt.axvline(x=train_date, linestyle='--', color='red')
    plt.xticks(rotation=45)
    plt.legend()
    print('1-stage ' + model + ' model In-Sample MAPE: ' + str(mean_absolute_percentage_error(df_st[df_st['Date'] <= train_date]['True'], df_st[df_st['Date'] <= train_date][model])))
    print('2-stage ' + model + ' model In-Sample MAPE: ' + str(mean_absolute_percentage_error(df_st[df_st['Date'] <= train_date]['True'], df_st[df_st['Date'] <= train_date][model + ' + ML'])))
    print('1-stage ' + model + ' model Out-of-Sample MAPE: ' + str(mean_absolute_percentage_error(df_st[df_st['Date'] > train_date]['True'], df_st[df_st['Date'] > train_date][model])))
    print('2-stage ' + model + ' model Out-of-Sample MAPE: ' + str(mean_absolute_percentage_error(df_st[df_st['Date'] > train_date]['True'], df_st[df_st['Date'] > train_date][model + ' + ML'])))

#############################################################################
        
#############################################################################    
############# Second-stage wrapper implementation
def second_stage(X_train, y_train, first_stage_train,  # y should be equal to true value - first-stage value
                 X_test, y_test, first_stage_test,
                 ml_models=['lin', 'elastic', 'cart', 'rf', 'xgb', 'xst', 'linear_svm', 'kernel_svm'],
                 params=None):
    
    if params is None:
        params={}
        params['xgb'] = {'gamma': [1, 5], 'max_depth': [3, 5]}
        params['cart'] = {'max_depth': [3, 5, 10, None]}
        params['rf'] = {'max_depth': [3, 5, 10, None], 'min_samples_leaf': [1, 2, 5]}
        params['xst'] = {'max_depth': [3], 'min_samples_leaf': [2], 'nearest_leaves_k': [5]}
    
    dic_results = {}
    train_dic_results = {}
    dic_results_mape = {}
    train_dic_results_mape = {}
    model_dict = {}

    ml_mapping = {}
    ml_mapping['lin'] = [LinearRegression(), False]
    ml_mapping['elastic'] = [ElasticNetCV(), False]
    ml_mapping['xgb'] = [XGBRegressor(learning_rate=0.05, n_estimators=100, silent=True), True]
    ml_mapping['cart'] = [DecisionTreeRegressor(), True]
    ml_mapping['rf'] = [RandomForestRegressor(), True]
    ml_mapping['linear_svm'] = [LinearSVR(), False]
    ml_mapping['kernel_svm'] = [SVR(gamma='auto'), False]
    
    if 'xst' in ml_models:
        from XSTrees_v2 import XSTreesRegressor
        ml_mapping['xst'] = [XSTreesRegressor(n_estimators=100, n_sampled_trees=100), True]
    
    dic_results['first_stage']= r2_score(first_stage_test + y_test, first_stage_test)
    train_dic_results['first_stage'] = r2_score(first_stage_train + y_train, first_stage_train)
    dic_results_mape['first_stage']= mean_absolute_percentage_error(first_stage_test + y_test, first_stage_test)
    train_dic_results_mape['first_stage'] = mean_absolute_percentage_error(first_stage_train + y_train, first_stage_train)
        
    for model_name in ml_mapping.keys():
        if model_name in ml_models:
            if ml_mapping[model_name][1]:
                model = GridSearchCV(ml_mapping[model_name][0], params[model_name])
            else:
                model = ml_mapping[model_name][0]
            
            model.fit(X_train, y_train)
            model_dict[model_name] = model
            dic_results, train_dic_results, dic_results_mape, train_dic_results_mape = update_results(X_train, y_train, first_stage_train,  # y should be equal to true value - first-stage value
                   X_test, y_test, first_stage_test,
                   model,
                   model_name,
                   dic_results,
                   train_dic_results,
                   dic_results_mape,
                   train_dic_results_mape)
    
    output = {'Model': list(dic_results.keys()),
          'In-Sample R2': [100*train_dic_results[i] for i in list(dic_results.keys())],
          'Out-of-Sample R2': [100*dic_results[i] for i in list(dic_results.keys())],
          'In-Sample MAPE': [100*train_dic_results_mape[i] for i in list(dic_results.keys())],
          'Out-of-Sample MAPE': [100*dic_results_mape[i] for i in list(dic_results.keys())]}

    pd.options.display.float_format = "{:.2f}".format
    dfp = pd.DataFrame(output)
    return(dfp, model_dict)

############# Second-stage training
def train_state_models(df, all_states, features, true, predicted, ml_models):
    all_models = {}
    for state in all_states:
        df_st = df.query('state == @state').sort_values('date')
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
    return(all_models)
#############################################################################

#############################################################################
############# Aggregation helper functions for the per-state models
def agg_predict(X_agg, all_models):
    X_st = X_agg[0]
    X_rt = np.array(X_agg[1:])
    return all_models[X_st].predict(X_rt.reshape(1, -1))[0]

def matrix_agg_predict(X_total, all_models):
    n, p = X_total.shape
    predictions = [agg_predict(X_total[i, :], all_models) for i in range(n)]
    return predictions
#############################################################################