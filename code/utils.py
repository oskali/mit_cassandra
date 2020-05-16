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
#############################################################################

