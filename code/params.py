# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 21:21:18 2020

@author: omars
"""
#%% Libraries

from datetime import datetime
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR, LinearSVR

#%% User and path

USER = 'omar'

if USER == 'omar':
    df_path = 'C:\\Users\\omars\\Desktop\\covid19_georgia\\covid19_team2\data\\input\\07_08_2020_states_combined.csv'

if USER == 'david':
    import os
    df_path = os.path.join(os.path.dirname(os.getcwd()), "data", "input", "07_08_2020_states_combined.csv")


elif USER == 'lpgt':
    df_path = r'../data/input/06_15_2020_MA_only.csv'

#%% Target and column names

target_col = 'deaths'
date_col='date'
region_col='state'
population='population'

#%% Run Parameters

random_state=42
retrain=True
training_cutoff = '2020-05-01'
validation_cutoff = '2020-05-15'
regions = ['New York', 'Massachusetts'] # regions to predict
unformatted_dates = ['2020-07-01', '2020-08-15'] # dates to predict
n_samples=3

#%% Load and Save Parameters

train_knn = True
train_mdp = True
train_sir = True
train_agg = True
train_ci = True
load_knn = True
load_mdp = True
load_sir = True
load_agg = True
load_ci = True
sir_file = 'sir.pickle'
knn_file = 'knn.pickle'
mdp_file = 'mdp.pickle'
agg_file = 'agg.pickle'
ci_file = 'ci.pickle'
export_file = 'export.csv'

#%% Parameters SIR

nmin=100
optimizer='Nelder-Mead'

sir_params_dict = \
    {
        "nmin": nmin,
        'date': date_col,
		'region': region_col,
		'target': target_col,
		'population': population,
        "optimizer": optimizer,
        # "betavals": [0.10, 0.15, 0.9, 0.95, 1.1, 1.2],
        # "gammavals": [0.01, 0.03, 0.25, 0.27, 0.29],
        # "avals": [0.333, 0.142, 0.0909, 0.0714, 0.0526],
        # "muvals": [0.001, 0.003, 0.005, 0.007],
        # "train_valid_split": 0.8,
        'initial_param' :[0.4, 0.06],
        "nmin_train_set": 10
    }

#%% Parameters KNN

knn_params_dict = \
    {
        "deterministic": True,
        'date':date_col,
		'region':region_col,
		'target':target_col,
}

#%% Parameters MDP
days_avg=3
horizon=5
n_iter=30
n_folds_cv=5
clustering_distance_threshold=0.1
splitting_threshold=0.
classification_algorithm='DecisionTreeClassifier'
clustering_algorithm='Agglomerative'
n_clusters=None
action_thresh=[]
features_list=[]
verbose=True
features = []
n_jobs = 1

mdp_params_dict = \
    {
        "days_avg": days_avg,
        "horizon": horizon,
        "n_iter": n_iter,
        "n_folds_cv": n_folds_cv,
        "clustering_distance_threshold": clustering_distance_threshold,
        "splitting_threshold": splitting_threshold,
        "classification_algorithm": classification_algorithm,
        "clustering_algorithm": clustering_algorithm,
        "n_clusters": n_clusters,
        "action_thresh": action_thresh,
        "features_list": features_list,
        "verbose": verbose,
        "n_jobs": n_jobs,
        "date_colname": date_col,
        "target_colname": target_col,
        "region_colname": region_col,
        "random_state": random_state,
    }

#%% Parameters AGG

per_region = True
ml_methods = [#'lin',
              'elastic',
              #'cart',
              #'rf',
              #'xgb',
              #'linear_svm',
              #'kernel_svm'
              ]
ml_hyperparams={'xgb': {'gamma': [1, 5], 'max_depth': [3, 5]},
                'cart': {'max_depth': [3, 5, 10, None]},
                'rf': {'max_depth': [3, 5, 10, None], 'min_samples_leaf': [1, 2, 5]},
                }

ml_mapping = {'lin' : [LinearRegression(), False],
              'elastic' : [ElasticNetCV(fit_intercept=False), False],
              'xgb' : [XGBRegressor(learning_rate=0.05, n_estimators=100, silent=True), True],
              'cart' : [DecisionTreeRegressor(), True],
              'rf' : [RandomForestRegressor(), True],
              'linear_svm' : [LinearSVR(), False],
              'kernel_svm' : [SVR(gamma='auto'), False]}

#%% Parameters CI

ci_range = 0.95

#%% Date Formatting
dates = [datetime.strptime(d, '%Y-%m-%d') for d in unformatted_dates]
