# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 21:21:18 2020

@author: omars
"""
#%% Libraries

from datetime import datetime

#%% User and path

USER = 'lpgt'

if USER == 'omar':
    df_path = r'C:\Users\david\Desktop\MIT\Courses\Research internship\covid19_team2\data\input\06_15_2020_states_combined.csv'
elif USER == 'lpgt':
    df_path = r'../data/input/06_15_2020_MA_only.csv'

#%% Target and column names

target_col = 'cases'
date_col = 'date'
region_col = 'state'
population = 'population'
nmin = 100

#%% Run Parameters

random_state = 42
training_cutoff = '2020-05-25'
regions = ['Massachusetts'] # regions to predict
unformatted_dates = ['2020-07-01', '2020-08-15'] # dates to predict

#%% Load and Save Parameters

train_knn = False
train_mdp = False
train_sir = True
load_knn = False
load_mdp = False
load_sir = True
sir_file = 'sir.pickle'
knn_file = 'knn.pickle'
mdp_file = 'mdp.pickle'

#%% Parameters SIR

sir_params_dict = \
    {
        "nmin": 100,
        "optimizer": 'Nelder-Mead',
        "betavals": [0.10, 0.15, 0.9, 0.95, 1.1, 1.2],
        "gammavals": [0.01, 0.03, 0.25, 0.27, 0.29],
        "avals": [0.333, 0.142, 0.0909, 0.0714, 0.0526],
        "muvals": [0.001, 0.003, 0.005, 0.007],
        "train_valid_split": 0.8,
        "nmin_train_set": 10
    }
#%% Parameters KNN

knn_params_dict = \
    {
        "deterministic": True,
}


#%% Parameters MDP

mdp_params_dict = \
    {
        "days_avg": 3,
        "horizon": 5,
        "n_iter": 100,
        "n_folds_cv": 5,
        "clustering_distance_threshold": 0.05,
        "splitting_threshold": 0.,
        "classification_algorithm": 'DecisionTreeClassifier',
        "clustering_algorithm": 'Agglomerative',
        "n_clusters": None,
        "action_thresh": [],
        "features_list": ["home_time", "part_time"],
        "verbose": True,
        "n_jobs": 3,
        "date_colname": date_col,
        "target_colname": target_col,
        "region_colname": region_col,
        "random_state": random_state,
    }

#%% Date Formatting
dates = [datetime.strptime(d, '%Y-%m-%d') for d in unformatted_dates]
