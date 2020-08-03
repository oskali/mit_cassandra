# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 21:21:18 2020

@author: omars
"""
# %% Libraries

from datetime import datetime
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR, LinearSVR
import os

# %% User and path

USER = 'david'

if USER == 'omar':
    df_path = 'C:\\Users\\omars\\Desktop\\covid19_georgia\\covid19_team2\data\\input\\07_08_2020_states_combined.csv'

if USER == 'david':
    df_path = "C:\\Users\\david\\Dropbox (MIT)\\COVID-19-Team2\Data\\07_22_2020_counties_combined_w_pct.csv"

elif USER == 'lpgt':
    df_path = r'../data/input/06_15_2020_MA_only.csv'

# %% Target and column names

target_col = 'cases'
date_col = 'date'
region_col = 'fips'
population = 'population'

# %% Run Parameters

random_state = 42
retrain = False

training_agg_cutoff = '2020-06-01'
training_cutoff = '2020-06-15'
validation_cutoff = '2020-07-15'

regions_dict = {
    "fips": [25017, 34023],
    "state": ['New York', 'Massachusetts'],
}
regions = regions_dict[region_col]  # regions to predict  #
unformatted_dates = ['2020-07-01', '2020-08-15']  # dates to predict  #

restriction_dict = {
    "fips":
        {
            "state": ["Massachusetts", "New Jersey", "Connecticut", "New Hampshire",
                      # "Alabama",
                      "Florida", "California"],
            "county": ["Queens", "New York", "Bronx"]
        },
    "state": None
}
n_samples = 3

# %% Load and Save Parameters

train_knn = False
train_mdp = True
train_sir = False
train_knn_agg = False
train_mdp_agg = True
train_sir_agg = False
train_agg = True
train_ci = True
load_knn = True
load_mdp = True
load_sir = True
load_agg = True
load_ci = True
sir_file = 'models\\sir_{}_{}_{}.pickle'.format(training_cutoff.replace("-", ""), target_col, region_col)
knn_file = 'models\\knn_{}_{}_{}.pickle'.format(training_cutoff.replace("-", ""), target_col, region_col)
mdp_file = 'models\\mdp_{}_{}_{}.pickle'.format(training_cutoff.replace("-", ""), target_col, region_col)
agg_file = 'models\\agg_{}_{}_{}.pickle'.format(training_cutoff.replace("-", ""), target_col, region_col)
ci_file = 'models\\ci_{}_{}_{}.pickle'.format(training_cutoff.replace("-", ""), target_col, region_col)
export_file = 'export_{}_{}.csv'.format(training_cutoff.replace("-", ""), target_col, region_col)

# %% Parameters SIR

nmin = {"fips": 100, "state": 100}
optimizer = 'Nelder-Mead'

sir_params_dict = \
    {
        "nmin": nmin[region_col],
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
        #'initial_param' :[0.4, 0.06],
        #"nmin_train_set": 10
    }

# %% Parameters KNN

knn_params_dict = \
    {
        "deterministic": True,
        'date': date_col,
        'region': region_col,
        'target': target_col,
    }

# %% Parameters MDP

region_exceptions = {
    "state":
        ['Guam', 'Northern Mariana Islands',
         'Puerto Rico', 'Diamond Princess',
         'Grand Princess', 'American Samoa', 'Virgin Islands'],
    "fips":
        [1001, 1003, 1005, 1007, 1009, 1011, 1013, 1015, 1017, 1051, 1053, 1067,
         1039.0, 1041.0, 1063.0, 1071.0, 1087.0, 1099.0, 1133.0, 1021, 1023, 1025,
         1031, 1033, 1043, 1045, 1035.0, 1129.0,
         6101.0,
         1047, 1049.0, 1079.0, 1105.0,
         2043.0,
         12005.0, 12087.0, 12075.0, 12043.0, 12089.0]}

mdp_features_dict = \
    {
        'state':
            {"deaths": ["cases_pct3", "cases_pct5"],
             "cases": ["cases_pct3", "cases_pct5"]},
        'fips':
            {"deaths": [],
             "cases": []}
    }

mdp_params_dict = \
    {
        "days_avg": 3,
        "horizon": 8,
        "n_iter": 100,
        "n_folds_cv": 5,
        "clustering_distance_threshold": 0.1,
        "splitting_threshold": 0.,
        "classification_algorithm": 'DecisionTreeClassifier',
        "clustering_algorithm": 'Agglomerative',
        "n_clusters": None,
        "action_thresh": ([], 0),  # ([-250, 200], 1),
        "features_list": mdp_features_dict[region_col][target_col],
        "verbose": 1,
        "n_jobs": 1,
        "date_colname": date_col,
        "target_colname": target_col,
        "region_colname": region_col,
        "random_state": random_state,
        "keep_first": False,
        "save": False,
        "savepath": "",  # os.path.dirname(mdp_file),
        "region_exceptions": region_exceptions[region_col]
    }
# %% Parameters AGG

per_region = True
ml_methods = [  # 'lin',
    'elastic',
    # 'cart',
    # 'rf',
    # 'xgb',
    # 'linear_svm',
    # 'kernel_svm'
]
ml_hyperparams = {'xgb': {'gamma': [1, 5], 'max_depth': [3, 5]},
                  'cart': {'max_depth': [3, 5, 10, None]},
                  'rf': {'max_depth': [3, 5, 10, None], 'min_samples_leaf': [1, 2, 5]},
                  }

ml_mapping = {'lin': [LinearRegression(), False],
              'elastic': [ElasticNetCV(fit_intercept=False), False],
              'xgb': [XGBRegressor(learning_rate=0.05, n_estimators=100, silent=True), True],
              'cart': [DecisionTreeRegressor(), True],
              'rf': [RandomForestRegressor(), True],
              'linear_svm': [LinearSVR(), False],
              'kernel_svm': [SVR(gamma='auto'), False]}

# %% Parameters CI

ci_range = 0.95

# %% Date Formatting
dates = [datetime.strptime(d, '%Y-%m-%d') for d in unformatted_dates]
