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
from pandas import date_range
import os

# %% User and path

USER = 'david'

if USER == 'omar':
    df_path = 'C:\\Users\\omars\\Desktop\\covid19_georgia\\covid19_team2\data\\input\\07_08_2020_states_combined.csv'
    default_path = os.getcwd()

if USER == 'david':
    # df_path = "C:\\Users\\david\\Dropbox (MIT)\\COVID-19-Team2\Data\\08_13_2020_counties_combined_seird.csv"
    default_path = "C:\\Users\\david\\Dropbox (MIT)\\COVID-19-Team2\Data\\"
    # df_path = r'C:\Users\david\Dropbox (MIT)\COVID-19-Team2\Data\08_14_2020_states_and_countries_combined_restricted_.csv'
    # df_path = r'C:\Users\david\Dropbox (MIT)\COVID-19-Team2\Data\08_14_2020_states_and_countries_combined_restricted_new.csv'
    # df_path = r'C:\Users\david\Dropbox (MIT)\COVID-19-Team2\Data\08_14_2020_states_and_countries_combined_restricted_morocco.csv'
    # df_path = r'C:\Users\david\Dropbox (MIT)\COVID-19-Team2\Data\08_13_2020_counties_countries_combined_seird.csv'
    # df_path = r'C:\Users\david\Dropbox (MIT)\COVID-19-Team2\Data\09_02_2020_states_countries_combined_with_Gglmobility.csv'
    df_path = r'C:\Users\david\Dropbox (MIT)\COVID-19-Team2\Data\08_15_2020_states_countries_combined_with_GooPCADavid.csv'

elif USER == 'lpgt':
    #df_path = r'../../../../../../Dropbox (MIT)/COVID-19-Team2/Data/08_11_2020_states_combined.csv'
    default_path =  os.getcwd()

    # Please uncomment and adjust the appropriate path

    # [Long-term prediction state]
    #df_path = r'../../../../../../Dropbox (MIT)/COVID-19-Team2/Data/09_02_2020_states_combined.csv'

    # [Long-term prediction county]
    df_path = r'../../../../../../Dropbox (MIT)/COVID-19-Team2/Data/08_13_2020_counties_combined_seird.csv'

# %% Target and column names

target_col = 'cases'
date_col = 'date'
region_col = 'state'
population = 'population'
tests_col = 'people_tested'

# %% Run Parameters

random_state = 42
retrain = False

training_agg_cutoff = '2020-07-01'
training_cutoff = '2020-08-01'
validation_cutoff = '2020-08-15'

regions_dict = {
    "fips": [
        6037, 6085, 6073, 6059, 6075, 6001, 6081, 6065, 6067, 6013, 6071, 6029,
        6111, 6077, 6083, 6107, 6019, 6099, 6041, 6025, 6097, 6095, 6053, 6079,
        6031, 6047, 6087, 6113, 6061, 6039, 6055, 6035, 6069, 6007, 6017, 6101,
        6089, 6115, 6021, 6057, 6011, 6045, 6023, 6103, 6033, 25017, 25025,
        25021, 25009, 25027, 25023, 25005, 25013, 25001, 25003, 25015, 25011],
    "state": ['Massachusetts', 'California', "Morocco"],
    # "state": ["Morocco"],
}
regions = regions_dict[region_col]  # regions to predict  #
unformatted_dates = [datetime.strftime(_, "%Y-%m-%d") for _ in date_range('2020-09-02', '2020-12-15', freq="1D")]  # dates to predict  #

restriction_dict = {
    "fips":
        {
            "state": [
                "Massachusetts",
                "New Jersey",
                "Connecticut",
                "New Hampshire",
                # "Alabama",
                # "Florida",
                "California",
                "Countries"
                      ],
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
train_mdp_agg = False
train_sir_agg = False
train_agg = False
train_ci = False
train_preval = False
load_mdp = True
load_knn = False
load_sir = False
load_agg = False
load_ci = False
load_preval = False
sir_file = os.path.join('models', 'sir_{}_{}_{}_country_action.pickle'.format(training_cutoff.replace("-", ""), target_col, region_col))
knn_file = os.path.join('models', 'knn_{}_{}_{}_country_action.pickle'.format(training_cutoff.replace("-", ""), target_col, region_col))
mdp_file = os.path.join('models', 'mdp_{}_{}_{}_country_action.pickle'.format(training_cutoff.replace("-", ""), target_col, region_col))
agg_file = os.path.join('models', 'agg_{}_{}_{}_country_action.pickle'.format(training_cutoff.replace("-", ""), target_col, region_col))
ci_file = os.path.join('models', 'ci_{}_{}_{}_action.pickle'.format(training_cutoff.replace("-", ""), target_col, region_col))
preval_file = os.path.join('models', 'preval_{}_{}_{}_action.pickle'.format(training_cutoff.replace("-", ""), target_col, region_col))
export_file = 'export_{}_{}_action.csv'.format(training_cutoff.replace("-", ""), target_col, region_col)

# %% Parameters SIR

nmin = {"fips": 100, "state": 200}
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
        # 'initial_param' :[0.4, 0.06],
        # "nmin_train_set": 10
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

region_exceptions_dict = {
    "state":
        ['Guam', 'Northern Mariana Islands',
         'Puerto Rico', 'Diamond Princess',
         'Grand Princess', 'American Samoa',
         'Virgin Islands', 'Hawaii', "Benin", "Ecuador",
         "Jordan", "Lithuania", "Uganda",
         "Georgia", "International", "Mongolia"
         ],
    "fips":
        []}

mdp_features_dict = \
    {
        'state':
            {"deaths": ["workplaces", "cases_pct3", "cases_pct5"],
             "cases": ["mobility_pca", "cases_pct3", "cases_pct5"]},  # ["cases_nom", "cases_pct3", "cases_pct5"]},
        'fips':
            {"deaths": [],
             "cases": []}
    }

mdp_params_dict = \
    {
        "days_avg": 7,
        "horizon": 4,
        "n_iter": 180,
        "n_folds_cv": 4,
        "clustering_distance_threshold": 0.1,
        "splitting_threshold": 0.,
        "classification_algorithm": 'DecisionTreeClassifier',
        "clustering_algorithm": 'Agglomerative',
        "n_clusters": None,
        "action_thresh": ([-83, 64.], 1),  # ([-25, 25.], 1), # ([], 0),  # ([-250, 200], 1),
        "features_list": mdp_features_dict[region_col][target_col],
        "verbose": 1,
        "n_jobs": -1,
        "date_colname": date_col,
        "target_colname": target_col,
        "region_colname": region_col,
        "random_state": random_state,
        "keep_first": True,
        "save": False,
        "plot": False,
        "savepath": "",  # os.path.dirname(mdp_file),
        "region_exceptions": None
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

ci_range = 0.75

# %% Parameters Prevalence

alpha = 0.11

# %% Date Formatting
dates = [datetime.strptime(d, '%Y-%m-%d') for d in unformatted_dates]
