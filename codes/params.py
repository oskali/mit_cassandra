# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 21:21:18 2020

@author: omars
"""
# %% Libraries

from datetime import datetime, timedelta
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from pandas import date_range
import os


# %% Mutable variables

default_path = r"../data/01_05_2021_states_combined_temperature_demographics_and_holidays_v3_lags_pred_sep.csv"
df_path = r"../data/01_05_2021_states_combined_temperature_demographics_and_holidays_v3_lags_pred_sep.csv"

# default_path = r"../data/01_05_2021_states_combined_temperature_demographics_and_holidays_v3_lags_pred_sep.csv"
# df_path = r"../data/01_05_2021_states_combined_temperature_demographics_and_holidays_v3_lags_pred_sep.csv"
target_col = 'cases'
region_col = 'state'

# date0
# training_agg_cutoff = '2020-05-16'
# training_cutoff = '2020-06-01'
# validation_cutoff = '2020-06-27'

 # date1
# training_agg_cutoff = '2020-10-07'
# training_cutoff = '2020-11-02'
# validation_cutoff = '2020-11-28'

# date2
training_agg_cutoff = '2020-08-05'
training_cutoff = '2020-08-31'
validation_cutoff = '2020-09-26'
days_to_predict = 3*31

# %% Creation of the experiument paths
# repository_name = input("name of experiment: ")
repository_name = f"experiment_features_{target_col}_{region_col}_{training_cutoff}_{validation_cutoff}"

print(
f"""

-------------------------------
0) Setting of the environment folder : {repository_name}
""")

result_path = os.path.join("../results", repository_name)
model_path = os.path.join("../models", repository_name)

if not os.path.exists(result_path):
    print(
"""
    Result repository doesn't exist. 
    Creation of experiment repository...""")
    os.makedirs(result_path)

if not os.path.exists(model_path):
    print(
"""
    Model repositories doesn't exist. 
    Creation of experiment repository...""")
    os.makedirs(model_path)
    print(
"""
    Creation of experiment repository: done.
    
""")


# %% Target and column names


date_col = 'date'

population = 'population'
tests_col = 'people_tested'
new_cases = True
infection_period = 3
severe_infections = .15

restriction_dict = {
    "fips":
        {
            "state": [
                      "Massachusetts",
                      "New Jersey",
                      "Connecticut", "New Hampshire",
                      "Florida", "California"
                      ],
            "county": ["Queens", "New York", "Bronx"]
        },
    "state": {"state": ["Alabama","Alaska","Arizona","Arkansas","California","Colorado",
  "Connecticut","Delaware","Florida","Georgia","Hawaii","Idaho","Illinois",
  "Indiana","Iowa","Kansas","Kentucky","Louisiana","Maine","Maryland",
  "Massachusetts","Michigan","Minnesota","Mississippi","Missouri","Montana",
  "Nebraska","Nevada","New Hampshire","New Jersey","New Mexico","New York",
  "North Carolina","North Dakota","Ohio","Oklahoma","Oregon","Pennsylvania",
  "Rhode Island","South Carolina","South Dakota","Tennessee","Texas","Utah",
  "Vermont","Virginia","Washington","West Virginia","Wisconsin","Wyoming"]
    }
}
# %% Run Parameters

random_state = 42

    # "state": {"state": ["Alabama","Alaska","Arizona","Arkansas","California","Colorado",
    # "Connecticut","Delaware","Florida","Georgia","Hawaii","Idaho","Illinois",
    # "Indiana","Iowa","Kansas","Kentucky","Louisiana","Maine","Maryland",
    # "Massachusetts","Michigan","Minnesota","Mississippi","Missouri","Montana",
    # "Nebraska","Nevada","New Hampshire","New Jersey","New Mexico","New York",
    # "North Carolina","North Dakota","Ohio","Oklahoma","Oregon","Pennsylvania",
    # "Rhode Island","South Carolina","South Dakota","Tennessee","Texas","Utah",
    # "Vermont","Virginia","West Virginia","Washington","Wisconsin","Wyoming"]



# %% backtesting parameters
regions_dict = {
    "fips": [25017, 34023],
    "state": ['Massachusetts'],
}

zip_codes = [2139, 2141]
use_zips = True

date_start = str(datetime.strptime(validation_cutoff, "%Y-%m-%d") + timedelta(days=1))[0:10]
date_end = str(datetime.strptime(validation_cutoff, "%Y-%m-%d") + timedelta(days=days_to_predict))[0:10]

regions = regions_dict[region_col]  # regions to predict  #
unformatted_dates = [datetime.strftime(_, "%Y-%m-%d") for _ in date_range(date_start, date_end, freq="1D")]


n_samples = 3

# %% Load and Save Parameters

train_knn = False
train_mdp = False
# NEW
train_rmdp = False
train_sir = False
train_bilstm = False

train_agg = True

train_knn_agg = False
train_mdp_agg = True
# NEW
train_rmdp_agg = False
train_sir_agg = False
train_bilstm_agg = False

train_ci = False
train_preval = False

train = True
retrain = False
batcktest = True
test = False

train_knn_ret = False
train_mdp_ret = False
# NEW
train_rmdp_ret = False
train_sir_ret = False
train_bilstm_ret = False

load_knn = True
load_mdp = True
load_rmdp = True
load_bilstm = True
load_sir = False

load_agg = True

load_knn_agg = True
load_mdp_agg = True
load_rmdp_agg = True
load_bilstm_agg = True
load_sir_agg = False


load_ci = False
load_preval = False
load_new_cases = False

sir_file = os.path.join(model_path, 'sir_{}_{}_{}.pickle'.format(training_cutoff.replace("-", ""), target_col, region_col))
knn_file = os.path.join(model_path, 'knn_{}_{}_{}.pickle'.format(training_cutoff.replace("-", ""), target_col, region_col))
mdp_file = os.path.join(model_path, 'mdp_{}_{}_{}.pickle'.format(training_cutoff.replace("-", ""), target_col, region_col))
rmdp_file = os.path.join(model_path, 'rmdp_{}_{}_{}.pickle'.format(training_cutoff.replace("-", ""), target_col, region_col))
agg_file = os.path.join(model_path, 'agg_{}_{}_{}.pickle'.format(training_cutoff.replace("-", ""), target_col, region_col))
bilstm_file = os.path.join(model_path, 'bilstm_{}_{}_{}.pickle'.format(training_cutoff.replace("-", ""), target_col, region_col))
ci_file = os.path.join(model_path, 'ci_{}_{}_{}.pickle'.format(training_cutoff.replace("-", ""), target_col, region_col))
preval_file = os.path.join(model_path, 'preval_{}_{}_{}.pickle'.format(training_cutoff.replace("-", ""), target_col, region_col))
export_file = 'export_{}_{}.csv'.format(training_cutoff.replace("-", ""), target_col, region_col)

# %% Parameters SIR

nmin = {"cases": 
            {"fips": 1, "state": 1},
        "deaths":
            {"fips": 1, "state": 1}
        }    
nmin = nmin[target_col]

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
        'date': date_col,
        'region': region_col,
        'target': target_col,
        'r': 3,
        'extra_features': ['temperature']
    }



# %% Parameters MDP

region_exceptions_dict = {
    "state":
        [
         ],
    "fips":
        []}

mdp_features_dict = \
    {
        'state':
            {"deaths": ["cases_pct3", 
                        "cases_pct5",
                        # "cases_pct30",
                        # "cases_pct3_lag30", 
                        "temperature_7",
                        # "prcp_median_scaled_5"
                        ],
             "cases": ["cases_pct3", 
                       "cases_pct5",
                       "cases_pct30",
                    #    "cases_pct3_lag30", 
                       "temperature",
                       "temperature_7"]},
                       
                    #    "prcp_median_scaled_5"]},  # ["cases_nom", "cases_pct3", "cases_pct5"]},
        'fips':
            {"deaths": [],
             "cases": []}
    }

mdp_params_dict = \
    {
        "days_avg": 3,  # 3
        "horizon": 8,  # 8
        "n_iter": 100,
        "n_folds_cv": 4,
        "clustering_distance_threshold": 0.1,
        "splitting_threshold": 0.,
        "classification_algorithm": 'DecisionTreeClassifier',
        "clustering_algorithm": 'Agglomerative',
        "n_clusters": None,
        "action_thresh": ([], 0),  # ([-250, 200], 1),
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

# %% Parameters rMDP

rmdp_features_dict = \
    {
        'state':
            {"deaths": [
                 # growth rate variables
                 "cases_pct3", "cases_pct5", 
                 "cases_pct10", "cases_pct20",
                 "cases_nom", "deaths_nom",
                 "cases_pct3_lag7", "cases_pct3_lag14", 
                 "cases_pct3_lag30", 
                 # "cases_pct3_lag60",
                 "cases_pct5_lag7", "cases_pct5_lag14", 
                 "cases_pct5_lag30", 
                 # "cases_pct5_lag60", 

                 # relative growth rates
                 "cases_r1w_2w", "cases_r1w_1m", "cases_r3d_2w", "death2cases_2w",

                 # holidays & population & racial
                 "isholiday", 
                 # "popdensity", 'in_school_pct_popdensity', 'median_income',
                 ## "civilian_labor_force_pct_popdensity",

                ##  "black_pop_pct_popdensity", "black_pop_pct_white_pop",
                ##  "other_race_pop_pct_popdensity", "other_race_pop_pct_white_pop",
                ##  "not_us_citizen_pop_pct_popdensity", "not_us_citizen_pop_pct_white_pop",
                ##  "not_hispanic_pop_pct_popdensity", "not_hispanic_pop_pct_white_pop",
                ##  "hispanic_pop_pct_popdensity", "hispanic_pop_pct_white_pop",
                ##  "asian_pop_pct_popdensity", "asian_pop_pct_white_pop",

                ##  "white_pop_pct_popdensity",
                ##  "pop_determined_poverty_status_pct_popdensity",

                 # change in mobility
                 "commute_90_more_mins",

                 # temperature
                 "temperature", "temperature_3", "temperature_7",
                 # "prcp_median_scaled_5"
                 ],
             "cases": [
                 # growth rate variables
                 "cases_pct3", "cases_pct5", 
                 "cases_pct10", "cases_pct20",
                 "cases_nom", "deaths_nom",
                 "cases_pct3_lag7", 
                 "cases_pct3_lag30", 
                 # "cases_pct3_lag60",
                 "cases_pct5_lag7", 
                 "cases_pct5_lag30", 
                 # "cases_pct5_lag60", 

                 # relative growth rates
                 "cases_r1w_2w", "cases_r1w_1m", "cases_r3d_2w", "death2cases_2w",

                 # holidays & population & racial
                 "isholiday", 
                 # "popdensity", 'in_school_pct_popdensity', 'median_income',
                 ## "civilian_labor_force_pct_popdensity",

                ##  "black_pop_pct_popdensity", "black_pop_pct_white_pop",
                ##  "other_race_pop_pct_popdensity", "other_race_pop_pct_white_pop",
                ##  "not_us_citizen_pop_pct_popdensity", "not_us_citizen_pop_pct_white_pop",
                ##  "not_hispanic_pop_pct_popdensity", "not_hispanic_pop_pct_white_pop",
                ##  "hispanic_pop_pct_popdensity", "hispanic_pop_pct_white_pop",
                ##  "asian_pop_pct_popdensity", "asian_pop_pct_white_pop",

                ##  "white_pop_pct_popdensity",
                ##  "pop_determined_poverty_status_pct_popdensity",

                 # change in mobility
                 "commute_90_more_mins",

                 # temperature
                 "temperature", "temperature_7", "prcp_median_scaled_5"
                 ]
             },
        'fips':
            {"deaths": [],
             "cases": []}
    }
    

rmdp_params_dict = \
    {
        "days_avg": 3,
        "horizon": 8,
        "test_horizon": 8,
        "error_computing": "horizon",
        "error_function_name": "exp_relative",
        "alpha": 2e-3,
        "n_iter": 150,
        "n_folds_cv": 30,
        "randomized": True,
        "randomized_split_pct": 0.7,
        "reward_name": "RISK",
        "clustering_distance_threshold": 0.09,
        "splitting_threshold": 0.,
        "classification_algorithm": "DecisionTreeClassifier",
        "clustering_algorithm": 'Agglomerative',  # 'Agglomerative'
        "n_clusters": None,
        "action_thresh":  ([], None),  # ([-0.2, 0.05], 1, 10),  # ([-250, 200], 1, None),  # ([], 0, None), # ([-0.2, 0.05], 1, 10)
        "features_list": rmdp_features_dict[region_col][target_col],
        "nfeatures": 10,
        "verbose": 1,
        "n_jobs": 24,
        "date_colname": date_col,
        "target_colname": target_col,
        "region_colname": region_col,
        "random_state": random_state,
        "keep_first": True,  # True,
        "save": True,
        "plot": False,
        "savepath": ""
    }

# %% Parameters AGG

per_region = True
ml_methods = [
    'lin',
    'elastic',
    # 'cart',
    # 'rf',
    # 'xgb',
    'linear_svm',
    # 'kernel_svm'
]
ml_hyperparams = {'xgb': {'gamma': [1, 5], 'max_depth': [3, 5]},
                  'cart': {'max_depth': [3, 5, 10, None]},
                  'rf': {'max_depth': [3, 5, 10, None], 'min_samples_leaf': [1, 2, 5]},
                  }

ml_mapping = {# 'lin': [LinearRegression(fit_intercept=False), False],
              'elastic': [ElasticNetCV(fit_intercept=False, positive=True), False],
              'xgb': [XGBRegressor(learning_rate=0.05, n_estimators=100, silent=True), True],
              'cart': [DecisionTreeRegressor(), True],
              'rf': [RandomForestRegressor(), True],
              'linear_svm': [LinearSVR(), False],
              'kernel_svm': [SVR(gamma='auto'), False],
              'nn': [MLPRegressor(learning_rate = 'constant', learning_rate_init = 1e-4, 
                     max_iter=500, early_stopping = True, 
                     n_iter_no_change = 1), False]}

# %% Parameters CI

ci_range = 0.95

# %% Parameters Prevalence

alpha = 0.11

# %% Date Formatting

if new_cases:
    unformatted_dates = [datetime.strftime(_, "%Y-%m-%d") for _ in date_range(min(unformatted_dates), max(unformatted_dates), freq="1D")]

out_of_sample_dates = [datetime.strptime(d, '%Y-%m-%d') for d in unformatted_dates]

if new_cases:
    out_of_sample_dates = list(reversed([min(out_of_sample_dates) - timedelta(i+1) for i in range( infection_period)])) + out_of_sample_dates

# %% Parameters Bi LSTM

bilstm_params_dict = \
    {
        "deterministic": True,
        'date': date_col,
        'region': region_col,
        'target': target_col,
        "use_auto": False
    }
