# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 21:21:18 2020

@author: omars
"""
#%% Libraries

from datetime import datetime
import os
from pandas import date_range

#%% User and path

# df_path = r'C:\Users\david\Desktop\MIT\Courses\Research internship\covid19_team2\data\input\06_15_2020_states_combined.csv'
# df_path = r'C:\Users\david\Desktop\MIT\Courses\Research internship\covid19_team2\data\input\06_15_2020_states_combined_nom_n_pct.csv'
# df_path = r'C:\Users\david\Dropbox (MIT)\COVID-19-Team2\Data\07_16_2020_states_combined_w_pct.csv'
# df_path = r'C:\Users\david\Desktop\MIT\Courses\Research internship\covid19_team2\data\input\05_27_states_combined_v2_w_trend.csv'
# df_path = "C:\\Users\\david\\Dropbox (MIT)\\COVID-19-Team2\Data\\07_11_2020_counties_combined_NYNJMA.csv"
# df_path = "C:\\Users\\david\\Dropbox (MIT)\\COVID-19-Team2\Data\\07_16_2020_states_combined_w_pct.csv"
# df_path = None
df_path = r'C:\Users\david\Dropbox (MIT)\COVID-19-Team2\Data\08_14_2020_states_and_countries_combined_restricted_.csv'

#%% Target and column names

target_col = 'cases'
date_col = 'date'
region_col = 'state'
population = 'population'
nmin = {"fips": 100, "state": 200}
tests_col = 'people_tested'

#%% Run Parameters

random_state = 42
training_agg_cutoff = '2020-07-15'
training_cutoff = '2020-08-01'
validation_cutoff = None

regions_dict = {
    "fips": [25017, 34023],
    "state": ['Massachusetts'],
}
regions = regions_dict[region_col]  # regions to predict  #
unformatted_dates = [datetime.strftime(_, "%Y-%m-%d") for _ in date_range('2020-08-02', '2020-12-15', freq="1D")]  # dates to predict  #

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

#%% Load and Save Parameters

train_mdp = True
train_mdp_gs = False
load_mdp = True

EXPERIMENT_NAME = '22 - 20200822 - Massachusetts with Boosted MDP new pred'
default_path = r"C:\Users\david\Dropbox (MIT)\COVID-19-Team2\Data"

mdp_file = lambda mode, folder : os.path.join(r"C:\Users\david\Desktop\MIT\Courses\Research internship\results",
                        EXPERIMENT_NAME,
                        "MDPs_without_actions",
                        mode,
                        folder,
                        "mdp_{}_{}_{}.pkl".format(training_cutoff.replace("-", ""), target_col, region_col))

mdp_gs_savepath = os.path.join(r"C:\Users\david\Desktop\MIT\Courses\Research internship\results",
                               EXPERIMENT_NAME)  # experiment name

mdp_gs_file = os.path.join(mdp_gs_savepath, 'mdp_gs.pkl')

# %% Parameters MDP

region_exceptions_dict = {
    "state":
        ['Guam', 'Northern Mariana Islands', 'Puerto Rico',
         'Diamond Princess',
         'Grand Princess', 'American Samoa', 'Virgin Islands',
         'Hawaii', "Benin", "Ecuador",
         "Jordan", "Lithuania", "Uganda",
         "Georgia"
         ],
    "fips":
        []}

mdp_features_dict = \
    {
        'state':
            {"deaths": ["cases_pct3", "cases_pct5", "cases_pct10"],
             "cases": ["cases_pct3", "cases_pct5", "cases_pct10"]},
        'fips':
            {"deaths": [],
             "cases": []}
    }

mdp_params_dict = \
    {
        "days_avg": 3,
        "horizon": 5,
        "test_horizon": 5,
        "error_computing": "horizon",
        "error_function_name": "exp_relative",
        "alpha": 2e-3,
        "n_iter": 350,
        "n_folds_cv": 4,
        "clustering_distance_threshold": 0.1,
        "splitting_threshold": 0.,
        "classification_algorithm": "RandomForestClassifier",
        "clustering_algorithm": 'Agglomerative',
        "n_clusters": None,
        "action_thresh": ([], 0),  # ([-250, 200], 1),
        "features_list": mdp_features_dict[region_col][target_col],
        "verbose": 2,
        "n_jobs": -1,
        "date_colname": date_col,
        "target_colname": target_col,
        "region_colname": region_col,
        "random_state": random_state,
        "keep_first": True,  # True,
        "save": True,
        "plot": True,
        "savepath": os.path.dirname(mdp_file("", ""))
    }


#%% Parameters MDP - Grid Search

mdp_hparams_dict = \
    {
        "days_avg": [3],
        "horizon": [5, 8],
        "n_iter": [140],
        "clustering_distance_threshold": [0.08, 0.1],
        "classification_algorithm": ['RandomForestClassifier'],
        "clustering_algorithm": ['Agglomerative'],
        "error_computing": ["exponential", "id", "horizon"],
        "alpha": [2e-3]
    }

# MDPGridSearch hyperparameters
mdp_gs_params_dict = \
    {
        "target_colname": target_col,
        "region_colname": region_col,
        "date_colname": date_col,
        "features_list": mdp_features_dict[region_col][target_col],
        "action_thresh": ([], 0), # ([-250, 200], 1),
        "hyperparams": mdp_hparams_dict,
        "n_folds_cv": 6,
        "verbose": 0,
        "n_jobs": 1,
        "mdp_n_jobs": 1,
        "random_state": random_state,
        "save": True,
        "plot": True,
        "savepath": mdp_gs_savepath,
        "ignore_errors": True,
        "mode": "TIME_CV",
    }


#%% Date Formatting
dates = [datetime.strptime(d, '%Y-%m-%d') for d in unformatted_dates]
