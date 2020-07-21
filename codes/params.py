# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 21:21:18 2020

@author: omars
"""
#%% Libraries

from datetime import datetime
import os

#%% User and path

USER = 'omar'

if USER == 'omar':
    # df_path = r'C:\Users\david\Desktop\MIT\Courses\Research internship\covid19_team2\data\input\06_15_2020_states_combined.csv'
    # df_path = r'C:\Users\david\Desktop\MIT\Courses\Research internship\covid19_team2\data\input\06_15_2020_states_combined_nom_n_pct.csv'
    # df_path = r'C:\Users\david\Desktop\MIT\Courses\Research internship\covid19_team2\data\input\07_08_2020_states_combined_w_pct.csv'
    df_path = r'C:\Users\david\Desktop\MIT\Courses\Research internship\covid19_team2\data\input\05_27_states_combined_v2_w_trend.csv'

#%% Target and column names

target_col = 'cases'
date_col = 'date'
region_col = 'state'
population = 'population'
nmin = 100

#%% Run Parameters

random_state = 42
training_cutoff = '2020-06-15'
validation_cutoff = '2020-06-28'
regions = ['New York', 'Massachusetts']  # regions to predict
unformatted_dates = ['2020-07-01', '2020-08-15']  # dates to predict

#%% Load and Save Parameters

train_knn = False
train_mdp = True
train_mdp_gs = False
train_sir = False
load_knn = False
load_mdp = True
load_sir = False
sir_file = 'sir.pickle'
knn_file = 'knn.pickle'
mdp_file = os.path.join(os.path.dirname(os.getcwd()), 'models', "20200720", 'mdp_timecv150_DT_h8_10pct_cases_wo_act.pickle')

mdp_gs_savepath = os.path.join(r"C:\Users\david\Desktop\MIT\Courses\Research internship\results",
                                           '10 - 20200714 Action learning')  # experiment name
mdp_gs_file = os.path.join(mdp_gs_savepath, 'mdp_gs.pickle')


#%% Parameters SIR

sir_params_dict = \
    {
        "nmin": 120,
        "optimizer": 'Nelder-Mead',
        "initial_param": [0.4, 0.06],
        "nmin_train_set": 10
    }
#%% Parameters KNN

knn_params_dict = \
    {
        "deterministic": True,
}


#%% Parameters MDP

mdp_features_dict = \
    {
        "deaths": ["mobility_score_trend", "cases_pct3", "cases_pct5"],
        "cases": ["mobility_score_trend", "cases_pct3", "cases_pct5"]
    }

mdp_params_dict = \
    {
        "days_avg": 3,
        "horizon": 8,
        "n_iter": 150,
        "n_folds_cv": 5,
        "clustering_distance_threshold": 0.1,
        "splitting_threshold": 0.,
        "classification_algorithm": 'DecisionTreeClassifier',
        "clustering_algorithm": 'Agglomerative',
        "n_clusters": None,
        "action_thresh": ([-250, 200], 1),
        "features_list": mdp_features_dict[target_col],
        "verbose": 2,
        "n_jobs": 4,
        "date_colname": date_col,
        "target_colname": target_col,
        "region_colname": region_col,
        "random_state": random_state,
        "keep_first": True,
        "save": True,
        "savepath": os.path.dirname(mdp_file),
    }


#%% Parameters MDP - Grid Search

mdp_hparams_dict = \
    {
        "days_avg": [3],
        "horizon": [8, 15],
        "n_iter": [100],
        "clustering_distance_threshold": [0.08, 0.1],
        "classification_algorithm": ['DecisionTreeClassifier', 'RandomForestClassifier'],
        "clustering_algorithm": ['Agglomerative'],
    }

# MDPGridSearch hyperparameters
mdp_gs_params_dict = \
    {
        "target_colname": target_col,
        "region_colname": region_col,
        "date_colname": date_col,
        "features_list": mdp_features_dict[target_col],
        "action_thresh": ([-350, -250, 200], 2),
        "hyperparams": mdp_hparams_dict,
        "n_folds_cv": 3,
        "verbose": 0,
        "n_jobs": 3,
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
