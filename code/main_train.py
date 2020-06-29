# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 21:24:25 2020

@author: omars
"""

#%% Libraries and Parameters

from data_utils import (save_model, load_data)
from params import (train_sir, train_knn, train_mdp,
                    nmin, date_col, region_col, target_col, population,
                    optimizer, initial_param, deterministic, days_avg,
horizon, n_iter, n_folds_cv, clustering_distance_threshold, splitting_threshold, classification_algorithm, clustering_algorithm, n_clusters, action_thresh, features_list, verbose, random_state, sir_file, knn_file, mdp_file)

from sir_model import SIRModel
from knn_model import KNNModel
from mdp_model import MDPModel
import warnings
warnings.filterwarnings("ignore")

#%% Load Data
_, df_train, _ = load_data()

#%% Train and Save Models
if train_sir:
    sir = SIRModel(nmin=nmin,
                   date=date_col,
                   region=region_col,
                   target=target_col,
                   population=population,
                   optimizer=optimizer,
                   initial_param=initial_param)
    sir.fit(df_train)
    save_model(sir, sir_file)

if train_knn:
    knn = KNNModel(date=date_col,
                   region=region_col,
                   target=target_col,
                   deterministic=deterministic)
    knn.fit(df_train)
    save_model(knn, knn_file)

if train_mdp:
    mdp = MDPModel(days_avg=days_avg,
                   horizon=horizon,
                   n_iter=n_iter,
                   n_folds_cv=n_folds_cv,
                   clustering_distance_threshold=clustering_distance_threshold,
                   splitting_threshold=splitting_threshold,
                   classification_algorithm=classification_algorithm,
                   clustering_algorithm=clustering_algorithm,
                   n_clusters=n_clusters,
                   action_thresh=action_thresh,
                   date_colname=date_col,
                   region_colname=region_col,
                   features_list=features_list,
                   target_colname=target_col,
                   random_state=random_state,
                   verbose=verbose)
    mdp.fit(df_train)
    save_model(mdp, mdp_file)