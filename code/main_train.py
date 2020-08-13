# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 21:24:25 2020

@author: omars
"""

#%% Libraries and Parameters

from data_utils import (save_model, load_model, load_data, dict_to_df)
from params import (train_sir, train_knn, train_mdp, train_agg, train_ci,
                    date_col, region_col, target_col, sir_file, knn_file, mdp_file, agg_file, ci_file,
                    validation_cutoff, training_cutoff, training_agg_cutoff,
                    per_region, ml_methods, ml_mapping, ml_hyperparams, ci_range,
                    knn_params_dict, sir_params_dict, mdp_params_dict, retrain,
                    train_mdp_agg, train_sir_agg, train_knn_agg,
                    load_sir, load_knn, load_mdp)

from sir_model import SIRModel
from knn_model import KNNModel
from mdp_model import MDPModel
from agg_model import AGGModel
from confidence_intervals import CI
import warnings
warnings.filterwarnings("ignore")

#%% Load Data

df, df_train, df_validation = load_data(validation_cutoff=validation_cutoff)
_, df_train_agg, df_validation_agg = load_data(training_cutoff=training_agg_cutoff,
                                               validation_cutoff=training_cutoff)

#%% Train and Save Models

models = []
regions_val = list(set(df_validation[region_col]))
dates_val = list(set(df_validation[date_col]))
validation_predictions = {}

if train_sir:
    sir = SIRModel(**sir_params_dict)
    sir.fit(df_train)
    save_model(sir, sir_file)
if load_sir:
    sir = load_model(sir_file)
    validation_predictions['sir'] = sir.predict(regions_val, dates_val)
    models.append('sir')

if train_knn:
    knn = KNNModel(**knn_params_dict)
    knn.fit(df_train)
    models.append('knn')
    save_model(knn, knn_file)
    validation_predictions['knn'] = knn.predict(regions_val, dates_val)
if load_knn:
    knn = load_model(knn_file)
    validation_predictions['knn'] = knn.predict(regions_val, dates_val)
    models.append('knn')

if train_mdp:
    if __name__ == "__main__":
        mdp = MDPModel(**mdp_params_dict)
        mdp.fit(df_train)
        models.append('mdp')
        save_model(mdp, mdp_file)
        validation_predictions['mdp'] = mdp.predict(regions_val, dates_val)
if load_mdp:
    mdp = load_model(mdp_file)
    validation_predictions['mdp'] = mdp.predict(regions_val, dates_val)
    models.append('mdp')

if train_agg:

    models_agg = []
    validation_predictions_agg = {}
    regions_agg = list(set(df_validation_agg[region_col]))
    dates_agg = list(set(df_validation_agg[date_col]))

    # train SEIRD
    if train_sir_agg:
        sir_agg = SIRModel(**sir_params_dict)
        sir_agg.fit(df_train_agg)
        save_model(sir_agg, sir_file.replace(".pickle", "_agg.pickle"))
    if load_sir:
        sir_agg = load_model(sir_file.replace(".pickle", "_agg.pickle"))
        validation_predictions_agg['sir'] = sir_agg.predict(regions_agg, dates_agg)
        models_agg.append('sir')

    # train kNN
    if train_knn_agg:
        knn_agg = KNNModel(**knn_params_dict)
        knn_agg.fit(df_train_agg)
        save_model(knn_agg, knn_file.replace(".pickle", "_agg.pickle"))
    if load_knn:
        knn_agg = load_model(knn_file.replace(".pickle", "_agg.pickle"))
        validation_predictions_agg['knn'] = knn_agg.predict(regions_agg, dates_agg)
        models_agg.append('knn')

    # train MDP
    if __name__ == "__main__":
        if train_mdp_agg:
            mdp_agg = MDPModel(**mdp_params_dict)
            mdp_agg.fit(df_train_agg)
            save_model(mdp_agg, mdp_file.replace(".pickle", "_agg.pickle"))
        if load_mdp:
            mdp_agg = load_model(mdp_file.replace(".pickle", "_agg.pickle"))
            validation_predictions_agg['mdp'] = mdp_agg.predict(regions_agg, dates_agg)
            models_agg.append('mdp')

    df_agg = dict_to_df(validation_predictions_agg,
                        df_validation_agg)

    agg = AGGModel(date=date_col,
                   region=region_col,
                   target=target_col,
                   models=models_agg,
                   per_region=per_region,
                   ml_methods=ml_methods,
                   ml_mapping=ml_mapping,
                   ml_hyperparams=ml_hyperparams)
    agg.fit(df_agg)
    save_model(agg, agg_file)

    validation_predictions['agg'] = agg.predict(regions_val, dates_val, validation_predictions)
    df_agg = dict_to_df(validation_predictions,
                        df_validation)
    models.append('agg')

if train_ci & train_agg:
    ci = CI(region_col=region_col,
            target_col=target_col,
            ci_range=ci_range,
            models=models)
    ci.fit(df_agg)
    save_model(ci, ci_file)

# if retrain:
#     if train_sir:
#         sir = SIRModel(**sir_params_dict)
#         sir.fit(df)
#         save_model(sir, sir_file)
#
#     if train_knn:
#         knn = KNNModel(**knn_params_dict)
#         knn.fit(df)
#         save_model(knn, knn_file)
#
#     if train_mdp:
#         mdp = MDPModel(**mdp_params_dict)
#         mdp.fit(df)
#         save_model(mdp, mdp_file)


