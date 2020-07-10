# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 21:24:25 2020

@author: omars
"""

#%% Libraries and Parameters

from data_utils import (save_model, load_data, dict_to_df)
from params import (train_sir, train_knn, train_mdp, train_agg, train_ci,
                    date_col, region_col, target_col, sir_file, knn_file, mdp_file, agg_file, ci_file, validation_cutoff, per_region, ml_methods, ml_mapping, ml_hyperparams, ci_range, knn_params_dict, sir_params_dict, mdp_params_dict)

from sir_model import SIRModel
from knn_model import KNNModel
from mdp_model import MDPModel
from agg_model import AGGModel
from confidence_intervals import CI
import warnings
warnings.filterwarnings("ignore")

#%% Load Data

df, df_train, df_validation = load_data(validation_cutoff=validation_cutoff)

#%% Train and Save Models

models = []
if train_sir:
    sir = SIRModel(**sir_params_dict)
    sir.fit(df_train)
    models.append('sir')
    save_model(sir, sir_file)

if train_knn:
    knn = KNNModel(**knn_params_dict)
    knn.fit(df_train)
    models.append('knn')
    save_model(knn, knn_file)

if train_mdp:
    if __name__ == "__main__":
        mdp = MDPModel(**mdp_params_dict)
        mdp.fit(df_train)
        models.append('mdp')
        save_model(mdp, mdp_file)

if train_agg:
    validation_predictions = {}
    regions_ = list(set(df_validation[region_col]))
    dates_ = list(set(df_validation[date_col]))
    if train_sir:
        validation_predictions['sir'] = sir.predict(regions_, dates_)
    if train_knn:
        validation_predictions['knn'] = knn.predict(regions_, dates_)
    if train_mdp:
        validation_predictions['mdp'] = mdp.predict(regions_, dates_)
    df_agg = dict_to_df(validation_predictions,
                        df_validation)
    agg = AGGModel(date=date_col,
                   region=region_col,
                   target=target_col,
                   models=models,
                   per_region=per_region,
                   ml_methods=ml_methods,
                   ml_mapping=ml_mapping,
                   ml_hyperparams=ml_hyperparams)
    agg.fit(df_agg)
    save_model(agg, agg_file)

    validation_predictions['agg'] = agg.predict(regions_, dates_, validation_predictions)
    df_agg = dict_to_df(validation_predictions,
                        df_validation)
    models.append('agg')

if train_ci:
    ci = CI(region_col=region_col,
            target_col=target_col,
            ci_range=ci_range,
            models=models)
    ci.fit(df_agg)
    save_model(ci, ci_file)



