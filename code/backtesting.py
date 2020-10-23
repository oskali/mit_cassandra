# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 16:32:59 2020

@author: omars
"""

#%% Libraries

from data_utils import (load_data, dict_to_df, get_mapes, load_model)
from params import (load_sir, load_knn, load_mdp, load_bilstm, load_agg, load_ci, sir_file,
                    knn_file, mdp_file, bilstm_file, agg_file, ci_file, training_cutoff,
                    validation_cutoff, region_col, target_col,
                    date_col, export_file)#, add_countries)
import warnings

warnings.filterwarnings("ignore")

#%%

df, df_train, df_test = load_data(training_cutoff=training_cutoff, validation_cutoff=validation_cutoff)

regions = list(set(df_test[region_col]))
dates = list(set(df_test[date_col]))

#%% Load Models and Make Predictions

output = {}
models = []
if load_sir:
    sir = load_model(sir_file)
    output['sir'] = sir.predict(regions, dates)
    models.append('sir')
    
if load_knn:
    knn = load_model(knn_file)
    output['knn'] = knn.predict(regions, dates)
    models.append('knn')

if load_mdp:
    mdp = load_model(mdp_file)
    output['mdp'] = mdp.predict(regions, dates)
    models.append('mdp')

if load_bilstm:
    bilstm = load_model(bilstm_file)
    output['bilstm'] = bilstm.predict(regions, dates)
    models.append('bilstm')

if load_agg:
    agg = load_model(agg_file)
    output['agg'] = agg.predict(regions, dates, output)
    models.append('agg')

if load_ci:
    ci = load_model(ci_file)
    sampled_output = ci.sample(output)
    low_output = ci.sample(output, how='low')
    high_output = ci.sample(output, how='high')
    for model in models:
        output[model + '_low'] = low_output[model]
        output[model + '_high'] = high_output[model]
        output[model + '_sample'] = sampled_output[model]

df_agg = dict_to_df(output,
                    df_test)

results = get_mapes(df_agg,
                    models,
                    region_col=region_col,
                    target_col=target_col)

export = df.merge(df_agg.iloc[:, :-1], how='left', on=[region_col, date_col])
export.to_csv(export_file)
