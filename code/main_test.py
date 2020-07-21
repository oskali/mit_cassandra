# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 22:02:43 2020

@author: omars
"""

#%% Libraries and Parameters

from data_utils import (load_model)
from params import (load_sir, load_knn, load_mdp, load_agg, load_ci, sir_file,
                    knn_file, mdp_file, agg_file, ci_file, regions, dates,
                    random_state, df_path, n_samples)
import warnings
warnings.filterwarnings("ignore")
import json
import os

#%% Load Models and Make Predictions

output = {}
if load_sir:
    sir = load_model(sir_file)
    output['sir'] = sir.predict(regions, dates)

if load_knn:
    knn = load_model(knn_file)
    output['knn'] = knn.predict(regions, dates)

if load_mdp:
    mdp = load_model(mdp_file)
    output['mdp'] = mdp.predict(regions, dates)

if load_agg:
    agg = load_model(agg_file)
    output['agg'] = agg.predict(regions, dates, output)

if load_ci:
    ci = load_model(ci_file)
    sampled_output = ci.sample(output, n_samples, random_state)
    #Generate JSONs with random samples per model
    pathstr = os.path.split(df_path)
    for model_type in sampled_output.keys():
        prediction_distribution = dict.fromkeys(regions)
        for state in regions:
            predictions = sampled_output[model_type][state]
            prediction_distribution[state] = predictions
        all_samples = []
        date_list = sampled_output[model_type][regions[0]][0].index.strftime('%Y-%m-%d').tolist()
        samples = dict(dates=date_list, samples=None)
        for t_i in range(n_samples):
            sample_dict = dict.fromkeys(regions)
            for state in regions:
                sample_dict[state] = list(prediction_distribution[state][t_i])
            all_samples.append(sample_dict)
        samples['samples'] = all_samples
    with open(os.path.join(pathstr[0], model_type + '_prevalence_output_samples.json'), 'w') as fp:
        json.dump(samples, fp)