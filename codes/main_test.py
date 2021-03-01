# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 22:02:43 2020

@author: omars
"""

#%% Libraries and Parameters

from data_utils import (load_model)
from params import (load_sir, load_knn, load_mdp, load_bilstm, load_agg, load_ci, sir_file,
                    knn_file, mdp_file, bilstm_file, agg_file, ci_file, regions, dates,
                    random_state, df_path, n_samples, load_preval, preval_file, training_cutoff, new_cases, infection_period, severe_infections)
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")
import json
import pickle
import os
from copy import deepcopy
from uszipcode import SearchEngine
import pandas as pd

#%% Load Models and Make Predictions

if any([datetime.strptime(training_cutoff, '%Y-%m-%d') > date for date in dates]):
    raise Exception('Prediction dates appear in the training data. Please make predictions for a date after ' + training_cutoff)

search = SearchEngine(simple_zipcode=True)

# if use_zips:
#     fips = pd.read_csv(fips_path)
#     regions_dic = {zipcode_number:fips.query('ZIP == @zipcode_number').iloc[0, 1] for zipcode_number in zip_codes}
#     regions = list(set(regions_dic.values()))
#     ratio_dic = {}
# for zipcode_number in zip_codes:
#     fips_number = regions_dic[zipcode_number]
#     zip_list = set(fips.query('STCOUNTYFP == @fips_number')['ZIP'])
#     total_population = sum([search.by_zipcode(z).population for z in zip_list if search.by_zipcode(z).population is not None])
#     population = search.by_zipcode(zipcode_number).population
#     ratio = population/total_population
#
#     ratio_dic[zipcode_number] = ratio

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

if load_bilstm:
    bilstm = load_model(bilstm_file)
    output['bilstm'] = bilstm.predict(regions, dates)

if load_agg:
    agg = load_model(agg_file)
    output['agg'] = agg.predict(regions, dates, output)

with open(os.path.join('output_predictions.pickle'), 'wb') as fp:
    pickle.dump(output, fp)

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
        samples['samples'] = [[{state:samples['samples'][i][state][date] for state in samples['samples'][0].keys()} for date in range(len(samples['dates']))] for i in range(n_samples)]

        if load_preval:
            preval = load_model(preval_file)
            confirmed_samples = deepcopy(samples)
            samples = preval.convert(samples)

        if new_cases:
            new_samples = {}
            new_samples['dates'] = samples['dates'][infection_period:]
            #new_samples['samples'] = []
            tags = ['Total_uncontained_infections', 'Total_contained_infections', 'Total_confirmed_infections', 'New_uncontained_infections', 'New_contained_infections', 'New_confirmed_infections']

            l = []
            for i in range(len(samples['samples'])):
                cl_dic = {}
                for state in samples['samples'][i][0].keys():
                    sub_sample = [samples['samples'][i][j][state] -samples['samples'][i][j-1][state] for j in range(1, len(dates))]
                    total_new = [sum(sub_sample[i+j] for i in range(infection_period)) for j in range(len(dates)-infection_period)]

                    total = [samples['samples'][i][j][state] for j in range(1, len(dates))]

                    total_severe = [severe_infections*a for a in total]
                    total_mild = [(1-severe_infections)*a for a in total]

                    total_new_severe = [severe_infections*a for a in total_new]
                    total_new_mild = [(1-severe_infections)*a for a in total_new]
                    confirmed_sub_sample = [confirmed_samples['samples'][i][j][state] -samples['samples'][i][j-1][state] for j in range(1, len(dates))]
                    confirmed_total_new = [sum(confirmed_sub_sample[i+j] for i in range(infection_period)) for j in range(len(dates)-infection_period)]

                    confirmed_sub_sample_total = [confirmed_samples['samples'][i][j][state] for j in range(1, len(dates))]
                    confirmed_total = [sum(confirmed_sub_sample_total[i+j] for i in range(infection_period)) for j in range(len(dates)-infection_period)]

                    cl_dic[state] = (total_severe, total_mild, confirmed_total, total_new_severe, total_new_mild, confirmed_total_new)
                sample_dic = [{state: {tags[i]: cl_dic[state][i][j] for i in range(len(tags))} for state in samples['samples'][i][0].keys()} for j in range(len(dates)-infection_period)]
                if use_zips:
                    sample_dic = [{zipcode_number: {tags[i]: cl_dic[regions_dic[zipcode_number]][i][j]*ratio_dic[zipcode_number] for i in range(len(tags))} for zipcode_number in zip_codes} for j in range(len(dates)-infection_period)]
                l.append(sample_dic)
            new_samples['samples'] = l
            samples = new_samples


        with open(os.path.join(pathstr[0], model_type + '_prevalence_output_samples.json'), 'w') as fp:
            json.dump(samples, fp)

