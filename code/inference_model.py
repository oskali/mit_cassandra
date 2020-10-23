# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 22:02:43 2020

@author: omars
"""

#%% Libraries
import datetime
import numpy as np
import os
import pandas as pd
from data_utils import (load_model)
from models.common.model import Model
from copy import deepcopy
from uszipcode import SearchEngine

#%% Inference Model
class InferenceModel(Model):
    def sample(self,  t_0: str, n_samples: int, dates: list, input_samples: dict) -> dict:

        search = SearchEngine(simple_zipcode=True)

        dir_path = os.path.dirname(os.path.abspath(__file__))
        repo_path = os.path.join(dir_path, '..', '..', '..')

        np.random.seed(self.model_parameters['random_seed'])
        random_state = self.model_parameters['random_seed']

        validation_cutoff = self.model_parameters['validation_cutoff']
        use_zips = self.model_parameters['use_zips']
        infection_period = self.model_parameters['infection_period']
        severe_infections = self.model_parameters['severe_infections']

        zip_codes = ['2139']

        fips = pd.read_csv(os.path.join(
            repo_path, self.model_parameters['fips_path']))

        sir_file = pd.read_csv(os.path.join(
            repo_path, self.model_parameters['sir_file']))

        knn_file = pd.read_csv(os.path.join(
            repo_path, self.model_parameters['knn_file']))

        mdp_file = pd.read_csv(os.path.join(
            repo_path, self.model_parameters['mdp_file']))

        bilstm_file = pd.read_csv(os.path.join(
            repo_path, self.model_parameters['bilstm_file']))

        agg_file = pd.read_csv(os.path.join(
            repo_path, self.model_parameters['agg_file']))

        ci_file = pd.read_csv(os.path.join(
            repo_path, self.model_parameters['ci_file']))

        preval_file = pd.read_csv(os.path.join(
            repo_path, self.model_parameters['preval_file']))

        output = {}

        if any([datetime.datetime.strptime(validation_cutoff, '%Y-%m-%d') <= date for date in dates]):
            raise Exception('Prediction dates appear in the training data. Please make predictions for a date after ' + validation_cutoff)

        if use_zips:
            regions_dic = {zipcode_number:fips.query('ZIP == @zipcode_number').iloc[0, 1] for zipcode_number in zip_codes}
            regions = list(set(regions_dic.values()))
            ratio_dic = {}
        for zipcode_number in zip_codes:
            fips_number = regions_dic[zipcode_number]
            zip_list = set(fips.query('STCOUNTYFP == @fips_number')['ZIP'])
            total_population = sum([search.by_zipcode(z).population for z in zip_list if search.by_zipcode(z).population is not None])
            population = search.by_zipcode(zipcode_number).population
            ratio = population/total_population

            ratio_dic[zipcode_number] = ratio

        sir = load_model(sir_file)
        output['sir'] = sir.predict(regions, dates)

        knn = load_model(knn_file)
        output['knn'] = knn.predict(regions, dates)

        mdp = load_model(mdp_file)
        output['mdp'] = mdp.predict(regions, dates)

        bilstm = load_model(bilstm_file)
        output['bilstm'] = bilstm.predict(regions, dates)

        agg = load_model(agg_file)
        output['agg'] = agg.predict(regions, dates, output)

        ci = load_model(ci_file)

        preval = load_model(preval_file)

        sampled_output = ci.sample(output, n_samples, random_state)
        model_type = 'agg'
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
        preval = load_model(preval_file)
        samples = preval.convert(samples)


        sampled_output = ci.sample(output, n_samples, random_state)
        #Generate JSONs with random samples per model
        model_type = 'agg'
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

        preval = load_model(preval_file)
        confirmed_samples = deepcopy(samples)
        samples = preval.convert(samples)

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

        return samples
