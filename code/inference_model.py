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

#%% Inference Model
class InferenceModel(Model):
    def sample(self,  t_0: str, n_samples: int, dates: list, input_samples: dict) -> dict:
        dir_path = os.path.dirname(os.path.abspath(__file__))
        repo_path = os.path.join(dir_path, '..', '..', '..')

        np.random.seed(self.model_parameters['random_seed'])
        random_state = self.model_parameters['random_seed']

        sir_file = pd.read_csv(os.path.join(
            repo_path, self.model_parameters['sir_file']))

        knn_file = pd.read_csv(os.path.join(
            repo_path, self.model_parameters['knn_file']))

        mdp_file = pd.read_csv(os.path.join(
            repo_path, self.model_parameters['mdp_file']))

        agg_file = pd.read_csv(os.path.join(
            repo_path, self.model_parameters['agg_file']))

        ci_file = pd.read_csv(os.path.join(
            repo_path, self.model_parameters['ci_file']))

        ci_file = pd.read_csv(os.path.join(
            repo_path, self.model_parameters['preval_file']))

        regions = ['Massachusetts', 'New York']
        output = {}

        sir = load_model(sir_file)
        output['sir'] = sir.predict(regions, dates)

        knn = load_model(knn_file)
        output['knn'] = knn.predict(regions, dates)

        mdp = load_model(mdp_file)
        output['mdp'] = mdp.predict(regions, dates)

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
        return samples