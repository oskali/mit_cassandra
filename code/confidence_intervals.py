# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 14:55:25 2020

@author: omars
"""

#%% Libraries
from scipy.stats import norm
from copy import deepcopy

#%% Model

class CI():

    def __init__(self,
                 region_col='state',
                 target_col='cases',
                 ci_range=0.75,
                 models=None):

        self.region_col = region_col
        self.target_col = target_col
        self.ci_range = ci_range
        self.models = models
        self.ci = {}
        self.mean = {}
        self.std = {}

    def fit(self,
            df_agg):

        df = deepcopy(df_agg)
        for model in self.models:
            df['error'] = ((df[model] - df[self.target_col])/df[model])
            grouped = df.groupby(self.region_col).agg({'error':['mean', 'std']}).reset_index()
            grouped.columns = [self.region_col, 'mean', 'std']

            self.mean[model] = {grouped[self.region_col].iloc[i] : grouped['mean'].iloc[i] for i in range(grouped.shape[0])}
            self.std[model] = {grouped[self.region_col].iloc[i] : grouped['std'].iloc[i] for i in range(grouped.shape[0])}

            {grouped[self.region_col].iloc[i] : norm.interval(self.ci_range, loc=grouped['mean'].iloc[i], scale=grouped['std'].iloc[i]) for i in range(grouped.shape[0])}

            self.ci[model] = {grouped[self.region_col].iloc[i] : norm.interval(self.ci_range, loc=grouped['mean'].iloc[i], scale=grouped['std'].iloc[i]) for i in range(grouped.shape[0])}

    def sample(self,
               output,
               n_samples=1,
               random_seed=42,
               how='random'):

        new_output = deepcopy(output)
        for model in new_output.keys():
            for region in new_output[model].keys():
                try:
                    if how == 'random':
                        if n_samples == 1:
                            seed = random_seed
                            while True:
                                spl = (norm.rvs(loc=self.mean[model][region], scale=self.std[model][region], random_state=seed)+1)
                                if spl > max(self.ci[model][region][0] + 1, 0) and spl < self.ci[model][region][1] + 1:
                                    new_output[model][region] = spl*output[model][region]
                                    break
                                else:
                                    seed += 1

                        else:
                            new_output[model][region] = []
                            seed = random_seed
                            while len(new_output[model][region]) < n_samples:
                                spl = (norm.rvs(loc=self.mean[model][region], scale=self.std[model][region], random_state=seed)+1)
                                if spl > max(self.ci[model][region][0] + 1, 0) and spl < self.ci[model][region][1] + 1:
                                    new_output[model][region].append(spl*output[model][region])
                                seed += 1
                    elif how == 'low':
                        new_output[model][region] = (self.ci[model][region][0]+1)*output[model][region]
                    else:
                        new_output[model][region] = (self.ci[model][region][1]+1)*output[model][region]

                except KeyError:
                    continue

        return new_output

