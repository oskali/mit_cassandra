# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 15:45:50 2020

@author: omars
"""

#%% Libraries
from copy import deepcopy

#%% Model

class PrevalenceModel():

    def __init__(self,
                 region_col='state',
                 date_col='date',
                 tests_col='people_tested',
                 population_col='population',
                 alpha=0.11):
        self.region_col = region_col
        self.date_col = date_col
        self.tests_col = tests_col
        self.population_col = population_col
        self.alpha = alpha
        self.dict_prob_test = {}
        self.dict_population = {}

    def fit(self,
            df):

        self.dict_prob_test = dict((df.groupby(self.region_col)[self.tests_col].last() / df.groupby(self.region_col)[self.population_col].last()).dropna())

        self.dict_population = dict(df.groupby(self.region_col)[self.population_col].last().dropna())

    def convert(self,
                samples):
        n_samples = len(samples['samples'])
        n_dates = len(samples['samples'][0])
        states = list(samples['samples'][0][0].keys())
        conv_samples = deepcopy(samples)
        for i in range(n_samples):
            for j in range(n_dates):
                for state in states:
                    conv_samples['samples'][i][j][state] = 100*(samples['samples'][i][j][state]*(self.alpha + (1-self.alpha)*self.dict_prob_test[state])/(self.dict_prob_test[state]))/(self.dict_population[state])
        return conv_samples



