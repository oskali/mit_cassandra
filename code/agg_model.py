# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 00:42:44 2020

@author: omars
"""

#%% Libraries

import numpy as np
import pandas as pd
from data_utils import (mape)
from params import (ml_methods, ml_hyperparams, ml_mapping, per_region)
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
# from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR, LinearSVR
import operator
from copy import deepcopy

#%% Model

class AGGModel():

    def __init__(self,
                 date='date',
                 region='state',
                 target='cases',
                 models=None,
                 per_region=per_region,
                 ml_methods=ml_methods,
                 ml_mapping=ml_mapping,
                 ml_hyperparams=ml_hyperparams):
        self.date = date
        self.region = region
        self.target = target
        self.models = models
        self.per_region = per_region
        self.ml_methods = ml_methods
        self.ml_mapping = ml_mapping
        self.ml_hyperparams = ml_hyperparams
        self.regressor = {}
        self.mape_regions = {}
        self.model_regions = {}

    def fit(self,
            df):
        if not self.per_region:

            X_train = df.loc[:, self.models]
            y_train = df.loc[:, self.target]
            best_mape = np.infty
            best_model = None
            for model_name in self.ml_mapping.keys():
                if model_name in self.ml_methods:
                    if self.ml_mapping[model_name][1]:
                        model = GridSearchCV((self.ml_mapping)[model_name][0](), self.ml_hyperparams[model_name])
                    else:
                        model = self.ml_mapping[model_name][0]()

                    model.fit(X_train, y_train)
                    current_mape = mape(y_train, model.predict(X_train))
                    if current_mape < best_mape:
                        best_mape = current_mape
                        best_model = model
            self.regressor = {region_name: best_model for region_name in set(df[self.region])}

        else:

            regions_ = set(df[self.region])
            for region_name in regions_:
                df_sub = df[df[self.region] == region_name]
                length = df_sub.shape[0]
                self.mape_regions[region_name] = {}
                for model in self.models:
                    try:
                        self.mape_regions[region_name][model] = mape(y_true=df_sub[self.target], y_pred=df_sub[model])
                    except:
                        pass

                models_to_keep = [key for key, values in self.mape_regions[region_name].items() if values / np.sqrt(length) < 1e-2]
                if not models_to_keep:
                    models_to_keep = [min(self.mape_regions[region_name].items(), key=operator.itemgetter(1))[0]]

                self.model_regions[region_name] = models_to_keep

                X_train = df_sub.loc[:, self.model_regions[region_name]]
                y_train = df_sub.loc[:, self.target]
                best_mape = np.infty
                best_model = None
                for model_name in self.ml_mapping.keys():
                    if model_name in self.ml_methods:
                        if self.ml_mapping[model_name][1]:
                            model_instance = deepcopy(self.ml_mapping[model_name][0])
                            model = GridSearchCV(model_instance, self.ml_hyperparams[model_name])
                        else:
                            model = deepcopy(self.ml_mapping[model_name][0])

                        model.fit(X_train, y_train)
                        current_mape = mape(y_train, model.predict(X_train))
                        if current_mape < best_mape:
                            best_mape = current_mape
                            best_model = model
                self.regressor[region_name] = best_model

    def predict(self, regions, dates, output):
        predictions = {}
        for region in regions:
            try:
                l_prov = []
                for model in self.model_regions[region]:
                    l_prov.append(list(output[model][region]))

                X = np.array(l_prov).transpose()
                predictions[region] = pd.DataFrame(self.regressor[region].predict(X), columns=['agg'], index=dates)['agg']
            except KeyError:
                continue
        return predictions




