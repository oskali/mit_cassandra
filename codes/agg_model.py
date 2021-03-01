# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 00:42:44 2020

@author: omars
"""

#%% Libraries

import numpy as np
import pandas as pd
from data_utils import (mape, wmape)
from params import (ml_methods, ml_hyperparams, ml_mapping, per_region)
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
# from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR, LinearSVR
from sklearn.metrics import mean_squared_error as mse
import operator
from copy import deepcopy
import math
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
                try:
                    df_sub = df[df[self.region] == region_name].sort_values(by=[self.date])
                    length = df_sub.shape[0]
                    self.mape_regions[region_name] = {}
                    for model in self.models:
                        if model != "sir":
                            try:
                                self.mape_regions[region_name][model] = mape(y_true=df_sub[self.target], y_pred=df_sub[model])
                            except:
                                print("except 1")
                                print(df_sub)
                                print(length)
                                pass

                    models_to_keep = [key for key, values in self.mape_regions[region_name].items() if values / np.sqrt(length) <= 1e-5]
                    if not models_to_keep:
                        models_to_keep = [min(self.mape_regions[region_name].items(), key=operator.itemgetter(1))[0]]

                    self.model_regions[region_name] = models_to_keep

                    X_train = df_sub.loc[:, self.model_regions[region_name]].diff().dropna().iloc[-15:, :]
                    y_train = df_sub.loc[:, self.target].diff().dropna().iloc[-15:]
                    best_score = np.infty
                    best_model = None
                    for model_name in self.ml_mapping.keys():
                        if model_name in self.ml_methods:
                            if self.ml_mapping[model_name][1]:
                                model_instance = deepcopy(self.ml_mapping[model_name][0])
                                model = GridSearchCV(model_instance, self.ml_hyperparams[model_name])
                            else:
                                model = deepcopy(self.ml_mapping[model_name][0])

                            model.fit(X_train, y_train)
                            try: 
                                model.coef_ = model.coef_ / (sum(model.coef_) + 1e-12)
                            except:
                                pass
                                print("err train (AGG): ", region_name, model_name)
                                # continue
                            current_score = mape(df_sub.loc[:, self.target].iloc[0] + np.cumsum(y_train), df_sub.loc[:, self.target].iloc[0] + np.cumsum(model.predict(X_train)))
                            if ( math.isnan(current_score) or math.isinf(current_score) ):
                                # print(region_name, model.predict(X_train))
                                current_score = wmape(df_sub.loc[:, self.target].iloc[0] + np.cumsum(y_train), df_sub.loc[:, self.target].iloc[0] + np.cumsum(model.predict(X_train)))

                                if ( math.isnan(current_score) or math.isinf(current_score) ):
                                    current_score = mse(y_train, model.predict(X_train))
                            
                            if current_score < best_score:
                                best_score = current_score
                                best_model = model
                    self.regressor[region_name] = best_model
                except:
                    print("ERROR: ", region_name)
                    pass

    def predict(self, regions, dates, output):
        predictions = {}
        for region in regions:
            # try:
            l_prov = pd.DataFrame(index=dates)
            for model in self.model_regions[region]:
                print(model)
                try:
                    output[model][region].name = model
                except KeyError:
                    print("DEBUG: ", output)

                l_prov = l_prov.join(output[model][region], how="inner")

            try:
                X = np.diff(l_prov.values, axis=0)
                predictions[region] = pd.DataFrame(
                    np.concatenate((np.array([0.]), self.regressor[region].predict(X)), axis=0), columns=['agg'], index=l_prov.index
                    )['agg'].cumsum() + l_prov.iloc[0].median()
            # except:
                # print("err: ", region)
                # print(l_prov)
            except:
                print("err: ", region)
                continue
        return predictions




