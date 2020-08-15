import numpy as np
import pandas as pd
import os
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima_model import ARIMA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.stats as stats

from lazypredict.Supervised import LazyRegressor
from sklearn.model_selection import train_test_split
from datetime import timedelta
from data_utils import *
from datetime import datetime

import math
from sklearn.preprocessing import MinMaxScaler
import random


def _mape(y_pred, y_true):
    return np.abs((y_true - y_pred) / y_true)


_mape = np.vectorize(_mape)

def predict_sml(row, prediction_size, model, scaler):
    df_frame = row[1:].to_frame().T
    current_state = scaler.inverse_transform(row[1:].values).tolist()
    predictions = []
    for _ in range(prediction_size):
        y = model.predict(pd.DataFrame(
            data=[scaler.inverse_transform(current_state)],
            index=[df_frame.index],
            columns=df_frame.columns))[0]
        predictions.append(y)
        current_state = [y] + current_state[:-1]
    return np.array(predictions)


#% Benchmark Experiment settings
class BenchMarkSettings:

    """

    Benchmark Setting Class:

    This class intends to define the experiment settings around a benchmark analyses.
    Instances of this class are used as dependencies to following BenchMarkModel classes:
    - SMLBenchMarkModel : Simple Machine Learning Models as benchmarks
    - LSTMBenchMarkModel : Long-Short Term model as a benchmark
    - KSMABenchMarkModel : kNN / (chained) SEIRD / MDP / Aggregated models

    """

    def __init__(self, region_colname, target_colname, date_colname, training_cutoff,
                 prediction_size=14, metric=_mape, useful_columns=None, savepath=None, verbose=0):
        self.region_colname = region_colname
        self.date_colname = date_colname
        self.target_colname = target_colname
        self.useful_columns = useful_columns
        self.training_cutoff = training_cutoff
        self.prediction_size = prediction_size
        self.metric = metric
        self.verbose = verbose
        self.savepath = savepath


#% Regular Machine Learning Models
class SMLBenchMarkModel:

    def __init__(self, benchmark):
        self.__dict__ = benchmark.__dict__.copy()
        self._benchmark = benchmark
        self._raw_data = None
        self.X = None
        self.models_dict = dict()
        self.observed_metrics = None
        self.last_target = None
        self.test_error = None

    def sanity_check_test(self):
        self.X = self.X.dropna(subset=self.useful_columns).copy()
        group_to_keep = []
        for group_name, group in self.X.groupby(self.region_colname):
            group_ = pd.DataFrame(index=pd.date_range(group[self.date_colname].min(),
                                                      group[self.date_colname].max(),
                                                      freq="1D", name=self.date_colname))

            group_ = group_.join(group.set_index(self.date_colname), how="left")
            if group_[self.target_colname].notnull().all():
                group_to_keep.append(group_name)
            else:
                if self.verbose >= 1:
                    print("Warning: Missing dates. {} '{}' has been removed.".format(self.region_colname, group_name))

        self.X = self.X[self.X[self.region_colname].isin(group_to_keep)].copy()

    def preprocess(self, scaling=False):

        # get the raw data
        self.X = self._raw_data.sort_values(by=[self.region_colname, self.date_colname]).copy()

        # verify there is no missing dates otherwise remove the region
        self.sanity_check_test()

        self.X = self.X[[self.region_colname, self.date_colname] + self.useful_columns].copy()
        # cases features
        if self.target_colname == "cases":
            self.X["cases"] = self.X.groupby(self.region_colname)["cases"].pct_change().values
            for lag in range(1, 11):
                self.X["cases_{}".format(lag)] = self.X.groupby(self.region_colname)["cases"].shift(lag).values
            self.X.drop("deaths", axis=1, inplace=True)

        # death features
        elif self.target_colname == "deaths":
            self.X["deaths"] = self.X.groupby(self.region_colname)["deaths"].pct_change().values
            for lag in range(1, 11):
                self.X["deaths_{}".format(lag)] = self.X.groupby(self.region_colname)["deaths"].shift(lag).values
            self.X.drop("cases", axis=1, inplace=True)

        else:
            raise Exception

        # compute target the target - growth forward
        self.X["target"] = self.X.groupby(self.region_colname)[self.target_colname].shift(-1).values
        self.X["prediction_date"] = self.X.groupby(self.region_colname)[self.date_colname].shift(-1)

        # split down into training and testing
        X_train = self.X[self.X["prediction_date"] <= self.training_cutoff
                         ].replace([-np.inf, np.inf], np.nan).dropna().drop("prediction_date", axis=1).set_index(
            [self.region_colname, self.date_colname]).copy()

        X_test = self.X[self.X["prediction_date"] > self.training_cutoff
                        ].replace([-np.inf, np.inf], np.nan).dropna().drop("prediction_date", axis=1)
        X_test = X_test.groupby(self.region_colname).head(self.prediction_size).set_index(
            [self.region_colname, self.date_colname]).copy()

        X_all = self.X.drop("prediction_date", axis=1).set_index(
            [self.region_colname, self.date_colname]).copy()

        self.X, self.y = X_train.loc[:, ~(X_train.columns == "target")], X_train["target"]
        self.X_val, self.y_val = X_test.loc[:, ~(X_test.columns == "target")], X_test["target"]
        self.X_all = X_all.loc[:, ~(X_all.columns == "target")].replace([-np.inf, np.inf], np.nan).dropna()

        # save last target
        self.last_target = self._raw_data[self._raw_data[self.date_colname].shift(1) == self.training_cutoff].set_index(
            self.region_colname)[self.target_colname].to_dict()

        self.last_target_all = self._raw_data.groupby(
            self.region_colname).last()[self.target_colname].to_dict()

        # scale the data
        if scaling:
            # fit scaler
            self.scaler = StandardScaler()
            self.X = pd.DataFrame(data=self.scaler.fit_transform(self.X),
                                  index=self.X.index, columns=self.X.columns).copy()
            # apply scaler
            # self.X_val = pd.DataFrame(data=self.scaler.transform(self.X_val),
            #                           index=self.X_val.index, columns=self.X_val.columns).copy()
            #
            # self.X_all = pd.DataFrame(data=self.scaler.transform(self.X_all),
            #                           index=self.X_all.index, columns=self.X_all.columns).copy()

    def fit(self, data, scaling=True, save=True):

        # create training set
        self._raw_data = data.copy()
        self.preprocess(scaling=scaling)

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=.5, random_state=123)
        clf = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
        _, predictions = clf.fit(X_train, X_test, y_train, y_test, save=save)

        self.observed_metrics = predictions

        if save:
            self.models_dict = {key: value for key, value in clf.model_dict.items() if key in {"Ridge", "RandomForestRegressor", "XGBRegressor"}}

    def eval(self):

        # predictions
        predictions = pd.DataFrame(index=self.X_val.index)
        # predictions[self.date_colname] = predictions.groupby(by=self.region_colname)[self.date_colname].apply(
        #     lambda x: x + timedelta(self.prediction_size)).values

        # prediction
        for name, model in tqdm(self.models_dict.items()):
            prediction_model = []
            for region, row in self.X_val.reset_index().groupby(self.region_colname).first().iterrows():
                prediction_region = pd.DataFrame(
                    index=pd.date_range(start=row[0] + timedelta(1), periods=self.prediction_size, freq="1D", name=self.date_colname),
                    data=predict_sml(row, self.prediction_size, model, self.scaler),
                    columns=[name]).reset_index()
                prediction_region[self.region_colname] = region
                prediction_model.append(prediction_region.set_index([self.region_colname, self.date_colname]))
            prediction_model = pd.concat(prediction_model)
            predictions = predictions.join(prediction_model, how="left")

        predictions = (predictions+1).groupby(self.region_colname).cumprod().reset_index()
        dfs = []
        for group_name, group in predictions.groupby(self.region_colname):
            try:
                dfs.append(group.set_index([self.region_colname, self.date_colname]) * self.last_target[group_name])
            except KeyError:
                pass
        predictions = pd.concat(dfs)

        # add target
        predictions = predictions.join(
            self._raw_data.set_index([self.region_colname, self.date_colname])[self.target_colname], how="left")
        error = predictions.copy()

        self.val_predictions = predictions

        for name in self.models_dict.keys():
            error[name] = self.metric(error[name].values, error[self.target_colname].values)
        error.reset_index(inplace=True)
        error.drop([self.target_colname, self.date_colname], axis=1, inplace=True)

        error_per_state = error.groupby(self.region_colname).mean()
        error_last = error.dropna().groupby(self.region_colname).last()

        self.test_error = error_per_state
        self.last_error = error_last.copy()

        return error_per_state.describe().copy()

    def eval_dates(self, start_date, end_date):
        val = self.val_predictions.reset_index()
        val = val[val.date.between(start_date, end_date)]

        error = val.copy()
        for name in self.models_dict.keys():
            error[name] = self.metric(error[name].values, error[self.target_colname].values)
        error.drop([self.target_colname, self.date_colname], axis=1, inplace=True)

        error_per_state = error.groupby(self.region_colname).mean()
        error_last = error.dropna().groupby(self.region_colname).last()
        return error_per_state, error_last, val


#% Long-Short Term Memory
class LSTMBenchMarkModel:

    def __init__(self, benchmark, trunc=100):
        self.__dict__ = benchmark.__dict__.copy()
        self._benchmark = benchmark
        self._raw_data = None
        self.X = None
        self.models_dict = dict()
        self.observed_metrics = None
        self.last_target = None
        self.trunc = trunc
        self.column_order = None
        self.lstm_model = None
        self.test_error = None

    def sanity_check_test(self):
        self.X = self.X.dropna(subset=self.useful_columns).copy()
        group_to_keep = []
        for group_name, group in self.X.groupby(self.region_colname):
            group_ = pd.DataFrame(index=pd.date_range(group[self.date_colname].min(),
                                                      group[self.date_colname].max(),
                                                      freq="1D", name=self.date_colname))

            group_ = group_.join(group.set_index(self.date_colname), how="left")
            if group_[self.target_colname].notnull().all():
                group_to_keep.append(group_name)
            else:
                if self.verbose >= 1:
                    print("Warning: Missing dates. {} '{}' has been removed.".format(self.region_colname, group_name))

        self.X = self.X[self.X[self.region_colname].isin(group_to_keep)].copy()

    def preprocess(self, scaling=False):

        # get the raw data
        self.X = self._raw_data.sort_values(by=[self.region_colname, self.date_colname]).copy()

        # verify there is no missing dates otherwise remove the region
        self.sanity_check_test()

        self.X = self.X[[self.region_colname, self.date_colname, self.target_colname]].copy()

        # death features
        self.X[self.target_colname] = self.X.groupby(self.region_colname)[self.target_colname].pct_change().values
        self.X.dropna(subset=[self.target_colname], inplace=True)

        # crop data
        region_to_keep = self.X.groupby(self.region_colname)[self.target_colname].count()
        region_to_keep = region_to_keep[region_to_keep > self.trunc].index.tolist()
        self.X = self.X[self.X[self.region_colname].isin(region_to_keep)].groupby(self.region_colname).tail(
            self.trunc).copy()

        # compute target the target - growth forward
        self.X = self.X.pivot(index=self.date_colname, columns=self.region_colname, values=self.target_colname)
        self.column_order = self.X.columns

        # split down into training and testing
        X_train = self.X[self.X.index <= self.training_cutoff]
        X_train = [X_train[col].values.reshape((-1, 1)) for col in X_train.columns]

        X_test = self.X[self.X.index > self.training_cutoff]
        X_test = X_test.head(self.prediction_size)

        X_all = self.X.copy()
        X_all = [X_all[col].values.reshape((-1, 1)) for col in X_all.columns]

        # self.X, self.y = X_train.loc[:, ~(X_train.columns == "target")], X_train["target"]
        # self.X_val, self.y_val = X_test.loc[:, ~(X_test.columns == "target")], X_test["target"]
        # self.X_all = X_all.loc[:, ~(X_all.columns == "target")]
        # save last target
        self.last_target = self._raw_data[self._raw_data[self.date_colname] == self.training_cutoff].set_index(
            self.region_colname)[self.target_colname].to_dict()

        self.last_target_all = self._raw_data.groupby(
            self.region_colname).last()[self.target_colname].to_dict()

        # scale the data
        if scaling:

            normalized_data = []
            normalized_data_all = []
            scalers = []  # Store one scaler for each county

            for county_id in range(len(X_train)):
                scaler = MinMaxScaler(feature_range=(0, 1))
                scalers.append(scaler)

                normalized_county_data = scaler.fit_transform(X_train[county_id][:, 0:1])
                normalized_county_data_all = scaler.transform(X_all[county_id][:, 0:1])

                normalized_data.append(normalized_county_data)
                normalized_data_all.append(normalized_county_data_all)

            X_train = normalized_data
            X_all = normalized_data_all

            self.scaler = scalers

        X_train = np.dstack(X_train)
        X_train = np.moveaxis(X_train, 2, 0)
        X_all = np.dstack(X_all)
        X_all = np.moveaxis(X_all, 2, 0)

        self.X = X_train
        self.X_test = X_test
        self.X_all = X_all

    def fit(self, data, scaling=True):

        # create training set
        self._raw_data = data.copy()
        self.preprocess(scaling=scaling)

        OUTPUT_SIZE = self.prediction_size  # This is the number of days to predict
        TOTAL_LENGTH = self.X.shape[1] - OUTPUT_SIZE
        XY_SPLIT = TOTAL_LENGTH - OUTPUT_SIZE
        TRAIN_SPLIT = round(self.X.shape[0] * 0.8)

        if self.verbose >= 1:
            print("{} counties will be split into {} for training "
                  "and {} for testing".format(self.X.shape[0], TRAIN_SPLIT, self.X.shape[0] - TRAIN_SPLIT))
            print("Each time series has length {}. Training uses"
                  " the first {} days, and testing uses the last {} days.".format(self.X.shape[1], TOTAL_LENGTH,
                                                                                  TOTAL_LENGTH))
            print("Each input has length {} with a corresponding output of length {}.".format(XY_SPLIT, OUTPUT_SIZE))

        # Code used to including mobility, masks, and cases:
        train_X, train_Y = self.X[:TRAIN_SPLIT, :-OUTPUT_SIZE * 2], self.X[:TRAIN_SPLIT, -OUTPUT_SIZE * 2:-OUTPUT_SIZE]
        test_X, test_Y = self.X[TRAIN_SPLIT:, OUTPUT_SIZE:-OUTPUT_SIZE], self.X[TRAIN_SPLIT:, -OUTPUT_SIZE:]

        if self.verbose >= 1:
            print(train_X.shape)
            print(train_Y.shape)
            print(test_X.shape)
            print(test_Y.shape)

        tf.keras.backend.set_epsilon(1)
        simple_lstm_model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(20, input_shape=train_X.shape[-2:], return_sequences=False),
            tf.keras.layers.Dense(20),
            tf.keras.layers.Dense(train_Y.shape[1]),
        ])
        simple_lstm_model.compile(optimizer='adam', loss='mae')

        history = simple_lstm_model.fit(train_X, train_Y, epochs=30, batch_size=2, verbose=self.verbose, shuffle=False)

        if self.verbose >= 2:
            plt.plot(history.history['loss'], label='train')
            plt.legend()
            plt.show(block=False)

        if self.verbose >= 1:
            results = simple_lstm_model.evaluate(test_X, test_Y, batch_size=2)
            print("Test error:", results)

        self.lstm_model = simple_lstm_model

    def eval(self):

        test_predict = self.lstm_model.predict(self.X[:, :-self.prediction_size])

        # scale back
        for row_id in range(test_predict.shape[0]):
            test_predict[row_id, :] = self.scaler[row_id].inverse_transform(
                test_predict[row_id, :].reshape((-1, 1))).reshape(-1)

        # predicted growth rate
        X_pred = test_predict.reshape((test_predict.shape[0], test_predict.shape[1])).T
        X_pred = pd.DataFrame(data=(1. + X_pred).cumprod(axis=0),
                              columns=self.column_order.tolist(),
                              index=pd.date_range(start=self.training_cutoff,
                                                  periods=self.prediction_size,
                                                  freq="1D",
                                                  name=self.date_colname) + timedelta(1)
                              ).reset_index()
        X_pred = pd.melt(X_pred,
                         id_vars=self.date_colname,
                         value_name=self.target_colname,
                         var_name=self.region_colname).set_index([self.region_colname, self.date_colname])

        # true growth rate
        predictions = (self.X_test + 1.).cumprod(axis=0)
        predictions = pd.melt(predictions.reset_index(),
                              id_vars=self.date_colname,
                              value_name=self.target_colname,
                              var_name=self.region_colname)

        predictions["LSTM"] = X_pred.values
        self.val_predictions = predictions

        # add target
        error = predictions.copy()
        error["LSTM"] = self.metric(error["LSTM"].values, error[self.target_colname].values)
        error.drop([self.target_colname, self.date_colname], axis=1, inplace=True)

        error_per_state = error.groupby(self.region_colname).mean()
        error_last = error.dropna().groupby(self.region_colname).last()
        self.test_error = error_per_state
        self.last_error = error_last.copy()

        return error_per_state.describe().copy()

    def predict(self):

        # predictions

        test_predict = self.lstm_model.predict(self.X_all[:, :-self.prediction_size])

        # scale back
        for row_id in range(test_predict.shape[0]):
            test_predict[row_id, :] = self.scaler[row_id].inverse_transform(
                test_predict[row_id, :].reshape((-1, 1))).reshape(-1)

        # predicted growth rate
        X_pred = test_predict.reshape((test_predict.shape[0], test_predict.shape[1])).T
        X_pred = pd.DataFrame(data=(1. + X_pred).cumprod(axis=0),
                              columns=self.column_order.tolist(),
                              index=pd.date_range(start=self._raw_data[self.date_colname].max(),
                                                  periods=self.prediction_size,
                                                  freq="1D",
                                                  name=self.date_colname) + timedelta(1)
                              )

        for col in X_pred.columns:
            X_pred[col] *= self.last_target_all[col]

        X_pred = pd.melt(X_pred.reset_index(),
                         id_vars=self.date_colname,
                         value_name="LSTM",
                         var_name=self.region_colname)

        # true growth rate

        return X_pred

    def eval_dates(self, start_date, end_date):
        val = self.val_predictions.reset_index()
        val = val[val.date.between(start_date, end_date)]

        error = val.copy()
        for name in self.models_dict.keys():
            error[name] = self.metric(error[name].values, error[self.target_colname].values)
        error.drop([self.target_colname, self.date_colname], axis=1, inplace=True)

        error_per_state = error.groupby(self.region_colname).mean()
        error_last = error.dropna().groupby(self.region_colname).last()
        return error_per_state, error_last, val

#% kNN SEIRD MDP Aggregated
class KSMABenchMarkModel:

    def __init__(self, benchmark, models_path_dict, load_model_dict):
        self.__dict__ = benchmark.__dict__.copy()
        self._benchmark = benchmark
        self._raw_data = None
        self.models_path_dict = models_path_dict
        self.load_model_dict = load_model_dict
        self.observed_metrics = None
        self.last_target = None
        self.column_order = None
        self.used_models = None
        self.test_error = None

    def sanity_check_test(self):
        self.X = self.X.dropna(subset=self.useful_columns).copy()
        group_to_keep = []
        for group_name, group in self.X.groupby(self.region_colname):
            group_ = pd.DataFrame(index=pd.date_range(group[self.date_colname].min(),
                                                      group[self.date_colname].max(),
                                                      freq="1D", name=self.date_colname))

            group_ = group_.join(group.set_index(self.date_colname), how="left")
            if group_[self.target_colname].notnull().all():
                group_to_keep.append(group_name)
            else:
                if self.verbose >= 1:
                    print("Warning: Missing dates. {} '{}' has been removed.".format(self.region_colname, group_name))

        self.X = self.X[self.X[self.region_colname].isin(group_to_keep)].copy()

    def preprocess(self):

        # get the raw data
        self.X = self._raw_data.sort_values(by=[self.region_colname, self.date_colname]).copy()

        # verify there is no missing dates otherwise remove the region
        self.sanity_check_test()

        self.X = self.X[[self.region_colname, self.date_colname, self.target_colname]].copy()

        X_test = self.X[self.X[self.date_colname] > self.training_cutoff]
        self.X_test = X_test.groupby(self.region_colname).head(self.prediction_size)

        self.X_all = self.X.copy()

        # self.X, self.y = X_train.loc[:, ~(X_train.columns == "target")], X_train["target"]
        # self.X_val, self.y_val = X_test.loc[:, ~(X_test.columns == "target")], X_test["target"]
        # self.X_all = X_all.loc[:, ~(X_all.columns == "target")]
        # save last target
        self.last_target = self._raw_data[self._raw_data[self.date_colname] == self.training_cutoff].set_index(
            self.region_colname)[self.target_colname].to_dict()

        self.last_target_all = self._raw_data.groupby(
            self.region_colname).last()[self.target_colname].to_dict()

    def fit(self, data):

        # create training set
        self._raw_data = data.copy()
        self.preprocess()

    def eval(self):

        output = {}
        output_agg = {}
        regions = self.X_test[self.region_colname].unique()
        dates = pd.to_datetime(self.X_test[self.date_colname].unique())
        used_models = []

        if self.load_model_dict["sir"]:
            try:
                sir = load_model(self.models_path_dict["sir"])
                output_agg['sir'] = sir.predict(regions, dates)
                output['sir'] = pd.DataFrame.from_dict(output_agg['sir'])
                output['sir'] = pd.melt(output['sir'].reset_index(),
                                        id_vars="index",
                                        value_name="sir",
                                        var_name=self.region_colname).rename(
                    columns={"index": self.date_colname}).set_index(
                    [self.region_colname, self.date_colname])
                used_models.append("sir")
            except:
                pass

        if self.load_model_dict["mdp"]:
            try:
                mdp = load_model(self.models_path_dict["mdp"])
                output_agg['mdp'] = mdp.predict(regions, dates)
                output['mdp'] = pd.DataFrame.from_dict(output_agg['mdp'])
                output['mdp'] = pd.melt(output['mdp'].reset_index(),
                                        id_vars=self.date_colname,
                                        value_name="mdp",
                                        var_name=self.region_colname).set_index(
                    [self.region_colname, self.date_colname])
                used_models.append("mdp")
            except:
                pass

        if self.load_model_dict["knn"]:
            try:
                knn = load_model(self.models_path_dict["knn"])
                output_agg['knn'] = knn.predict(regions, dates)
                output['knn'] = pd.DataFrame.from_dict(output_agg['knn'])
                output['knn'] = pd.DataFrame.from_dict(knn.predict(regions, dates))
                output['knn'] = pd.melt(output['knn'].reset_index(),
                                        id_vars="index",
                                        value_name="knn",
                                        var_name=self.region_colname).rename(
                    columns={"index": self.date_colname}).set_index(
                    [self.region_colname, self.date_colname])
                used_models.append("knn")
            except:
                pass

        if self.load_model_dict["agg"]:
            try:
                agg = load_model(self.models_path_dict["agg"])
                output['agg'] = pd.DataFrame.from_dict(agg.predict(regions, dates, output_agg))
                output['agg'] = pd.melt(output['agg'].reset_index(),
                                        id_vars="index",
                                        value_name="agg",
                                        var_name=self.region_colname).rename(
                    columns={"index": self.date_colname}).set_index(
                    [self.region_colname, self.date_colname])
                used_models.append("agg")
            except:
                pass

        predictions = pd.DataFrame()

        try:
            key = random.choice(used_models)
            pred_models = [key]
            predictions = output[key].copy()
            for key_, pred in output.items():
                if key_ != key:
                    try:
                        predictions[key_] = predictions.join(pred, how='left')[key_]
                        pred_models.append(key_)
                    except:
                        pass

            self.used_models = pred_models

        except IndexError:
            self.used_models = []
            return predictions

        # true growth rate
        predictions[self.target_colname] = predictions.join(
            self.X_test.set_index([self.region_colname, self.date_colname]),
            how="left")[self.target_colname].values

        self.val_predictions = predictions
        error = predictions.reset_index().copy()

        for model_name in used_models:
            error[model_name] = self.metric(error[model_name].values, error[self.target_colname].values)

        error.drop([self.target_colname, self.date_colname], axis=1, inplace=True)

        error_per_state = error.dropna().groupby(self.region_colname).mean()
        error_last = error.dropna().groupby(self.region_colname).last()
        self.test_error = error_per_state
        self.last_error = error_last.copy()

        return error_per_state.describe().copy()

    def predict(self):

        output = {}
        output_agg = {}
        regions = self.X_test[self.region_colname].unique()
        dates = pd.date_range(self._raw_data.date.max()+timedelta(1), periods=self.prediction_size, freq="1D")

        if self.load_model_dict["sir"]:
            sir = load_model(self.models_path_dict["sir"])
            output_agg['sir'] = sir.predict(regions, dates)
            output['sir'] = pd.DataFrame.from_dict(output_agg['sir'])
            output['sir'] = pd.melt(output['sir'].reset_index(),
                                    id_vars="index",
                                    value_name="sir",
                                    var_name=self.region_colname).rename(
                columns={"index": self.date_colname}).set_index(
                [self.region_colname, self.date_colname])

        if self.load_model_dict["knn"]:
            knn = load_model(self.models_path_dict["knn"])
            output_agg['knn'] = knn.predict(regions, dates)
            output['knn'] = pd.DataFrame.from_dict(output_agg['knn'])
            output['knn'] = pd.DataFrame.from_dict(knn.predict(regions, dates))
            output['knn'] = pd.melt(output['knn'].reset_index(),
                                    id_vars="index",
                                    value_name="knn",
                                    var_name=self.region_colname).rename(
                columns={"index": self.date_colname}).set_index(
                [self.region_colname, self.date_colname])

        if self.load_model_dict["mdp"]:
            mdp = load_model(self.models_path_dict["mdp"])
            output_agg['mdp'] = mdp.predict(regions, dates)
            output['mdp'] = pd.DataFrame.from_dict(output_agg['mdp'])
            output['mdp'] = pd.melt(output['mdp'].reset_index(),
                                    id_vars=self.date_colname,
                                    value_name="mdp",
                                    var_name=self.region_colname).set_index(
                [self.region_colname, self.date_colname])

        if self.load_model_dict["agg"]:
            agg = load_model(self.models_path_dict["agg"])
            output['agg'] = pd.DataFrame.from_dict(agg.predict(regions, dates, output_agg))
            output['agg'] = pd.melt(output['agg'].reset_index(),
                                    id_vars="index",
                                    value_name="agg",
                                    var_name=self.region_colname).rename(
                columns={"index": self.date_colname}).set_index(
                [self.region_colname, self.date_colname])

        # prediction over the prediction size
        predictions = pd.DataFrame()

        try:
            key = "sir"
            used_models = [key]
            predictions = output[key].copy()
            for key_, pred in output.items():
                if key_ != key:
                    predictions[key_] = predictions.join(pred.copy(), how='left')[key_]
                    used_models.append(key_)
        except IndexError:
            return predictions

        return predictions.reset_index()

    def eval_dates(self, start_date, end_date):
        val = self.val_predictions.reset_index()
        val = val[val.date.between(start_date, end_date)]

        error = val.copy()
        for name in self.used_models:
            error[name] = self.metric(error[name].values, error[self.target_colname].values)
        error.drop([self.target_colname, self.date_colname], axis=1, inplace=True)

        error_per_state = error.groupby(self.region_colname).mean()
        error_last = error.dropna().groupby(self.region_colname).last()
        return error_per_state, error_last, val


if __name__ == "__main__":
    # df_path = r'C:\Users\david\Dropbox (MIT)\COVID-19-Team2\Data\07_22_2020_counties_combined.csv'
    df_path = r'C:\Users\david\Dropbox (MIT)\COVID-19-Team2\Data\07_16_2020_states_combined.csv'
    import warnings

    warnings.filterwarnings("ignore")

    # %% Load Data
    # df_train = pd.read_csv(df_path, parse_dates=["date"])
    df_train, _, _ = load_data(
        file=r"C:\Users\david\Dropbox (MIT)\COVID-19-Team2\Data\07_16_2020_states_combined_w_pct.csv",
        training_cutoff='2020-06-15',
        validation_cutoff=None,  # '2020-07-15',
        )

    benchmark = BenchMarkSettings(
        region_colname="state",
        target_colname="cases",
        date_colname="date",
        training_cutoff="20200510",
        prediction_size=5,
        useful_columns=["cases", "deaths"],  # , "stayathome"],
        metric=_mape,
        verbose=0,
    )

    # # simple machine learning model
    # sml_models = SMLBenchMarkModel(benchmark)
    # sml_models.fit(df_train)
    # sml_error = sml_models.eval()
    # sml_error_2w = sml_models.eval_dates("20200721", "20200804")
    # # sml_predictions = sml_models.predict()

    # # LSTM model
    # lstm_model = LSTMBenchMarkModel(benchmark)
    # lstm_model.fit(df_train)
    # lstm_error = lstm_model.eval()
    # lstm_predictions = lstm_model.predict()

    # LSTM model
    model_path_dict = {
        "mdp": r"C:\Users\david\Desktop\MIT\Courses\Research internship\master_branch\covid19_team2\code\models\mdp_20200510_cases_state.pickle",
        "sir": r"C:\Users\david\Desktop\MIT\Courses\Research internship\master_branch\covid19_team2\code\models\sir_20200510_cases_state.pickle",
        "knn": r"C:\Users\david\Desktop\MIT\Courses\Research internship\master_branch\covid19_team2\code\models\knn_20200510_cases_state.pickle",
        "agg": r"C:\Users\david\Desktop\MIT\Courses\Research internship\master_branch\covid19_team2\code\models\agg_20200510_cases_state.pickle",
        "ci": r"C:\Users\david\Desktop\MIT\Courses\Research internship\master_branch\covid19_team2\code\models\ci_20200510_cases_state.pickle",
        }

    load_model_dict = {
        "mdp": True,
        "sir": True,
        "knn": True,
        "agg": True,
        "ci": True,
        }

    ksma_model = KSMABenchMarkModel(benchmark,
                                    models_path_dict=model_path_dict,
                                    load_model_dict=load_model_dict)
    ksma_model.fit(df_train)
    ksma_error = ksma_model.eval()
    ksma_predictions = ksma_model.eval_dates("20200702", "20200709")

    print(ksma_error)
