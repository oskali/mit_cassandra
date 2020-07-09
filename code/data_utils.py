# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 21:24:44 2020

@author: omars
"""

#%% Libraries

import pickle
from datetime import datetime
import pandas as pd
import numpy as np
from params import (target_col, date_col, region_col, training_cutoff, df_path, nmin)

#%% Helper Functions

def save_model(model,
               filename):
    file_pi = open(filename, 'wb')
    pickle.dump(model, file_pi)

def load_model(filename):
    filehandler = open(filename, 'rb')
    return(pickle.load(filehandler))

def load_data(file=df_path,
              target=target_col,
              date=date_col,
              region=region_col,
              training_cutoff=training_cutoff,
              validation_cutoff=None,
              nmim=nmin):
    df = pd.read_csv(file)
    df.columns = map(str.lower, df.columns)
    df= df[df[target] >= nmin]
    df[date] = df[date].apply(lambda x: datetime.strptime(x,'%Y-%m-%d'))
    df = df.sort_values(by = [region, date])
    df_train = df[df[date] <= training_cutoff]
    if validation_cutoff is None:
        df_test = df[df[date] > training_cutoff]
    else:
        df_test = df[[a and b for a, b in zip(df[date] > training_cutoff, df[date] <= validation_cutoff)]]
    return(df, df_train, df_test)

def dict_to_df(output,
               df_validation,
               region_col=region_col,
               date_col=date_col,
               target_col=target_col):
    models = list(output.keys())
    regions = list(set(df_validation[region_col]))
    dates = list(set(df_validation[date_col]))
    predictions_rows = []
    for region in regions:
        for date in dates:
            prediction = [region, date]
            for model in models:
                if region in output[model].keys():
                    prediction.append(output[model][region].loc[date])
                else:
                    prediction.append(np.nan)
            predictions_rows.append(prediction)
    df_predictions = pd.DataFrame(predictions_rows, columns=[region_col,date_col] + models)
    df_agg = df_predictions.merge(df_validation.loc[:, [region_col, date_col, target_col]], how='left', on=[region_col, date_col])
    return df_agg.dropna()

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))

def get_mapes(df,
              models,
              region_col='state',
              target_col='cases'):

    results = []
    for region in set(df[region_col]):
        df_sub = df[df[region_col] == region]
        results.append([region] + [mape(df_sub[target_col], df_sub[model]) for model in models])
    results.append(['Average'] + [mape(df[target_col], df[model]) for model in models])
    return(pd.DataFrame(results, columns=[region_col] + ['MAPE_' + model for model in models]))