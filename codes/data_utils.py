# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 21:24:44 2020

@author: omars
"""

#%% Libraries

import os
import pickle
import json
from datetime import datetime
import pandas as pd
from codes.params import (target_col, date_col, region_col, training_cutoff, df_path, nmin)
# from codes.mdp_model import MDPModel

#%% Helper Functions


def save_model_json(model, filename):
    dir_path = os.path.dirname(filename)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    with open(filename, 'wb') as file_pi:
        json.dump(model, file_pi)


def save_model(model, filename):
    dir_path = os.path.dirname(filename)
    if not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path)
        except FileNotFoundError:
            pass
    file_pi = open(filename, 'wb')
    pickle.dump(model, file_pi)


def load_model(filename):
    filehandler = open(filename, 'rb')
    return(pickle.load(filehandler))


# def load_data(file=df_path,
#               target=target_col,
#               date=date_col,
#               region=region_col,
#               training_cutoff=training_cutoff,
#               nmim=nmin):
#     df = pd.read_csv(file)
#     df.columns = map(str.lower, df.columns)
#     df= df[df[target] >= nmin]
#     df[date] = df[date].apply(lambda x: datetime.strptime(x,'%Y-%m-%d'))
#     df = df.sort_values(by = [region, date])
#     df_train = df[df[date] <= training_cutoff]
#     df_test = df[df[date] > training_cutoff]
#     return df, df_train, df_test

def load_data(file=df_path,
              target=target_col,
              date=date_col,
              region=region_col,
              training_cutoff=training_cutoff,
              validation_cutoff=None,
              nmin=nmin):
    df = pd.read_csv(file)
    df.columns = map(str.lower, df.columns)
    df = df[df[target] >= nmin]
    df[date] = df[date].apply(lambda x: datetime.strptime(x,'%Y-%m-%d'))
    df = df.sort_values(by=[region, date])
    df_train = df[df[date] <= training_cutoff]
    if validation_cutoff is None:
        df_test = df[df[date] > training_cutoff]
    else:
        df_test = df[[a and b for a, b in zip(df[date] > training_cutoff, df[date] <= validation_cutoff)]]
    return(df, df_train, df_test)
