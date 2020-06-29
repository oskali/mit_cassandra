# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 21:24:44 2020

@author: omars
"""

#%% Libraries

import pickle
from datetime import datetime
import pandas as pd
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
              nmim=nmin):
    df = pd.read_csv(file)
    df.columns = map(str.lower, df.columns)
    df= df[df[target] >= nmin]
    df[date] = df[date].apply(lambda x: datetime.strptime(x,'%Y-%m-%d'))
    df = df.sort_values(by = [region, date])
    df_train = df[df[date] <= training_cutoff]
    df_test = df[df[date] > training_cutoff]
    return(df, df_train, df_test)