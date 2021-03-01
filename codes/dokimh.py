import warnings
warnings.filterwarnings("ignore")
import sys
# insert at 1, 0 is the script path (or '' in REPL)
# sys.path.insert(1, 'cassandra/')
#%% Libraries
from datetime import datetime
import numpy as np
import pandas as pd
from knn_model import KNNModel
from knn_utils import wmape, mod_date
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
cwd = os.getcwd()
print(cwd)
address = '../data/01_05_2021_states_combined.csv'
state_df = pd.read_csv(address)

state_df['date'] = pd.to_datetime(state_df['date'])
split_date = '10-01-2020'
regions = ["Alabama","Alaska","Arizona","Arkansas","California","Colorado",
    "Connecticut","Delaware","Florida","Georgia","Hawaii","Idaho","Illinois",
    "Indiana","Iowa","Kansas","Kentucky","Louisiana","Maine","Maryland",
    "Massachusetts","Michigan","Minnesota","Mississippi","Missouri","Montana",
    "Nebraska","Nevada","New Hampshire","New Jersey","New Mexico","New York",
    "North Carolina","North Dakota","Ohio","Oklahoma","Oregon","Pennsylvania",
    "Rhode Island","South Carolina","South Dakota","Tennessee","Texas","Utah",
    "Vermont","Virginia","West Virginia","Washington","Wisconsin","Wyoming"]
state_df = state_df.loc[state_df.state.isin(regions)] # isolating only the region of interest
dates = list(state_df.loc[state_df.date >= split_date].date.unique()) #dates we want to predict (only the max of those is crucial)

state_df = state_df[['state','date','deaths']]



train = state_df.loc[(state_df.date < split_date) ]
test = state_df.loc[(state_df.date >= split_date) ]
cassandra = KNNModel(region='state',target='deaths')
cassandra.fit(df = train, regions= regions)

df_simple = cassandra.predict(regions= regions, dates= dates)

# df_simple['date'] = pd.to_datetime(df_simple['date']) #fixing the date format