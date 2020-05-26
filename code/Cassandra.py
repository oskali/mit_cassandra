import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

from support_knn_time_series_covid import *

address = 'covid_preds/'

state_df = pd.read_csv(address + '05_19_states_cases_measures_mobility.csv') #dataset from Boyan

state_df = state_df.drop(columns = ['Unnamed: 0'])
state_df = state_df.rename(columns = {'date':'Date'})

state_df = state_df[['state','Date','cases']].groupby(['state', 'Date']).sum().reset_index()
state_df = state_df.sort_values(by = ['state','Date'])

forward_days = 43 #how many days forward you wanna predict for
split_date = '2020-05-18' #split date between train and test ( first day of test)
prediction_day = '2020-05-18' #1st day you wanna run predictions from. Will get predictions from this day to #forward days forward. MIght me before the split day.


df_simple, df_with_growth_rates = predict_covid(df = state_df, memory = 7, forward_days = forward_days, split_date = split_date, prediction_day = prediction_day, real_GR = True)
#the df_simple  has only information about the predicted cases
df_simple.to_csv(address + 'predicted{}_cases_for_{}_split_{}.csv'.format(forward_days, prediction_day, split_date), index = False)
#the df_with_growth_rates has only information about the Growth rate, predicted cases, predicted accumulative growthrates, actual accumulative growthrates etc.
# df_with_growth_rates.to_csv(address + 'predicted{}_GR_and_cases_for_{}_split_{}.csv'.format(forward_days, prediction_day, split_date), index = False)
