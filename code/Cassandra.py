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

state_df = pd.read_csv(address + '05_19_states_cases_measures_mobility.csv')

state_df = state_df.drop(columns = ['Unnamed: 0'])
state_df = state_df.rename(columns = {'date':'Date'})

state_df = state_df[['state','Date','cases']].groupby(['state', 'Date']).sum().reset_index()
state_df = state_df.sort_values(by = ['state','Date'])

state_df.to_csv('05_19_states_cases_measures_mobility_formatted.csv', index = False)


forward_days = 14
split_date = '2020-05-05'
day_0 = '2020-05-05'
deterministic = True


if deterministic:
	deterministic_label = ''
else:
	deterministic_label = 'markov_'

df_simple, df_with_growth_rates = predict_covid(df = state_df, memory = 7, forward_days = forward_days, split_date = split_date, day_0 = day_0, real_GR = True, deterministic = deterministic)


df_simple.to_csv(address + deterministic_label + 'predicted{}_cases_for_{}_split_{}.csv'.format(forward_days, day_0, split_date), index = False)
df_with_growth_rates.to_csv(address + deterministic_label + 'predicted{}_GR_and_cases_for_{}_split_{}.csv'.format(forward_days, day_0, split_date), index = False)
