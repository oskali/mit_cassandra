# -*- coding: utf-8 -*-
"""
Created on Thu May 28 20:29:42 2020

@author: omars
"""
#############################################################################
#%%
# Load Libraries
import numpy as np
import pandas as pd
import datetime
import warnings
from copy import deepcopy
from mdp_model import MDP_model
from tqdm import tqdm
from utils import (sir_fit_predict, knn_fit_predict, mean_absolute_percentage_error, alpha, wrapper, cols_to_keep)
from scipy.stats import norm
warnings.filterwarnings("ignore")
#############################################################################

#############################################################################
#%%
path = 'C:/Users/omars/Desktop/covid19_georgia/large_data/input/'
file = '05_27_states_combined.csv'
training_cutoff = '2020-04-30'
nmin = 20
deterministic = True
if deterministic:
	deterministic_label = ''
else:
	deterministic_label = 'markov_'
run_mdp = True
sgm = .1
n_iter_mdp = 50
n_iter_ci = 10
ci_range = 0.75
#############################################################################

#############################################################################
#%%
df_orig = pd.read_csv(path + file)
#############################################################################

#############################################################################
print('Data Wrangling in Progress...')
df = deepcopy(df_orig)
df.columns = map(str.lower, df.columns)
df = df.query('cases >= @nmin')
df['date'] = df['date'].apply(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d'))
df = df.sort_values(by = ['state','date'])
states = sorted(list(set(df['state'])))
pop_df = df.loc[:, ['state', 'population']]
pop_dic = {pop_df .iloc[i, 0] : pop_df .iloc[i, 1] for i in range(pop_df .shape[0])}

df_train = df[df['date'] <= training_cutoff]
df_test = df[df['date'] > training_cutoff]

pred_out = len(set(df_test.date))
day_0 = str(df_test.date.min())[:10]
print('Data Wrangling Complete.')
#############################################################################

#############################################################################
#%%
print('SIR Model Training in Progress...')
sir_output = sir_fit_predict(df_train, pop_dic, pred_out, nmin)
sir_output = sir_output.rename(
    columns={'prediction':'sir_prediction'}).loc[:, ['state','date', 'sir_prediction']]

df = df.merge(sir_output, how='left', on=['state', 'date'])
print('SIR Model Complete.')
#############################################################################

#############################################################################
#%%
print('kNN Model Training in Progress...')
knn_output, _ = knn_fit_predict(
    df=df, memory=7, forward_days=pred_out, split_date =training_cutoff,
    day_0 = day_0, real_GR=True, deterministic=deterministic)

knn_output = knn_output.rename(
    columns={'pred_cases':'knn_prediction'}).loc[:, ['state','date', 'knn_prediction']]
knn_output['date'] = knn_output['date'].apply(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d'))

df = df.merge(knn_output, how='left', on=['state', 'date'])
df.knn_prediction = np.where([a and b for a, b in zip(df.knn_prediction.isnull(), df.date <= training_cutoff)], df.cases, df.knn_prediction)
print('kNN Model Complete.')
#############################################################################

#############################################################################
#%%
if run_mdp:
    print('MDP Model Training in Progress...')
    df_train = df_orig[df_orig['date'] <= training_cutoff].drop(columns='people_tested').dropna(axis=0)

    mdp = MDP_model()
    mdp.fit(df_train,
            h=5,
            n_iter=n_iter_mdp,
            d_avg=3,
            distance_threshold = 0.1)

    mdp_output = pd.DataFrame()
    for i in range(pred_out):
        mdp_output = mdp_output.append(mdp.predict_all(n_days=i))

    mdp_output = mdp_output.rename(columns={'TIME': 'date', 'cases':'mdp_prediction'}).loc[:, ['state','date', 'mdp_prediction']]

    df = df.merge(mdp_output, how='left', on=['state', 'date'])
    df.mdp_prediction = np.where([a and b for a, b in zip(df.mdp_prediction.isnull(), df.date <= training_cutoff)], df.cases, df.mdp_prediction)
    print('MDP Model Complete.')
else:
    print('MDP Model Skipped.')
#############################################################################

#############################################################################
#%%
print('Training Weights for the Aggregated Model...')
df_test = df[df['date'] > training_cutoff].dropna()
states_test = set(df_test.state)

sir_mape, knn_mape, mdp_mape = {}, {}, {}
for state in states_test:
    sub = df_test.query('state == @state')
    sir_mape[state] = mean_absolute_percentage_error(sub.cases, sub.sir_prediction)
    knn_mape[state] = mean_absolute_percentage_error(sub.cases, sub.knn_prediction)
    if run_mdp:
        mdp_mape[state] = mean_absolute_percentage_error(sub.cases, sub.mdp_prediction)

if run_mdp:
    weights = {state: np.array([(1/sir_mape[state]), (1/knn_mape[state]), (1/mdp_mape[state])])/((1/sir_mape[state])+(1/knn_mape[state])+(1/mdp_mape[state])) for state in states_test}
else:
    weights = {state: np.array([(1/sir_mape[state]), (1/knn_mape[state])])/((1/sir_mape[state])+(1/knn_mape[state])) for state in states_test}

df = df.reset_index()

if run_mdp:
    df['agg_prediction'] = [weights[df.state[i]][0]*df.sir_prediction[i] + weights[df.state[i]][1]*df.knn_prediction[i] + weights[df.state[i]][2]*df.mdp_prediction[i] if df.state[i] in weights.keys() else df.knn_prediction[i] for i in range(len(df))]
else:
     df['agg_prediction'] = [weights[df.state[i]][0]*df.sir_prediction[i] + weights[df.state[i]][1]*df.knn_prediction[i] if df.state[i] in weights.keys() else df.knn_prediction[i] for i in range(len(df))]

print('Aggregated Model Complete.')
#############################################################################

#############################################################################
#%%
print('Computing Prevalence...')
df['daily_prevalence'] = [(cases/tests)*population*alpha(tests/population) if new_tests != 0 else np.nan for cases, tests, population, new_tests in zip((df['cases'] - df.groupby(['state'])['cases'].shift(1)), df['people_tested'], df['population'], (df['people_tested'] - df.groupby(['state'])['people_tested'].shift(1)))]
df['prevalence'] = np.where(df.daily_prevalence.isnull(), df.cases - df.groupby(['state'])['cases'].shift(1), df.daily_prevalence)
df['prevalence'] = df.groupby(['state'])['prevalence'].cumsum()
# df = df.sort_values(['state', 'date'])
# df['prevalence'] = [(float(cases)/tests)*population*alpha(tests/population, p=0, q=1) for cases, tests, population in zip(df['cases'], df['people_tested'], df['population'])]

# del finalDf['prevalence']
# finalDf = finalDf.merge(df.loc[:, ['state', 'date', 'prevalence']], how='left', on=['state', 'date'])
# finalDf.to_csv('C:/Users/omars/Desktop/df_bis.csv')
print('Prevalence Computed.')
#############################################################################

#############################################################################
#%%
print('Exporting Results...')
df.to_csv('C:/Users/omars/Desktop/df.csv')
finalDf = deepcopy(df)
print('Results Saved.')
#############################################################################

#############################################################################
#%%
print('Computing Confidence Intervals... (1/2)')
df = deepcopy(df_orig)
df.columns = map(str.lower, df.columns)
df = df.query('cases >= @nmin')
df.groupby(['state', 'date']).first().reset_index()
df['date'] = df['date'].apply(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d'))
df = df.sort_values(by = ['state','date']).reset_index()
n = df.shape[0]
states = sorted(list(set(df['state'])))
pop_df = df.loc[:, ['state', 'population']]
pop_dic = {pop_df .iloc[i, 0] : pop_df .iloc[i, 1] for i in range(pop_df .shape[0])}

sirl, knnl, mdpl, aggl = {i: [] for i in range(n)}, {i: [] for i in range(n)}, {i: [] for i in range(n)}, {i: [] for i in range(n)}
for _ in tqdm(range(n_iter_ci)):
    df['new_cases'] = (df['cases'] - df.groupby(['state'])['cases'].shift(1)).apply(lambda x: (1+ 0.1*np.random.randn())*x)
    df['new_cases'] = np.where(df.new_cases.isnull(), df.cases, df.new_cases)
    df['cases'] = df.groupby('state')['new_cases'].cumsum()

    sirp, knnp, mdpp, aggp = wrapper(df_orig,
                                     df,
                                     training_cutoff,
                                     pop_dic,
                                     weights,
                                     nmin=nmin,
                                     n_iter_mdp=n_iter_mdp,
                                     deterministic=deterministic,
                                     run_mdp=run_mdp)

    for i in range(n):
        sirl[i].append(list(sirp)[i])
        knnl[i].append(list(knnp)[i])
        mdpl[i].append(list(mdpp)[i])
        aggl[i].append(list(aggp)[i])



df['sir_ci'] = [norm.interval(ci_range, loc=np.nanmean(sirl[i]), scale=np.nanstd(sirl[i])) for i in range(n)]
df['knn_ci'] = [norm.interval(ci_range, loc=np.nanmean(knnl[i]), scale=np.nanstd(knnl[i])) for i in range(n)]
df['mdp_ci'] = [norm.interval(ci_range, loc=np.nanmean(mdpl[i]), scale=np.nanstd(mdpl[i])) for i in range(n)]
df['agg_ci'] = [norm.interval(ci_range, loc=np.nanmean(aggl[i]), scale=np.nanstd(aggl[i])) for i in range(n)]

finalDf = finalDf.merge(df.loc[:, ['state',
                         'date',
                         'sir_ci',
                         'knn_ci',
                         'mdp_ci',
                         'agg_ci']] ,how='left', on=['state', 'date'])

for model_str in ['sir', 'knn', 'mdp', 'agg']:
    finalDf[model_str + '_lower'] = finalDf[model_str + '_ci'].apply(lambda x: x[0])
    finalDf[model_str + '_upper'] = finalDf[model_str + '_ci'].apply(lambda x: x[1])


finalDf = finalDf.loc[:, cols_to_keep]
print('Confidence Intervals Computed. (1/2)')
#############################################################################

#############################################################################
#%%
print('Computing Confidence Intervals... (2/2)')
df = deepcopy(finalDf)
states = sorted(list(set(df['state'])))
df_test = df.query('date > @training_cutoff')
models = ['sir', 'knn', 'mdp', 'agg']
for model in models:
    finalDf[model + '_per_residuals'] = (finalDf['cases'] - finalDf[model + '_prediction'])/(finalDf[model+ '_prediction'])
    globals()[model + '_grouped'] = finalDf.groupby('state').agg({model+ '_per_residuals': ['mean', 'std']})

dicGrouped = {(model, state): norm.interval(ci_range, loc=globals()[model + '_grouped'].loc[state,:].iloc[0], scale=globals()[model + '_grouped'].loc[state,:].iloc[1]) for model in models for state in states}

for model in models:
    finalDf[model + '_lower'] = [(1+dicGrouped[(model, state)][0])*prediction for state, prediction in zip(finalDf['state'], finalDf[model + '_prediction'])]
    finalDf[model + '_upper'] = [(1+dicGrouped[(model, state)][1])*prediction for state, prediction in zip(finalDf['state'], finalDf[model + '_prediction'])]


finalDf = finalDf.loc[:, cols_to_keep]
finalDf = finalDf.sort_values(['state', 'date'])
print('Confidence Intervals Computed. (2/2)')
#############################################################################




