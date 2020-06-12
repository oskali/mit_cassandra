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
import json
import os
from copy import deepcopy
from tqdm import tqdm
from utils import (sir_fit_predict, mean_absolute_percentage_error, alpha, wrapper, cols_to_keep)
from scipy.stats import norm
warnings.filterwarnings("ignore")
#############################################################################

#############################################################################
#%%
path = '/Users/bpeshlov/dev/covid19/Code/code/gperakis_prevalence/'
os.chdir(path)
file = '05_27_states_combined.csv'
file_output = "dummy_test.csv"
training_cutoff = '2020-04-30'
nmin = 20 #minimum number of cases where training set starts
n_samples = 30 #random samples from predictions distribution
random_seed = 632 #used for sampling predictions
deterministic = True
if deterministic:
    deterministic_label = ''
else:
    deterministic_label = 'markov_'
run_mdp = False
run_knn = False
run_sir = True
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
#Added params to outputs; beta = params[0], gamma = params[1]
sir_output, params = sir_fit_predict(df_train, pop_dic, pred_out, nmin)
sir_output = sir_output.rename(
    columns={'prediction':'sir_prediction'}).loc[:, ['state','date', 'sir_prediction']]

df = df.merge(sir_output, how='left', on=['state', 'date'])
print('SIR Model Complete.')
#############################################################################

#############################################################################
#%%
print('Training Weights for the Aggregated Model...')
df_test = df[df['date'] > training_cutoff].dropna()
states_test = set(df_test.state)

sir_mape, knn_mape, mdp_mape = {}, {}, {}
for state in states_test:
    sub = df_test.query('state == @state')
    if run_sir:
        sir_mape[state] = mean_absolute_percentage_error(sub.cases, sub.sir_prediction)
    if run_knn:
        knn_mape[state] = mean_absolute_percentage_error(sub.cases, sub.knn_prediction)
    if run_mdp:
        mdp_mape[state] = mean_absolute_percentage_error(sub.cases, sub.mdp_prediction)

df = df.reset_index()

df['agg_prediction'] = df.sir_prediction

print('Aggregated Model Complete.')
#############################################################################

#############################################################################
#%%
print('Computing Prevalence...')
df['daily_prevalence'] = [(cases/tests)*population*alpha(tests/population) if new_tests != 0 else np.nan for cases, tests, population, new_tests in zip((df['cases'] - df.groupby(['state'])['cases'].shift(1)), df['people_tested'], df['population'], (df['people_tested'] - df.groupby(['state'])['people_tested'].shift(1)))]
df['prevalence'] = np.where(df.daily_prevalence.isnull(), df.cases - df.groupby(['state'])['cases'].shift(1), df.daily_prevalence)
df['prevalence'] = df.groupby(['state'])['prevalence'].cumsum()
print('Prevalence Computed.')
#############################################################################

#############################################################################
#%%
print('Exporting Results...')
df.to_csv(path + file_output)
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

sirl, aggl = {i: [] for i in range(n)},  {i: [] for i in range(n)}
for _ in tqdm(range(n_iter_ci)):
    df['new_cases'] = (df['cases'] - df.groupby(['state'])['cases'].shift(1)).apply(lambda x: (1+ 0.1*np.random.randn())*x)
    df['new_cases'] = np.where(df.new_cases.isnull(), df.cases, df.new_cases)
    df['cases'] = df.groupby('state')['new_cases'].cumsum()
    sirp, aggp = wrapper(df_orig,
                        df,
                        training_cutoff,
                        pop_dic,
                        1,
                        nmin=nmin,
                        n_iter_mdp=n_iter_mdp,
                        deterministic=deterministic,
                        run_mdp=run_mdp)
    for i in range(n):
        sirl[i].append(list(sirp)[i])
        aggl[i].append(list(aggp)[i])



df['sir_ci'] = [norm.interval(ci_range, loc=np.nanmean(sirl[i]), scale=np.nanstd(sirl[i])) for i in range(n)]
df['agg_ci'] = [norm.interval(ci_range, loc=np.nanmean(aggl[i]), scale=np.nanstd(aggl[i])) for i in range(n)]

finalDf = finalDf.merge(df.loc[:, ['state',
                         'date',
                         'sir_ci',
                         'agg_ci']] ,how='left', on=['state', 'date'])

for model_str in ['sir', 'agg']:
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
models = ['sir', 'agg']
for model in models:
    finalDf[model + '_per_residuals'] = (finalDf['cases'] - finalDf[model + '_prediction'])/(finalDf[model+ '_prediction'])
    globals()[model + '_grouped'] = finalDf.groupby('state').agg({model+ '_per_residuals': ['mean', 'std']})

dicGrouped = {(model, state): norm.interval(ci_range, loc=globals()[model + '_grouped'].loc[state,:].iloc[0], scale=globals()[model + '_grouped'].loc[state,:].iloc[1]) for model in models for state in states}

for model in models:
    finalDf[model + '_lower'] = [(1+dicGrouped[(model, state)][0])*prediction for state, prediction in zip(finalDf['state'], finalDf[model + '_prediction'])]
    finalDf[model + '_upper'] = [(1+dicGrouped[(model, state)][1])*prediction for state, prediction in zip(finalDf['state'], finalDf[model + '_prediction'])]


finalDf = finalDf.loc[:, cols_to_keep]
finalDf = finalDf.sort_values(['state', 'date'])
finalDf.to_csv(path + 'sir_final_df.csv')
print('Confidence Intervals Computed. (2/2)')
#############################################################################

#############################################################################
print('Saving metadata.json')
beta = params[0]#learned SIR parameters
gamma = params[1]
#training_cutoff = '2020-04-30'
#nmin = 100
uuid = "1591155120-state-prevalence-model-2020-06-11" #placeholder UPDATE
git_hash = "fa69f4587fe41094a74c1865857a31bcf0d1d2bf" #placeholder UPDATE

new_meta = {}
new_meta['model_type'] = 'prevalance'
new_meta['model_name'] = 'state_prevalance_sir_model'
new_meta['model_id'] = 'V0.01'
new_meta['model_pyfile'] = 'main.py'
new_meta['model_pyclass'] = ''
new_meta['input_models'] = []
new_meta['model_parameters'] = {
    "random_seed": random_seed,
    "sir_beta": beta,
    "sir_gamma": gamma,
    "sir_min_cases": nmin
}
new_meta['t0_date'] = training_cutoff
new_meta['n_samples'] = n_samples
new_meta['uuid'] = uuid
new_meta['git_hash'] = git_hash

with open(path + 'prevalence_output_metadata.json', 'w') as fp:
    json.dump(new_meta, fp)

print('Saved {}prevalence_output_metadata.json'.format(path))
################################################################################

################################################################################
print('Saving samples.json')
#Format final df
state = 'Massachusetts'#output only MA predictions
dff = finalDf.query('state == @state').query('date > @training_cutoff')
dff = dff[['date','state','agg_prediction','agg_lower','agg_upper']].dropna()
dff['date'] = dff['date'].apply(lambda x: datetime.datetime.strftime(x,'%Y-%m-%d'))

np.random.seed(random_seed)
samples = dict(dates=dff['date'].to_list(), samples=None)
lo = dff['agg_lower'].to_list()#lower bound of predictions CIs
up = dff['agg_upper'].to_list()
#Random normal distributions of predictions
pred_dist =[]
for lb,ub in zip(lo,up):
    mu = lb + (ub-lb)/2 #middle/mean of this CI
    sigma = (ub - lb)/6.02 #StDev (6*StDev =99.7% of Norm)
    pred_dist.append(np.random.normal(mu, sigma, n_samples).tolist())

all_samples=[]
for samp in range(n_samples):
    single_sample = []
    for t_i in range(len(lo)):
        single_sample.append({'MA': pred_dist[t_i][samp]})
    all_samples.append(single_sample)
samples['samples'] = all_samples

with open(path + 'prevalence_output_samples.json', 'w') as fp:
    json.dump(samples, fp)

print('Saved {}prevalence_output_samples.json'.format(path))



