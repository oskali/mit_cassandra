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
from utils import (sir_fit_predict, knn_fit_predict, mean_absolute_percentage_error, alpha, wrapper, ml_models, models,
                   metrics)
from scipy.stats import norm
warnings.filterwarnings("ignore")
from twostage_utils import (train_state_models, matrix_agg_predict)
#############################################################################

#############################################################################
#%%
path = 'C:/Users/omars/Desktop/covid19_georgia/large_data/input/'
file = '06_15_2020_states_combined.csv'
training_cutoff = '2020-05-25'
nmin = 20
deterministic = True
if deterministic:
	deterministic_label = ''
else:
	deterministic_label = 'markov_'
run_sir = True
run_knn = True
run_mdp = True
run_scnd = True
target = 'deaths'
mdp_region_col = 'state' # str, col name of region (e.g. 'state')
mdp_date_col = 'date' # str, col name of time (e.g. 'date')
mdp_features_cols = [] # list of strs: feature columns

sgm = .1
n_iter_mdp = 50
n_iter_ci = 10
ci_range = 0.75

cols_to_keep = ['state',
                'date',
                target,
                'prevalence'] + [model + '_' + metric for model in models for metric in metrics]
#############################################################################

#############################################################################
#%%
df_orig = pd.read_csv(path + file)
#############################################################################

#############################################################################
print('Data Wrangling in Progress...')
df = deepcopy(df_orig)
df.columns = map(str.lower, df.columns)
#df = df.query('cases >= @nmin')
df= df[df[target] >= nmin]
df['date'] = df['date'].apply(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d'))
df_orig['date'] = df_orig['date'].apply(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d'))
df = df.sort_values(by = ['state','date'])
states = sorted(list(set(df['state'])))
pop_df = df.loc[:, ['state', 'population']]
pop_dic = {pop_df .iloc[i, 0] : pop_df .iloc[i, 1] for i in range(pop_df .shape[0])}
features = list(df.columns[5:35])


df = df.loc[:, df.columns[df.isnull().sum() * 100 / len(df) < 20]]
features = list(set(features).intersection(set(df.columns)))

df_train = df[df['date'] <= training_cutoff]
df_test = df[df['date'] > training_cutoff]
pred_out = len(set(df_test.date))
day_0 = str(df_test.date.min())[:10]
print('Data Wrangling Complete.')
#############################################################################

#############################################################################
#%%
if run_sir and target == 'cases':
    print('SIR Model Training in Progress...')
    sir_output = sir_fit_predict(df_train, pop_dic, pred_out, nmin)
    sir_output = sir_output.rename(
        columns={'prediction':'sir_prediction'}).loc[:, ['state','date', 'sir_prediction']]

    df = df.merge(sir_output, how='left', on=['state', 'date'])
    print('SIR Model Complete.')
else:
    print('SIR Model Skipped.')
#############################################################################

#############################################################################
#%%
if run_knn:
    print('kNN Model Training in Progress...')
    knn_output, _ = knn_fit_predict(
        df=df, memory=7, forward_days=pred_out, split_date =training_cutoff,
        day_0 = day_0, real_GR=True, deterministic=deterministic,
        target=target)

    knn_output = knn_output.rename(
        columns={'pred_' + target:'knn_prediction'}).loc[:, ['state','date', 'knn_prediction']]
    knn_output['date'] = knn_output['date'].apply(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d'))

    df = df.merge(knn_output, how='left', on=['state', 'date'])
    df.knn_prediction = np.where([a and b for a, b in zip(df.knn_prediction.isnull(), df.date <= training_cutoff)], df[target], df.knn_prediction)
    print('kNN Model Complete.')
else:
    print('kNN Model Skipped.')
#############################################################################

#############################################################################
#%%
if run_mdp:
    print('MDP Model Training in Progress...')
    #df_train = df_orig[df_orig['date'] <= training_cutoff].drop(columns='people_tested').dropna(axis=0)
    df_train = df_orig[df_orig['date'] <= training_cutoff]
    mdp = MDP_model()
    mdp_abort=False
    try:
        mdp.fit(df_train,
                target_col = target, # str: col name of target (i.e. 'deaths')
                region_col = mdp_region_col, # str, col name of region (i.e. 'state')
                date_col = mdp_date_col, # str, col name of time (i.e. 'date')
                features_cols = mdp_features_cols, # list of strs: feature columns
                h=5,
                n_iter=n_iter_mdp,
                d_avg=3,
                distance_threshold = 0.1)
    except:
        print('MDP Model Aborted - check ERROR message.')
        mdp_abort=True
        run_mdp = False
    if not mdp_abort:
        mdp_output = pd.DataFrame()
        for i in range(pred_out):
            mdp_output = mdp_output.append(mdp.predict_all(n_days=i))
    
        mdp_output = mdp_output.rename(columns={'TIME': 'date', target:'mdp_prediction'}).loc[:, ['state','date', 'mdp_prediction']]
    
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
    if run_sir:
        sir_mape[state] = mean_absolute_percentage_error(sub.cases, sub.sir_prediction)
    if run_knn:
        knn_mape[state] = mean_absolute_percentage_error(sub.cases, sub.knn_prediction)
    if run_mdp:
        mdp_mape[state] = mean_absolute_percentage_error(sub.cases, sub.mdp_prediction)

weights = {}
for state in states_test:
    up = np.array([0, 0, 0])
    dn = 0
    if run_sir:
        dn += (1/sir_mape[state])
    if run_knn:
        dn += (1/knn_mape[state])
    if run_mdp:
        dn += (1/mdp_mape[state])

    if run_sir:
        up[0] = (1/sir_mape[state])/dn
    if run_knn:
        up[1] = (1/knn_mape[state])/dn
    if run_mdp:
        up[2] = (1/mdp_mape[state])/dn

    weights[state] = up

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
df = df.merge(df_orig.loc[:, ['state', 'date', 'people_tested']], on=['state','date'])
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
if run_scnd and run_sir:
    predicted = 'sir_prediction'
    true = target
    all_states = sorted(list(set(df['state'])))
    all_models, tst, baseline = train_state_models(df, all_states,
                                   features, true, predicted, ml_models)

    df_train = df[df['date'] <= training_cutoff]
    df_test = df[df['date'] > training_cutoff]
    X_train, y_train, first_stage_train = df_train.loc[:, ['state'] + features], df_train[true] - df_train[predicted], df_train[predicted]
    X_test, y_test, first_stage_test = df_test.loc[:, ['state'] + features], df_test[true] - df_test[predicted], df_test[predicted]
    X_train, y_train, first_stage_train = np.array(X_train), np.array(y_train), np.array(first_stage_train)
    X_test, y_test, first_stage_test = np.array(X_test), np.array(y_test), np.array(first_stage_test)
    predictions_train = matrix_agg_predict(X_train, all_models)
    predictions_test = matrix_agg_predict(X_test, all_models)

    df['twostage_prediction'] = df[predicted] + matrix_agg_predict(np.array(df.loc[:, ['state'] + features]), all_models)
    df_test = df[df['date'] > training_cutoff]
    evl = df_test[df_test['twostage_prediction'].apply(lambda x: not(np.isnan(x)))]
else:
    df['twostage_prediction'] = 0

#############################################################################

#############################################################################
#%%
print('Exporting Results...')
df.to_csv('C:/Users/omars/Desktop/df_' + target + '.csv')
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
models = ['agg']
if run_sir:
    models.append('sir')
if run_knn:
    models.append('knn')
if run_mdp:
    models.append('mdp')
for model in models:
    finalDf[model + '_per_residuals'] = (finalDf[target] - finalDf[model + '_prediction'])/(finalDf[model+ '_prediction'])
    globals()[model + '_grouped'] = finalDf.groupby('state').agg({model+ '_per_residuals': ['mean', 'std']})

dicGrouped = {(model, state): norm.interval(ci_range, loc=globals()[model + '_grouped'].loc[state,:].iloc[0], scale=globals()[model + '_grouped'].loc[state,:].iloc[1]) for model in models for state in states}

for model in models:
    finalDf[model + '_lower'] = [(1+dicGrouped[(model, state)][0])*prediction for state, prediction in zip(finalDf['state'], finalDf[model + '_prediction'])]
    finalDf[model + '_upper'] = [(1+dicGrouped[(model, state)][1])*prediction for state, prediction in zip(finalDf['state'], finalDf[model + '_prediction'])]


#finalDf = finalDf.loc[:, cols_to_keep]
finalDf = finalDf.sort_values(['state', 'date'])
print('Confidence Intervals Computed. (2/2)')
#############################################################################

# df['date'] = df['date'].apply(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d'))
# #%%
# state = 'Wyoming'
# df_sub = df.query('state == @state')
# plt.plot(df_sub.date, df_sub.twostage_prediction, label='Second Stage')
# plt.plot(df_sub.date, df_sub.sir_prediction, label='First Stage')
# plt.plot(df_sub.date, df_sub.cases, label='True')
# plt.legend()


# #%%
# state = 'New York'
# df_sub = df.query('state == @state')
# plt.plot(df_sub.date, df_sub.knn_prediction, label='KNN Predictions for Deaths')
# plt.plot(df_sub.date, df_sub.deaths, label='Detected Deaths')
# plt.axvline(x=datetime.datetime.strptime(training_cutoff,'%Y-%m-%d'),color='red')
# plt.legend()