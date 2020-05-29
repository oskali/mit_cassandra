# -*- coding: utf-8 -*-
"""
Created on Thu May 28 20:39:35 2020

@author: omars
"""
#############################################################################
# Load Libraries
import pandas as pd
import sir_ode
import sir_cost
import datetime
from copy import deepcopy
from scipy.integrate import odeint as ode
import scipy.optimize as optimize
import numpy as np
from mdp_model import MDP_model
from knn_utils import (get_best_parameters, match_to_real_growth, transpose_case_df)
#############################################################################


#############################################################################
# Helper variables
models = ['sir', 'knn', 'mdp', 'agg']
metrics = ['prediction', 'lower', 'upper']
cols_to_keep = ['state',
                'date',
                'cases',
                'prevalence'] + [model + '_' + metric for model in models for metric in metrics]

# Fit Predict fucntions for SIR and KNN
def sir_fit_predict(df_train, #dataset has columns state, date, cases
                    pop_dic,
                    pred_out=100,
                    nmin=100):
    dataset = deepcopy(df_train)
    dataset.set_index(['date'])
    output = pd.DataFrame()
    states = dataset.state.unique()
    for i in range(len(states)):
        state = states[i]
        train_set = dataset.query('state == @state').query('cases >= @nmin')
        if train_set.shape[0] > 10:
            timer = [j for j in range(len(train_set))]
            actual_time = [k for k in range(len(train_set) + pred_out)]
            data = train_set.loc[:, 'cases'].values
            times = timer
            params = [0.4, 0.06, pop_dic[state]]
            #paramnames = ['beta', 'gamma', 'k']
            ini = sir_ode.x0fcn(params,data)

    	    #Simulate and plot the model
            res = ode(sir_ode.model, ini, times, args=(params,))
            #sim_measure = sir_ode.yfcn(res, params)

    	    #Parameter estimation
            optimizer = optimize.minimize(sir_cost.NLL, params, args=(data,times), method='Nelder-Mead')
            paramests = np.abs(optimizer.x)
            iniests = sir_ode.x0fcn(paramests, data)
            xest = ode(sir_ode.model, iniests, times, args=(paramests,))
            est_measure = sir_ode.yfcn(xest, paramests)

            est_measure = np.array(est_measure)

            params = paramests
            ini1 = sir_ode.x0fcn(params,data)

    	    #Simulate and plot the model
            res = ode(sir_ode.model, ini1, actual_time, args=(params,))

            preds_test = sir_ode.yfcn(res, params)
            preds_test = np.delete(preds_test,times)
            pred_train = pd.DataFrame(est_measure)
            pred_test = pd.DataFrame(preds_test)
            df_fin = pred_train.append(pred_test, ignore_index = True)
            df_fin.columns = ["prediction"]

            last_date = train_set.date.tail(1)
            date_list = [last_date + datetime.timedelta(days=x) for x in range(pred_out)]
            df_fin["date"] = train_set.date.append(date_list, ignore_index = True)

            df_fin["state"] = states[i]
            output = output.append(df_fin, ignore_index = True) #df5 stores the output for all states together
    return output

def knn_fit_predict(df,
                    memory=7,
                    forward_days=7,
                    split_date='2020-05-01',
                    day_0='2020-05-01',
                    real_GR=False,
                    deterministic=True):
    '''
    everything before split_date is train

    '''
    #This section of code creates the forward and back features

    df0 = df.copy(deep=True) # deep copy might not be needed, just for security

    #remove some states/territories with late or small number of cases
    #CHANGE TO KEEPING STATES - hard copy in this code maybe global variable
    df0 = df0.loc[~df0['state'].isin(['West Virginia','District of Columbia','Puerto Rico','American Samoa', 'Diamond Princess','Grand Princess','Guam','Northern Mariana Islands','Virgin Islands'])]

    df0 = df0.sort_values(by=['state', 'date']) #has to be sorted by days to create growth rates
    df0['GrowthRate'] = (df0.groupby('state')['cases'].shift(0) / df0['cases'].shift(1) - 1) #group by state so that consecutive rows are consecutive days in a single state

    #create the t-1 to t-memory growth rates
    for i in range(memory):
        df0['GrowthRate_t-' + str(i+1)] = df0.groupby('state')['GrowthRate'].shift(i+1)

    df0['cases_t-1'] = df0['cases'].shift(1)

    #this is used only if we are using the alternate method where we run nearest neighbors on predictions in the train set
    if real_GR:
        for i in range(forward_days):
            df0['GrowthRate_t+' + str(i)] = df0.groupby('state')['GrowthRate'].shift(-i)

        for i in range(forward_days):
            df0['actual_growth_for_next_{}days'.format(i+1)] = (df0['cases'].shift(-i)/df0['cases'].shift(1)) - 1
    '''
    threshold: multiplier on the nearest distance that we cut off at when assigning weights, e.g. a point outside the threshold gets a weight of 0
    n: maximum number of nearest neighbors
    p: either L1 norm (manhattan) or L2 norm
    func: get_weights or get_weights_sq, whether the distance norm will be squared or not
    '''
    df0 = df0.dropna()
    threshold, n, p, func = get_best_parameters(df0, memory, split_date)

    #we run the method using the best parameters according to the split date
    predictions = match_to_real_growth(df0, threshold, n, p, func, memory, forward_days, day_0, split_date, deterministic)

    #we have finished producing predictions, and move on to converting predicted growth rates into predicted cases

    #convert growth rates to cumulative growth rates -- here we need to add 1 to each predicted growth rate so that when multiplied they represent growth rate over multiple days
    #the cumulative growth rate over n days starting today = (1+ GR_0) * (1+GR_1) * ... * (1+ GR_n-1)
    predictions['pred_growth_for_next_1days'] = predictions['pred_forward_day_0'] + 1
    predictions['pred_high_growth_for_next_1days'] = predictions['pred_high_day_0'] + 1
    predictions['pred_low_growth_for_next_1days'] = predictions['pred_low_day_0'] + 1

    for i in range(1,forward_days):
        predictions['pred_growth_for_next_{}days'.format(i+1)] = predictions['pred_growth_for_next_{}days'.format(i)]*(predictions['pred_forward_day_'+ str(i)] + 1)
        predictions['pred_high_growth_for_next_{}days'.format(i+1)] = predictions['pred_growth_for_next_{}days'.format(i)]*(predictions['pred_high_day_'+ str(i)] + 1)
        predictions['pred_low_growth_for_next_{}days'.format(i+1)] = predictions['pred_growth_for_next_{}days'.format(i)]*(predictions['pred_low_day_'+ str(i)] + 1)
    for i in range(forward_days):
        predictions['pred_growth_for_next_{}days'.format(i+1)] = predictions['pred_growth_for_next_{}days'.format(i+1)] - 1
        predictions['pred_high_growth_for_next_{}days'.format(i+1)] = predictions['pred_high_growth_for_next_{}days'.format(i+1)] - 1
        predictions['pred_low_growth_for_next_{}days'.format(i+1)] = predictions['pred_low_growth_for_next_{}days'.format(i+1)] - 1

    #convert cumulative growth rates to cases
    for i in range(forward_days):
        predictions['cases_predicted_day_' + str(i)] = np.round(predictions['cases_t-1']*(predictions['pred_growth_for_next_{}days'.format(i+1)]+1))
        predictions['cases_high_predicted_day_' + str(i)] = np.round(predictions['cases_t-1']*(predictions['pred_high_growth_for_next_{}days'.format(i+1)]+1))
        predictions['cases_low_predicted_day_' + str(i)] = np.round(predictions['cases_t-1']*(predictions['pred_low_growth_for_next_{}days'.format(i+1)]+1))

    columns_to_keep = ['state', 'date', 'cases'] + ['cases_predicted_day_' + str(i) for i in range(forward_days)] + ['cases_low_predicted_day_' + str(i) for i in range(forward_days)] + ['cases_high_predicted_day_' + str(i) for i in range(forward_days)]
    simple_output = predictions[columns_to_keep]

    #transpose simple output to have forward_days*50 rows
    transposed_simple_output = transpose_case_df(simple_output, forward_days, day_0)


    return transposed_simple_output, predictions
#############################################################################


#############################################################################
# Helper Functions
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))

def alpha(percentage_tested, p=0.1, q=0.2):
    beta1 = (1-q)/(1-p)
    beta0 = 1 - beta1
    return((beta1 * percentage_tested) + beta0)
#############################################################################


#############################################################################
# Wrapper
def wrapper(df_orig,
            df_input,
            training_cutoff,
            pop_dic,
            weights,
            nmin=20,
            n_iter_mdp=20,
            deterministic=True,
            run_mdp=False):

    df = deepcopy(df_input)
    df_train = df[df['date'] <= training_cutoff]
    df_test = df[df['date'] > training_cutoff]

    pred_out = len(set(df_test.date))
    day_0 = str(df_test.date.min())[:10]
    ###
    sir_output = sir_fit_predict(df_train, pop_dic, pred_out, nmin)
    sir_output = sir_output.rename(
        columns={'prediction':'sir_prediction'}).loc[:, ['state','date', 'sir_prediction']]

    df = df.merge(sir_output, how='left', on=['state', 'date'])
    ###

    ###

    knn_output, _ = knn_fit_predict(
        df=df, memory=7, forward_days=pred_out, split_date =training_cutoff,
        day_0 = day_0, real_GR=True, deterministic=deterministic)

    knn_output = knn_output.rename(
        columns={'pred_cases':'knn_prediction'}).loc[:, ['state','date', 'knn_prediction']]
    knn_output['date'] = knn_output['date'].apply(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d'))

    df = df.merge(knn_output, how='left', on=['state', 'date'])
    df.knn_prediction = np.where([a and b for a, b in zip(df.knn_prediction.isnull(), df.date <= training_cutoff)], df.cases, df.knn_prediction)
    ####

    ###
    if run_mdp:
        df_train = df_orig[df_orig['date'] <= training_cutoff].drop(columns='people_tested').dropna(axis=0)

        mdp = MDP_model()
        mdp.fit(df_train,
                h=5,
                n_iter=n_iter_mdp,
                d_avg=3,
                distance_threshold = 0.05)

        mdp_output = pd.DataFrame()
        for i in range(pred_out):
            mdp_output = mdp_output.append(mdp.predict_all(n_days=i))

        mdp_output = mdp_output.rename(columns={'TIME': 'date', 'cases':'mdp_prediction'}).loc[:, ['state','date', 'mdp_prediction']]

        df = df.merge(mdp_output, how='left', on=['state', 'date'])
        df.mdp_prediction = np.where([a and b for a, b in zip(df.mdp_prediction.isnull(), df.date <= training_cutoff)], df.cases, df.mdp_prediction)
    ###

    ###

    df = df.reset_index()
    if run_mdp:
        df['agg_prediction'] = [weights[df.state[i]][0]*df.sir_prediction[i] + weights[df.state[i]][1]*df.knn_prediction[i] + weights[df.state[i]][2]*df.mdp_prediction[i] if df.state[i] in weights.keys() else df.knn_prediction[i] for i in range(len(df))]
    else:
         df['agg_prediction'] = [weights[df.state[i]][0]*df.sir_prediction[i] + weights[df.state[i]][1]*df.knn_prediction[i] if df.state[i] in weights.keys() else df.knn_prediction[i] for i in range(len(df))]
         df['mdp_prediction'] = 0

    df = df.sort_values(by=['state','date'])

    return (np.array(df['sir_prediction']),
            np.array(df['knn_prediction']),
            np.array(df['mdp_prediction']),
            np.array(df['agg_prediction']))

    # return(df)
#############################################################################
