# -*- coding: utf-8 -*-
"""
Created on Thu May 28 20:39:35 2020

@author: omars
"""
#############################################################################
# Load Libraries
import pandas as pd
import sir_ode_dummy as sir_ode
import sir_cost_dummy as sir_cost
import datetime
from copy import deepcopy
from scipy.integrate import odeint as ode
import scipy.optimize as optimize
import numpy as np
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

    # knn_output, _ = knn_fit_predict(
    #     df=df, memory=7, forward_days=pred_out, split_date =training_cutoff,
    #     day_0 = day_0, real_GR=True, deterministic=deterministic)

    # knn_output = knn_output.rename(
    #     columns={'pred_cases':'knn_prediction'}).loc[:, ['state','date', 'knn_prediction']]
    # knn_output['date'] = knn_output['date'].apply(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d'))

    # df = df.merge(knn_output, how='left', on=['state', 'date'])
    # df.knn_prediction = np.where([a and b for a, b in zip(df.knn_prediction.isnull(), df.date <= training_cutoff)], df.cases, df.knn_prediction)
    # ####

    ###
    # if run_mdp:
    #     df_train = df_orig[df_orig['date'] <= training_cutoff].drop(columns='people_tested').dropna(axis=0)

    #     mdp = MDP_model()
    #     mdp.fit(df_train,
    #             h=5,
    #             n_iter=n_iter_mdp,
    #             d_avg=3,
    #             distance_threshold = 0.05)

    #     mdp_output = pd.DataFrame()
    #     for i in range(pred_out):
    #         mdp_output = mdp_output.append(mdp.predict_all(n_days=i))

    #     mdp_output = mdp_output.rename(columns={'TIME': 'date', 'cases':'mdp_prediction'}).loc[:, ['state','date', 'mdp_prediction']]

    #     df = df.merge(mdp_output, how='left', on=['state', 'date'])
    #     df.mdp_prediction = np.where([a and b for a, b in zip(df.mdp_prediction.isnull(), df.date <= training_cutoff)], df.cases, df.mdp_prediction)
    ###

    ###

    df = df.reset_index()
    df['agg_prediction'] = df.sir_prediction
    
    df = df.sort_values(by=['state','date'])

    return (np.array(df['sir_prediction']),
            # np.array(df['knn_prediction']),
            # np.array(df['mdp_prediction']),
            np.array(df['agg_prediction']))

    # return(df)
#############################################################################
