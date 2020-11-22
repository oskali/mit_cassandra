# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 15:01:53 2020
@author: leann, omars
"""

#%% Libraries

import numpy as np
from scipy.integrate import odeint as ode
import pandas as pd
from copy import deepcopy
import scipy.optimize as optimize
from datetime import timedelta
from sklearn.metrics import mean_squared_error
from math import sqrt
from tqdm import tqdm
import random as rd
import matplotlib.pyplot as plt



#%% Helper Functions



#ode differential equations
def model(ini, time_step, params):
    Y = np.zeros(5) #column vector for the state variables
    X = ini
    beta = params[0]
    gamma = params[1]
    a = params[3]
    mu = params[4]

    S = X[0]
    E = X[1]
    I = X[2]
    R = X[3]
    D = X[4]

    Y[0] = 0 - (beta*S*I)/params[2] #S
    Y[1] = (beta*S*I)/params[2] - a*E #E
    Y[2] = a*E - (gamma + mu)*I #I
    Y[3] = gamma*I #R
    Y[4] = mu*I #D
    return Y

#set up initial compartments
def inifcn(params, cases, deaths, iniest = None):
    if(iniest is None):
        S0 = params[2] - cases[0] - deaths[0]
        E0 = 0.0
        I0 = cases[0]
        R0 = 0.0
        D0 = deaths[0]
        X0 = [S0, E0, I0, R0, D0]
    else:
        diff = sum(iniest) - params[2]
        X0 = [(iniest[0] - diff), iniest[1], iniest[2], iniest[3], iniest[4]]
    return X0

#retrieve cumlative case compartments (active, recovered, dead)
def finfcn(res):
    return res[:,2], res[:,3], res[:,4]

#objective function to optimize hyperparameters
def NLL(params, cases, deaths, recover, death_lm, recover_lm, weight_dead, weight_recover, times, ini): #loss function
    params = np.abs(params)
    cases = np.array(cases)
    deaths = np.array(deaths)
    res = ode(model, inifcn(params,cases,deaths, ini), times, args =(params,))
    active, recovered, dead = finfcn(res)
    nll = sqrt(mean_squared_error(cases,active)) + death_lm*sqrt(mean_squared_error(deaths,dead)) + recover_lm*sqrt(mean_squared_error(recover,recovered))
    return nll

#%% Model

class SIRModel():
        def __init__(self,
                     nmin=100,
                     date='date',
                     region='state',
                     target='cases',
                     population='population',
                     optimizer='Nelder-Mead',
                     betavals = [],
                     gammavals = [0.01],
                     avals = [0.0714],#0.142
                     muvals = [0.001],
                     beta1vals = [ 0.01, 0.2, 1, 1.3, 1.6, 3, 3.5, 4, 5, 7],
                     #beta1vals = [],
                     #beta2vals = [],
                     beta2vals = [0.05, 2.25, 4.5, 5, 7, 10],
                     train_valid_split = 0.8,
                     nmin_train_set = 10,
                     death_lm = 2,
                     recover_lm = 2,
                     evals = [0.92],
                     lvals = [15],
                     tauvals = [10],
                     verbose = True):
            self.nmin = nmin
            self.date = date
            self.region = region
            self.target = target
            self.population = population
            self.optimizer = optimizer
            self.betavals = betavals
            self.gammavals = gammavals
            self.avals = avals
            self.muvals = muvals
            self.beta1vals = beta1vals
            self.beta2vals = beta2vals
            self.nmin_train_set = nmin_train_set
            self.train_valid_split = train_valid_split
            self.death_lm = death_lm
            self.recover_lm = recover_lm
            self.verbose = verbose
            self.trained_warmstart = None
            self.evals = evals
            self.lvals = lvals
            self.tauvals = tauvals
            self.trained_param = None

        def fit(self,
                df,
                retrain_warmstart = False):

            dataset = deepcopy(df)
            dataset.set_index([self.date])
            regions = dataset[self.region].unique()
            output = dict()
            warmstart = dict()
            population_df = dataset.loc[:, [self.region, self.population]]
            population = {population_df.iloc[i, 0] : population_df.iloc[i, 1] for i in range(population_df.shape[0])}
            
            
            for i in range(len(regions)):
                region = regions[i]
                train_full_set = dataset[[a for a in dataset[self.region] == region]]
                train_full_set = train_full_set.sort_values(self.date)
                train_full_set['active'] = train_full_set['cases'] - train_full_set['cases'].shift(14)
                train_full_set = train_full_set.dropna(subset=['active'])
                train_full_set = train_full_set[[a for a in train_full_set["active"] > self.nmin]]
                
                
                #for counties
                region_pop = population[region]
                if train_full_set.shape[0] > self.nmin_train_set and region_pop > 0:
                    train_full_set = train_full_set.sort_values(self.date)

                    # if self.region == 'fips':
                    #     try:
                    #         list_1 = []
                    #         for i in range(len(train_full_set)):
                    #             val_1 = train_full_set['active'].values[i]
                    #             val_2 = ((train_full_set['cases'].values[i])*val_1)/train_full_set['cases_state_state'].values[i]
                    #             list_1.append(val_2)
                    #         train_full_set['active'] = list_1
                    #     except:
                    #         pass

                    full_times = [j for j in range(len(train_full_set))]
                    full_cum_cases = train_full_set.loc[:, "cases"].values
                    full_cases = train_full_set.loc[:, "active"].values
                    full_dead = train_full_set.loc[:, "deaths"].values
                    full_recover = train_full_set.loc[:, "cases"].values - train_full_set.loc[:, "active"].values - train_full_set.loc[:, "deaths"].values

                    train_set, valid_set= np.split(train_full_set, [int(self.train_valid_split *len(train_full_set))])


                    timer = [j for j in range(len(train_set))]
                    train_cases = train_set.loc[:, "active"].values
                    train_cum_cases = train_set.loc[:, "cases"].values
                    train_dead = train_set.loc[:, "deaths"].values
                    train_recover = train_set.loc[:, "cases"].values - train_set.loc[:, "active"].values - train_set.loc[:, "deaths"].values
                    times = timer

                    valid_times = [j for j in range(len(valid_set))]
                    valid_cases = valid_set.loc[:, "active"].values
                    valid_dead = valid_set.loc[:, "deaths"].values
                    valid_recover = valid_set.loc[:, "cases"].values - valid_set.loc[:, "active"].values - valid_set.loc[:, "deaths"].values

                    region_pop = population[region]


                    if sum(train_dead)/sum(train_cases) > 0.01:
                        weight_dead = sum(train_cases)/sum(train_dead)
                    else:
                        weight_dead = 10

                    if sum(train_recover)/sum(train_cases) > 0.01:
                        weight_recover = sum(train_cases)/sum(train_recover)
                    else:
                        weight_recover = 10

                    i_t = full_cum_cases[1:(len(full_cum_cases)-1)] - full_cum_cases[0:(len(full_cum_cases)-2)]
                    r_t = full_recover[1:(len(full_recover)-1)] - full_recover[0:(len(full_recover)-2)]
                    d_t = full_dead[1:(len(full_dead)-1)] - full_dead[0:(len(full_dead)-2)]
                    S_t = np.array(region_pop) - full_cum_cases

                        #print(i_t)
                    M_t = 1
                    M_t_list = []
                    strange_list = []
                    p_list = []
                    wave_list = [0]
                    wave_start = 0
                    e= self.evals[0]
                    l = self.lvals[0]
                    tau = self.tauvals[0]
                    for t in range(2, len(i_t)):
                        beta = sum(i_t[max(0, wave_start-1):(t-1)])/sum(full_cases[wave_start:(t)]*S_t[wave_start:(t)]/region_pop)
                        i_t_pred = beta*S_t[(t-1)]*full_cases[(t-1)]/region_pop
                        strangeness = abs(i_t[(t-1)] - i_t_pred)
                        #print(strangeness)
                        strange_list.append(strangeness)
                        p = sum(strange_list >= strangeness)/(t-wave_start)
                        p_list.append(p)
                        #print(p)
                        M_t_prev = M_t
                        M_t = M_t * e*pow(p, (e-1))
                        M_t_list.append(M_t)
                        if M_t > l: #or abs(M_t - M_t_prev) > tau:
                            wave_start = t
                            wave_list.append(t)
                            M_t = 1
                            strange_list = []
                        #print(wave_list)
                        #foo = pd.DataFrame({'date':train_full_set['date'].values[4:], 'M_t':M_t_list})
                    print(wave_list)
                    pop_est = region_pop
                    prev_xest = None
                    for w in range(len(wave_list)):
                        #print("Wave nb {} for {}".format(w+1, region))
                        begin_wave = wave_list[w]
                        if w == len(wave_list)-1:
                            end_wave = len(full_cases)
                        else:
                            end_wave = wave_list[w+1]

                        wave_cases = full_cases[begin_wave:end_wave]
                        wave_cum_cases = full_cum_cases[begin_wave:end_wave]
                        wave_recover = full_recover[begin_wave:end_wave]
                        wave_dead = full_dead[begin_wave:end_wave]
                        #wave_times = full_times[begin_wave:end_wave]
                        wave_times = np.array(full_times[begin_wave:end_wave])

                        region_pop = population[region]

                        i_t = wave_cum_cases[1:(len(wave_cum_cases)-1)] - wave_cum_cases[0:(len(wave_cum_cases)-2)]
                        r_t = wave_recover[1:(len(wave_recover)-1)] - wave_recover[0:(len(wave_recover)-2)]
                        d_t = wave_dead[1:(len(wave_dead)-1)] - wave_dead[0:(len(wave_dead)-2)]
                        S_t = np.array(pop_est) - wave_cum_cases

                        beta = sum(i_t)/sum(wave_cases*S_t/pop_est)
                        gamma = sum(r_t)/sum(wave_cases)
                        mu = sum(d_t)/sum(wave_cases)

                        a = self.avals[0]

                        region_beta1vals = self.beta1vals.copy()
                        region_beta2vals = self.beta2vals.copy()
                        region_beta1vals.append(beta)
                        region_beta2vals.append(beta)

                        region_avals = self.avals.copy()

                        #Result less sensible in mu and gamma, only test the probabilistic approach
                        region_gammavals = [gamma]
                        region_muvals = [mu]

                        #region_gammavals = self.gammavals
                        #region_muvals = self.muvals
                        #region_gammavals.append(gamma)
                        #region_muvals.append(mu)

                        if w==0:
                            region_betavals = region_beta1vals
                        else:
                            region_betavals = region_beta2vals

                        param_list = []
                        mse_list = []
                        if self.verbose:
                            iteration = len(region_betavals)*len(region_gammavals)*len(region_avals)*len(region_muvals)
                            progress_bar = tqdm(range(iteration), desc = str(region))

                        beta_progress = range(len(region_betavals))
                        gamma_progress = range(len(region_gammavals))
                        a_progress = range(len(region_avals))
                        mu_progress = range(len(region_muvals))

                        for betaindex in beta_progress:
                            for gammaindex in gamma_progress:
                                for aindex in a_progress:
                                    for muindex in mu_progress:
                                        if self.verbose:
                                            progress_bar.update(1)

                                        params = [region_betavals[betaindex], region_gammavals[gammaindex], region_pop, region_avals[aindex], region_muvals[muindex]]
                                        if w==0:
                                            iniests = inifcn(params, wave_cases, wave_dead)
                                        else:
                                            iniests = prev_xest[len(prev_xest)-1]
                                        optimizer = optimize.minimize(NLL, params, args=(wave_cases, wave_dead, wave_recover, 2,2, weight_dead, weight_recover, wave_times, iniests), method= 'Nelder-Mead')

                                        paramests = np.abs(optimizer.x)
                                        #print(paramests[0])
                                        if w == 0:
                                            iniests = inifcn(paramests, wave_cases, wave_dead)
                                        else:
                                            iniests = prev_xest[len(prev_xest)-1]

                                        xest = ode(model, iniests, wave_times, args=(paramests,))
                                        param_list.append([region_betavals[betaindex],region_gammavals[gammaindex],region_avals[aindex],region_muvals[muindex]])

                                        active, recovered, dead = finfcn(xest)
                                        #print("len wave cases = {}, len active = {}".format(len(wave_cases), len(active)))
                                        mse = sqrt(mean_squared_error(wave_cases,active)) + self.death_lm*sqrt(mean_squared_error(wave_dead,dead)) + self.recover_lm*sqrt(mean_squared_error(wave_recover,recovered))
                                        mse_list.append(mse)
                                        
                                        # plt.figure()
                                        # plt.plot(wave_times, wave_cases, label = "cases")
                                        # plt.plot(wave_times, active, label = "sir")


                        #print(mse_list)
                        minindex = mse_list.index(min(mse_list))
                        beta = param_list[minindex][0]
                        gamma = param_list[minindex][1]
                        a = param_list[minindex][2]
                        mu = param_list[minindex][3]
                        params = [beta, gamma, region_pop, a, mu]
                        print("warm start beta = {}".format(beta))
                        paramnames = ['beta', 'gamma', 'pop', 'a', 'mu']
                        warmstart[region] = params

                        #Retrain on the full dataset with the optimal values

                        if w == 0:
                            iniests = inifcn(params, wave_cases, wave_dead)
                        else:
                            iniests = prev_xest[len(prev_xest)-1]

                        #optimizer = optimize.minimize(NLL, params, args=(wave_cases, wave_dead, wave_recover, self.death_lm, self.recover_lm, weight_dead, weight_recover, wave_times, iniests), method=self.optimizer)
                        optimizer = optimize.minimize(NLL, params, args=(wave_cases, wave_dead, wave_recover, 2,2, weight_dead, weight_recover, wave_times, iniests), method= 'Nelder-Mead')
                        paramests = np.abs(optimizer.x)
                        print("optimizer's beta = {}".format(paramests[0]))
                        
                        if w == 0:
                            iniests = inifcn(paramests, wave_cases, wave_dead)

                        xest = ode(model, iniests, wave_times, args=(paramests,))

                        if w == 0:
                            full_xest = xest
                        else:
                            full_xest = np.concatenate((full_xest, xest))


                        pop_est = paramests[2]
                        prev_xest = xest

                        if w == len(wave_list)-1:
                            #print(full_xest)
                            output[region] = [paramests, full_xest[0,:], full_xest[len(full_xest)-1,:], train_full_set.date.iloc[0], train_full_set.date.iloc[len(train_full_set) - 1], full_xest]
                            self.trained_param = output

                        
            self.trained_warmstart = warmstart





        def predict(self,
                    regions,
                    dates):
            results = dict()
            for i in range(len(regions)):
                region = regions[i]
                region_params = self.trained_param[region]
                params = region_params[0]
                #print(params)
                start_vals = region_params[1]
                end_vals = region_params[2]
                start_date = region_params[3]
                end_date = region_params[4]

                insample_dates = []
                outsample_dates = []
                for d in dates:
                    if d >= start_date and d <= end_date:
                        insample_dates.append(d)
                    elif d >= end_date:
                        outsample_dates.append(d)

                # Calculate training preds
                train_pred = pd.DataFrame()
                train_dates = pd.DataFrame()
                if len(insample_dates) > 0:
                    tDelta = end_date - start_date

                    times = [k for k in range(tDelta.days)]
                    ini = start_vals
                    paramests = params
                    res = ode(model, ini, times, args=(paramests,))
                    active, recovered, dead  = finfcn(res)
                    if self.target == "cases":
                        train_pred = active + recovered + dead
                    elif self.target == "active":
                        train_pred = active
                    elif self.target == "deaths":
                        train_pred = dead
                    train_dates = [start_date + timedelta(days=x) for x in range(tDelta.days)]


                # Calculate testing preds
                test_pred = pd.DataFrame()
                test_dates = pd.DataFrame()

                last_date = max(dates)
                tDelta = last_date - end_date

                times = [k for k in range(tDelta.days + 1)]
                ini1 = end_vals
                # Simulate the model
                res = ode(model, ini1, times, args=(params,))
                active, recovered, dead  = finfcn(res)
                if self.target == "cases":
                    test_pred = active + recovered + dead
                elif self.target == "active":
                    test_pred = active
                elif self.target == "deaths":
                    test_pred = dead
                test_dates = [end_date + timedelta(days=x) for x in range(tDelta.days + 1)]

                if len(outsample_dates) > 0 and len(insample_dates) > 0:
                    df_fin = pd.DataFrame(np.concatenate((train_pred, test_pred)), index=np.concatenate((train_dates, test_dates)))
                elif len(insample_dates) > 0:
                    df_fin = pd.DataFrame(train_pred, index=train_dates)
                else:
                    df_fin = pd.DataFrame(test_pred, index=test_dates)

                results[region] = df_fin.loc[list(np.array(dates)[[date >= start_date for date in dates]]), 0]
            return results
