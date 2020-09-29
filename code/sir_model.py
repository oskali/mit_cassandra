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
                     # beta1vals = [0.01, 0.2, 1],
                     # beta2vals = [0.05, 2.25, 4.5],
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
                train_full_set = dataset[[a and b for a, b in zip(dataset[self.region] == region, dataset["active"] > self.nmin)]]
                
                #for counties
                region_pop = population[region]
                if train_full_set.shape[0] > self.nmin_train_set and region_pop > 0:   
                    train_full_set = train_full_set.sort_values(self.date)
                    
                    if self.region == 'fips':
                        try:
                            list_1 = []
                            for i in range(len(train_full_set)):
                                val_1 = train_full_set['active'].values[i]
                                val_2 = ((train_full_set['cases'].values[i])*val_1)/train_full_set['cases_state_state'].values[i]
                                list_1.append(val_2)
                            train_full_set['active'] = list_1
                        except:
                            pass
                    
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
                        strange_list.append(strangeness)
                        p = sum(strange_list >= strangeness)/t
                        p_list.append(p)
                        #print(p)
                        M_t_prev = M_t
                        M_t = M_t * e*pow(p, (e-1))
                        M_t_list.append(M_t)
                        if M_t > l or abs(M_t - M_t_prev) > tau:
                            wave_start = t
                            wave_list.append(t)
                            M_t = 1
                            strange_list = []
                        #foo = pd.DataFrame({'date':train_full_set['date'].values[4:], 'M_t':M_t_list})
                        
                        
                        
                        
                    #ONLY FOR 2-WAVE VER
                    if(len(wave_list) == 1):
                        beta = sum(i_t)/sum(full_cases*S_t/region_pop)
                        gamma = sum(r_t)/sum(full_cases)
                        mu = sum(d_t)/sum(full_cases)
                        a = self.avals[0]
                        
                        params = [beta, gamma, region_pop, a, mu]
                        paramnames = ['beta', 'gamma', 'pop', 'a', 'mu']
                        warmstart[region] = params
                        
                        iniests = inifcn(params, full_cases, full_dead)
                        optimizer = optimize.minimize(NLL, params, args=(full_cases, full_dead, full_recover, self.death_lm, self.recover_lm, weight_dead, weight_recover, full_times, iniests), method=self.optimizer)
                        paramests = np.abs(optimizer.x)
                        iniests = inifcn(paramests, full_cases, full_dead)
                        xest = ode(model, iniests, full_times, args=(paramests,))

                        output[region] = [paramests, xest[0,:], xest[len(xest)-1,:], train_full_set.date.iloc[0], train_full_set.date.iloc[len(train_full_set) - 1], xest]

                        self.trained_param = output
                    else:
                        new_wave = wave_list[1]
                        first_cases = full_cases[:new_wave]
                        first_cum_cases = full_cum_cases[:new_wave]
                        first_recover = full_recover[:new_wave]
                        first_dead = full_dead[:new_wave]
                        first_times = full_times[:new_wave]
                        
                        second_cases = full_cases[new_wave:]
                        second_cum_cases = full_cum_cases[new_wave:]
                        second_recover = full_recover[new_wave:]
                        second_dead = full_dead[new_wave:]
                        second_times = [j for j in range((len(full_times)-new_wave))]
                        
                        #first wave
                        i_t = first_cum_cases[1:(len(first_cum_cases)-1)] - first_cum_cases[0:(len(first_cum_cases)-2)]
                        r_t = first_recover[1:(len(first_recover)-1)] - first_recover[0:(len(first_recover)-2)]
                        d_t = first_dead[1:(len(first_dead)-1)] - first_dead[0:(len(first_dead)-2)]
                        S_t = np.array(region_pop) - first_cum_cases
                    
                        beta = sum(i_t)/sum(first_cases*S_t/region_pop)
                        gamma = sum(r_t)/sum(first_cases)
                        mu = sum(d_t)/sum(first_cases)
                        a = self.avals[0]
                        
                        params = [beta, gamma, region_pop, a, mu]
                        paramnames = ['beta', 'gamma', 'pop', 'a', 'mu']
                        warmstart[region] = params
                        
                        iniests = inifcn(params, first_cases, first_dead)
                        optimizer = optimize.minimize(NLL, params, args=(first_cases, first_dead, first_recover, self.death_lm, self.recover_lm, weight_dead, weight_recover, first_times, iniests), method=self.optimizer)
                        paramests = np.abs(optimizer.x)
                        iniests = inifcn(paramests, first_cases, first_dead)
                        first_xest = ode(model, iniests, first_times, args=(paramests,))
                        
                        #second wave
                        i_t = second_cum_cases[1:(len(second_cum_cases)-1)] - second_cum_cases[0:(len(second_cum_cases)-2)]
                        r_t = second_recover[1:(len(second_recover)-1)] - second_recover[0:(len(second_recover)-2)]
                        d_t = second_dead[1:(len(second_dead)-1)] - second_dead[0:(len(second_dead)-2)]
                        S_t = np.array(paramests[2]) - second_cum_cases
                        
                        beta = sum(i_t)/sum(second_cases*S_t/region_pop)
                        gamma = sum(r_t)/sum(second_cases)
                        mu = sum(d_t)/sum(second_cases)
                        a = self.avals[0]
                        
                        params = [beta, gamma, paramests[2], a, mu]
                        paramnames = ['beta', 'gamma', 'pop', 'a', 'mu']

                        iniests = first_xest[len(first_xest)-1]
                        optimizer = optimize.minimize(NLL, params, args=(second_cases, second_dead, second_recover, self.death_lm, self.recover_lm, weight_dead, weight_recover, second_times, iniests), method=self.optimizer)
                        paramests = np.abs(optimizer.x)
                        iniests = first_xest[len(first_xest)-1]
                        second_xest = ode(model, iniests, second_times, args=(paramests,))
                        
                        xest = np.concatenate((first_xest, second_xest))

                        output[region] = [paramests, xest[0,:], xest[len(xest)-1,:], train_full_set.date.iloc[0], train_full_set.date.iloc[len(train_full_set) - 1], xest]

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

