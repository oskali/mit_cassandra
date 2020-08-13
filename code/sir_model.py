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
def inifcn(params, cases, deaths):
    S0 = params[2] - cases[0] - deaths[0]
    E0 = 0.0
    I0 = cases[0]
    R0 = 0.0
    D0 = deaths[0]
    X0 = [S0, E0, I0, R0, D0]
    return X0

#retrieve cumlative case compartments (active, recovered, dead)
def finfcn(res):
    return res[:,2], res[:,3], res[:,4]

#objective function to optimize hyperparameters
def NLL(params, cases, deaths, recover, death_lm, recover_lm, weight_dead, weight_recover, times): #loss function
    params = np.abs(params)
    cases = np.array(cases)
    deaths = np.array(deaths)
    res = ode(model, inifcn(params,cases,deaths), times, args =(params,))
    active, recovered, dead = finfcn(res)
    nll = sqrt(mean_squared_error(cases,active)) + weight_dead*death_lm*sqrt(mean_squared_error(deaths,dead)) + weight_recover*recover_lm*sqrt(mean_squared_error(recover,recovered)) #cases + lambda*deaths
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
                     betavals = [0.1, 0.9],
                     gammavals = [0.01, 0.25],
                     avals = [0.142, 0.0714],
                     muvals = [0.001, 0.005],
                     train_valid_split = 0.8,
                     nmin_train_set = 10,
                     death_lm = 1,
                     recover_lm = 1, 
                     verbose = False):
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
            nmin_value = self.nmin[self.region]

            for i in range(len(regions)):
                region = regions[i]
                train_full_set = dataset[[a and b for a, b in zip(dataset[self.region] == region, dataset["active"] > nmin_value)]]
                
                
                region_pop = population[region]
                if train_full_set.shape[0] > self.nmin_train_set and region_pop > 0:   
                    train_full_set = train_full_set.sort_values(self.date)
                    
                    if self.region == 'fips':
                        list_1 = []
                        for i in range(len(train_full_set)):
                            if train_full_set['state'].values[i] == 'Massachusetts':
                                #print('This county is in Mass')
                                val_1 = train_full_set['active'].values[i]
                                val_2 = ((train_full_set['cases'].values[i])*val_1)/train_full_set['state_cases'].values[i]
                                list_1.append(val_2)
                            elif train_full_set['state'].values[i] == 'New Jersey':
                                #print('This county is in NJ')
                                val_1 = train_full_set['active'].values[i]
                                val_2 = ((train_full_set['cases'].values[i])*val_1)/train_full_set['state_cases'].values[i]
                                list_1.append(val_2)
                        train_full_set['active'] = list_1                    
                    
                    
                    full_times = [j for j in range(len(train_full_set))]
                    full_cases = train_full_set.loc[:, "active"].values
                    full_dead = train_full_set.loc[:, "deaths"].values
                    full_recover = train_full_set.loc[:, "cases"].values - train_full_set.loc[:, "active"].values - train_full_set.loc[:, "deaths"].values
                    full_recover[full_recover<0] = 0
                    
                    train_set, valid_set= np.split(train_full_set, [int(self.train_valid_split *len(train_full_set))])

                    train_cases = train_set.loc[:, "active"].values
                    train_dead = train_set.loc[:, "deaths"].values
                    train_recover = train_set.loc[:, "cases"].values - train_set.loc[:, "active"].values - train_set.loc[:, "deaths"].values
                    train_recover[train_recover<0] = 0
                    times = [j for j in range(len(train_set))]

                    valid_times = [j for j in range(len(valid_set))]
                    valid_cases = valid_set.loc[:, "active"].values
                    valid_dead = valid_set.loc[:, "deaths"].values
                    valid_recover = valid_set.loc[:, "cases"].values - valid_set.loc[:, "active"].values - valid_set.loc[:, "deaths"].values
                    valid_recover[valid_recover<0] = 0

                    if sum(train_dead)/sum(train_cases) > 0.01:
                        weight_dead = sum(train_cases)/sum(train_dead)
                    else:
                        weight_dead = 10

                    if sum(train_recover)/sum(train_cases) > 0.01:
                        weight_recover = sum(train_cases)/sum(train_recover) 
                    else:
                        weight_recover = 10


                    
                    if self.trained_warmstart == None or retrain_warmstart == True:
                        param_list = []
                        mse_list = []
                        if self.verbose:
                            iteration = len(self.betavals)*len(self.gammavals)*len(self.avals)*len(self.muvals)
                            progress_bar = tqdm(range(iteration), desc = region)
                        else:
                            print(region)
              
                        beta_progress = range(len(self.betavals))
                        gamma_progress = range(len(self.gammavals))
                        a_progress = range(len(self.avals))
                        mu_progress = range(len(self.muvals))
    
                        for betaindex in beta_progress:
                            for gammaindex in gamma_progress:
                                for aindex in a_progress:
                                    for muindex in mu_progress:
                                        if self.verbose:
                                            progress_bar.update(1)
                                        params = [self.betavals[betaindex], self.gammavals[gammaindex], region_pop, self.avals[aindex], self.muvals[muindex]]
                                        optimizer = optimize.minimize(NLL, params, args=(train_cases, train_dead, train_recover, self.death_lm, self.recover_lm, weight_dead, weight_recover, times), method=self.optimizer)
                                        params = np.abs(optimizer.x)
                                        ini = inifcn(params, train_cases, train_dead)
                                        param_list.append([self.betavals[betaindex],self.gammavals[gammaindex],self.avals[aindex],self.muvals[muindex]])
                                        train_est = ode(model, ini, times, args=(params,)) 
                                        valid_ini = train_est[len(train_est)-1,:]
                                        valid_est = ode(model, valid_ini, valid_times, args=(params,))
                                        active, recovered, dead = finfcn(valid_est)
                                        mse = sqrt(mean_squared_error(valid_cases,active)) + weight_dead*self.death_lm*sqrt(mean_squared_error(valid_dead,dead)) + weight_recover*self.recover_lm*sqrt(mean_squared_error(valid_recover,recovered))
                                        mse_list.append(mse)

                        
                        minindex = mse_list.index(min(mse_list))
                        beta = param_list[minindex][0]
                        gamma = param_list[minindex][1]
                        a = param_list[minindex][2]
                        mu = param_list[minindex][3]
                        params = [beta, gamma, region_pop, a, mu]
                        paramnames = ['beta', 'gamma', 'pop', 'a', 'mu']
                        warmstart[region] = params
                        
                    else:
                        params = self.trained_warmstart[region]


                    optimizer = optimize.minimize(NLL, params, args=(full_cases, full_dead, full_recover, self.death_lm, self.recover_lm, weight_dead, weight_recover, full_times), method=self.optimizer)
                    paramests = np.abs(optimizer.x)
                    iniests = inifcn(paramests, full_cases, full_dead)
                    xest = ode(model, iniests, full_times, args=(paramests,))

                    output[region] = [paramests, xest[0,:], xest[len(xest)-1,:], train_full_set.date.iloc[0], train_full_set.date.iloc[len(train_full_set) - 1]]

                    
            
            self.trained_warmstart = warmstart
            self.trained_param = output
                    

        def predict(self,
                    regions,
                    dates):
            results = dict()
            for i in range(len(regions)):
                region = regions[i]
                if region in self.trained_param.keys():
                    region_params = self.trained_param[region]
                    params = region_params[0]
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