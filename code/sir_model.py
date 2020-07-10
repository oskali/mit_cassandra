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

#%% Helper Functions

def model(ini, time_step, params):
	Y = np.zeros(3) # Column vector for the region variables
	X = ini
	mu = 0
	beta = params[0]
	gamma = params[1]

	Y[0] = mu - beta*X[0]*X[1] - mu*X[0] #S
	Y[1] = beta*X[0]*X[1] - gamma*X[1] - mu*X[1] #I
	Y[2] = gamma*X[1] - mu*X[2] #R

	return Y

def x0fcn(params, data):
	S0 = 1.0 - (data[0]/params[2])
	I0 = data[0]/params[2]
	R0 = 0.0
	X0 = [S0, I0, R0]

	return X0


def yfcn(res, params):
	return res[:,1]*params[2]

def NLL(params, data, times): # Negative log likelihood
    params = np.abs(params)
    data = np.array(data)
    res = ode(model, x0fcn(params,data), times, args =(params,))
    y = yfcn(res, params)
    nll = sum((y) - (data*np.log(y)))
    #nll = -sum(np.log(poisson.pmf(np.round(data),np.round(y))))
    #nll = -sum(np.log(norm.pdf(data,y,0.1*np.mean(data))))
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
                     initial_param = [0.4, 0.06],
                     nmin_train_set = 10):
            self.nmin = nmin
            self.date = date
            self.region = region
            self.target = target
            self.population = population
            self.optimizer = optimizer
            self.initial_param = initial_param
            self.nmin_train_set = nmin_train_set
            self.trained_param = None

        def fit(self,
                df):

            dataset = deepcopy(df)
            dataset.set_index([self.date])
            output = pd.DataFrame()
            regions = dataset[self.region].unique()
            output = dict()
            population_df = dataset.loc[:, [self.region, self.population]]
            population = {population_df.iloc[i, 0] : population_df.iloc[i, 1] for i in range(population_df.shape[0])}
            for i in range(len(regions)):
                region = regions[i]
                print(region)
                train_set = dataset[[a and b for a, b in zip(dataset[self.region] == region, dataset[self.target] > self.nmin)]]
                if train_set.shape[0] > self.nmin_train_set:
                    timer = [j for j in range(len(train_set))]
                    data = train_set.loc[:, self.target].values
                    times = timer
                    params = self.initial_param + [population[region]]
                    #paramnames = ['beta', 'gamma', 'k']
                    #ini = x0fcn(params,data)


                    # Parameter estimation
                    optimizer = optimize.minimize(NLL, params, args=(data,times), method=self.optimizer)
                    paramests = np.abs(optimizer.x)
                    iniests =  x0fcn(paramests, data)
                    xest = ode(model, iniests, times, args=(paramests,))

                    output[region] = [paramests, xest[0,:], xest[len(xest)-1,:], train_set.date.iloc[0], train_set.date.iloc[len(train_set) - 1]]
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
                        paramests = self.trained_param[region][0]
                        res = ode(model, ini, times, args=(paramests,))
                        train_pred = yfcn(res, paramests)
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
                    test_pred = yfcn(res, params)
                    test_dates = [end_date + timedelta(days=x) for x in range(tDelta.days + 1)]

                    if len(outsample_dates) > 0 and len(insample_dates) > 0:
                        df_fin = pd.DataFrame(np.concatenate((train_pred, test_pred)), index=np.concatenate((train_dates, test_dates)))

                    elif len(insample_dates) > 0:
                        df_fin = pd.DataFrame(train_pred, index=train_dates)
                    else:
                        df_fin = pd.DataFrame(test_pred, index=test_dates)

                    results[region] = df_fin.loc[list(np.array(dates)[[date >= start_date for date in dates]]), 0]
            return results
