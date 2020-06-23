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
# Create class
class SIR:
    def fit(df_train, pop_dic, nmin=100):
        dataset = deepcopy(df_train)
        dataset.set_index(['date'])
        output = pd.DataFrame()
        states = dataset.state.unique()
        trained_param = dict()
        for i in range(len(states)):
            state = states[i]
            train_set = dataset.query('state == @state').query('cases >= @nmin')
            if train_set.shape[0] > 10:
                timer = [j for j in range(len(train_set))]
                data = train_set.loc[:, 'cases'].values
                times = timer
                params = [0.4, 0.06, pop_dic[state]]
                #paramnames = ['beta', 'gamma', 'k']
                ini = sir_ode.x0fcn(params,data)
        

                #Parameter estimation
                optimizer = optimize.minimize(sir_cost.NLL, params, args=(data,times), method='Nelder-Mead')
                paramests = np.abs(optimizer.x)
                iniests =  sir_ode.x0fcn(paramests, data)
                xest = ode(sir_ode.model, iniests, times, args=(paramests,))
                
                trained_param[state] = [paramests, xest[0,:], xest[len(xest)-1,:], train_set.date.iloc[0], train_set.date.iloc[len(train_set) - 1]]
        return(trained_param)

    def predict(trained_param, states, dates):
        results = dict()
        for i in range(len(states)):
            state = states[i]
            state_params = trained_param[state]
            params = state_params[0]
            start_vals = state_params[1]
            end_vals = state_params[2]
            start_date = state_params[3]
            end_date = state_params[4]
            
            insample_dates = []
            outsample_dates = []
            for d in dates:
                if d >= start_date and d <= end_date:
                    insample_dates.append(d)
                elif d >= end_date:
                    outsample_dates.append(d)
                    
            #calculate training preds
            train_pred = pd.DataFrame()
            train_dates = pd.DataFrame()
            if len(insample_dates) > 0:
                tDelta = end_date - start_date
                
                times = [k for k in range(tDelta.days)]
                ini = start_vals
        
                res = ode(sir_ode.model, ini, times, args=(paramests,))
                train_pred = sir_ode.yfcn(res, paramests)
                train_dates = [start_date + datetime.timedelta(days=x) for x in range(tDelta.days)]
                
                
            #calculate testing preds
            test_pred = pd.DataFrame()
            test_dates = pd.DataFrame()
            if len(outsample_dates) > 0:
                last_date = max(dates)
                tDelta = last_date - end_date 
                
                times = [k for k in range(tDelta.days + 1)]
                ini1 = end_vals
                #Simulate the model
                res = ode(sir_ode.model, ini1, times, args=(params,))
                test_pred = sir_ode.yfcn(res, params)
                test_dates = [end_date + datetime.timedelta(days=x) for x in range(tDelta.days + 1)]
            
            
            df_fin = pd.DataFrame(np.concatenate((train_pred, test_pred)), index=np.concatenate((train_dates, test_dates)))
            
            results[state] = df_fin.loc[dates, 0]
        return results

#############################################################################