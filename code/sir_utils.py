import numpy as np
import pandas as pd
from scipy.stats import poisson
from scipy.stats import norm
from scipy.integrate import odeint as ode
import scipy.stats as stats
import datetime
import scipy.optimize as optimize


def model(ini, time_step, params):
	Y = np.zeros(3) #column vector for the state variables
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


def NLL(params, data, times): #negative log likelihood
    params = np.abs(params)
    data = np.array(data)
    res = ode(model, x0fcn(params,data), times, args =(params,))
    y = yfcn(res, params)
    nll = sum((y) - (data*np.log(y)))
    #nll = -sum(np.log(poisson.pmf(np.round(data),np.round(y))))
    #nll = -sum(np.log(norm.pdf(data,y,0.1*np.mean(data))))
    return nll