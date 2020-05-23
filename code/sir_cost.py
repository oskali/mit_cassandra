import numpy as np
#import import_ipynb
import sir_ode
import sir_cost
from scipy.stats import poisson
from scipy.stats import norm

from scipy.integrate import odeint as ode

def NLL(params, data, times): #negative log likelihood
    params = np.abs(params)
    data = np.array(data)
    res = ode(sir_ode.model, sir_ode.x0fcn(params,data), times, args =(params,))
    y = sir_ode.yfcn(res, params)
    #nll = sum((y) - (data*np.log(y)))
    #nll = -sum(np.log(poisson.pmf(np.round(data),np.round(y))))
    nll = -sum(np.log(norm.pdf(data,y,0.1*np.mean(data))))
    return nll
