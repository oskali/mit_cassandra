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



def sir_fit(dataset) #dataset has columns state, date, cases
	pops_states = pd.read_csv('../data/input/state_population.csv') #change population file name and path here
	pred_out = 100
	pops = pops_states.iloc[:,1].values
	for i in range(len(pops)):
	    pops[i] = float(pops[i])

	d = dataset.iloc[:,1].values
	for i in range(len(d)):
	    d[i] = datetime.datetime.strptime(d[i],'%d-%m-%Y')   


	dataset['date'] = d
	dataset.set_index(['date'])
	df5 = pd.DataFrame()
	refarray = dataset.state.unique()
	state_model_dict = dict()
	for i in range(len(refarray)):
	    df = pd.DataFrame()
	    train_set = dataset.loc[(dataset['state'].isin([refarray[i]])) & (dataset['cases']>100)]#change minimum number of cases here
#	    actual_set = pd.DataFrame()
#	    preds_set = pd.DataFrame()
#	    train_set = df.loc[before_end_date1]
#	    actual_set = df.loc[before_end_date2]
#	    preds_set = pd.concat([train_set,actual_set]).drop_duplicates(keep=False)#prediction set to create test set results
	    
	    timer = []
	    for j in range(len(train_set)):
	        timer.append(j)
	    #train_set['time spent'] = timer

	    actual_time = []
	    for k in range(pred_out + len(train_set)):
	        actual_time.append(k)

#	    pred_time = []
#	    for x in range(len(actual_set)):
#	        pred_time.append(x)


	    data = train_set.iloc[:,2].values
	    times = timer
	    
	    params = [0.4, 0.06, pops[i]]#change initial parameters here
	    paramnames = ['beta', 'gamma', 'k']
	    ini = x0fcn(params,data)


	    #Simulate and plot the model 
	    res = ode(model, ini, times, args=(params,))

	    sim_measure = yfcn(res, params)
	    
	    #to plot the training set vs actual set with initialized parameters
	    
	    #plt.plot(timer, sim_measure, 'b-', linewidth=3, label='Model simulation')
	    #plt.plot(actual_set.iloc[:,1].values, actual_set.iloc[:,2].values, linewidth=2, label='Data')
	    #plt.xlabel('Time')
	    #plt.ylabel('Individuals')
	    #plt.legend()
	    #plt.show()
	   
	    
	    #Parameter estimation
	    optimizer = optimize.minimize(NLL, params, args=(data,times), method='Nelder-Mead')
	    paramests = np.abs(optimizer.x)
	    print(paramests)#print estimated parameters
	    iniests = x0fcn(paramests, data)

	    
	    xest = ode(model, iniests, times, args=(paramests,))
	    est_measure = yfcn(xest, paramests)
	    
	    #to plot the training set vs actual set with estimated results for the training set
	    
	    #f = plt.figure()
	    #plt.plot(timer, est_measure, 'b-', linewidth=3, label='Model simulation')
	    #plt.plot(actual_set.iloc[:,1], actual_set.iloc[:,2], 'k-o', linewidth=2, label='Data')

	    #plt.xlabel('Time')
	    #plt.ylabel('Individuals')
	    #plt.legend()
	    #plt.show()
	    #f.savefig(r'C:\Users\akars\OneDrive\Desktop\Summer 2020\MIT\Washington_Graph_May_Data.pdf')
	    est_measure = np.array(est_measure)
	    #df['predictions'] = est_measure

	    params = paramests
	    ini1 = x0fcn(params,data)

	    #Simulate and plot the model 
	    res = ode(model, ini1, actual_time, args=(params,))


	    preds_test = yfcn(res, params)
	    preds_test = np.delete(preds_test,times)
	    #print(preds_test) #print test set predictions

	    #to plot training set, actual set and test set predictions
	    
	    #f = plt.figure()
	    #plt.plot(timer, est_measure, 'b-', linewidth=3, label='Training set',color='red')
	    #plt.plot(preds_set.iloc[:,1].values, preds_test, 'b-', linewidth=3, label='Test set',color='blue')
	    #plt.plot(actual_set.iloc[:,1].values, actual_set.iloc[:,2].values, linewidth=2, label='Data',color='green')
	    #plt.xlabel('Time')
	    #plt.ylabel('Individuals')
	    #plt.legend()
	    #plt.show()
	    #f.savefig(r'C:\Users\akars\OneDrive\Desktop\Summer 2020\MIT\Washington_Graph_May_Data_Test_Data.pdf')

	    pred_train = pd.DataFrame(est_measure)
	    pred_test = pd.DataFrame(preds_test)
	    df_fin = pred_train.append(pred_test, ignore_index = True)
	    df_fin.columns = ["pred"]
#	    pred_all = np.array(pred_all)

	    last_date = train_set.date.tail(1)
	    date_list = [last_date + datetime.timedelta(days=x) for x in range(pred_out)]
	    df_fin["date"] = train_set.date.append(date_list, ignore_index = True)
#	    date_array = np.array(date_array)

		df_fin["state"] = refarray[i]

		df5 = df5.append(df_fin, ignore_index = True) #df5 stores the output for all states together   
	return df5



def sir_fit(dataset, train_df) #dataset has columns state, date                               
	pred = pd.merge(dataset,train_df,on=['date', 'state'])
	return pred