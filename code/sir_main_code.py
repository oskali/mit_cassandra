import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
#import import_ipynb
import sir_ode #file for ode functions
import sir_cost #file for optimizer function
import datetime
#matplotlib inline

from scipy.integrate import odeint as ode
import scipy.optimize as optimize

dataset = pd.read_csv('../data/input/May_data.csv', parse_dates = True)#change file name and path here

pops_states = pd.read_csv('../data/input/state_population.csv')#change population file name and path here
print('data read')
pops = pops_states.iloc[:,1].values
for i in range(len(pops)):
    pops[i] = float(pops[i])

d = dataset.iloc[:,1].values
for i in range(len(d)):
    d[i] = datetime.datetime.strptime(d[i],'%d-%m-%Y')   
dataset['date'] = d
dataset.set_index(['date'])


end_date1 = '2020-04-30'#set end of training set here
end_date2 = '2020-05-14'#set end of actual dataset here
before_end_date1 = dataset['date'] <= end_date1
before_end_date2 = dataset['date'] <= end_date2

df5 = pd.DataFrame()
refarray = ['Alabama','Alaska','Arizona','Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware', 'District of Columbia', 'Florida', 'Georgia', 'Guam', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Puerto Rico', 'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming']
for i in range(len(refarray)):
    df = pd.DataFrame()
    df = dataset.loc[(dataset['state'].isin([refarray[i]])) & (dataset['cases']>100)]#change minimum number of cases here
    train_set = pd.DataFrame()
    actual_set = pd.DataFrame()
    preds_set = pd.DataFrame()
    train_set = df.loc[before_end_date1]
    actual_set = df.loc[before_end_date2]
    preds_set = pd.concat([train_set,actual_set]).drop_duplicates(keep=False)#prediction set to create test set results
    
    timer = []
    for j in range(len(train_set)):
        timer.append(j)
    #train_set['time spent'] = timer

    actual_time = []
    for k in range(len(actual_set)):
        actual_time.append(k)

    pred_time = []
    for x in range(len(actual_set)):
        pred_time.append(x)


    data = train_set.iloc[:,2].values
    times = timer
    
    params = [0.4, 0.06, pops[i]]#change initial parameters here
    paramnames = ['beta', 'gamma', 'k']
    ini = sir_ode.x0fcn(params,data)


    #Simulate and plot the model 
    res = ode(sir_ode.model, ini, times, args=(params,))

    sim_measure = sir_ode.yfcn(res, params)
    
    #to plot the training set vs actual set with initialized parameters
    
    #plt.plot(timer, sim_measure, 'b-', linewidth=3, label='Model simulation')
    #plt.plot(actual_set.iloc[:,1].values, actual_set.iloc[:,2].values, linewidth=2, label='Data')
    #plt.xlabel('Time')
    #plt.ylabel('Individuals')
    #plt.legend()
    #plt.show()
   
    
    #Parameter estimation
    optimizer = optimize.minimize(sir_cost.NLL, params, args=(data,times), method='Nelder-Mead')
    paramests = np.abs(optimizer.x)
    print(paramests)#print estimated parameters
    iniests = sir_ode.x0fcn(paramests, data)

    
    xest = ode(sir_ode.model, iniests, times, args=(paramests,))
    est_measure = sir_ode.yfcn(xest, paramests)
    
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
    ini1 = sir_ode.x0fcn(params,data)


    #Simulate and plot the model 
    res = ode(sir_ode.model, ini1, actual_time, args=(params,))

    preds_test = sir_ode.yfcn(res, params)
    preds_test = np.delete(preds_test,times)
    print(preds_test) #print test set predictions

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

    df3 = pd.DataFrame(est_measure)
    df4 = pd.DataFrame(preds_test)
    df3 = df3.append(df4, ignore_index = True)
    df3 = np.array(df3)
    actual_set['predictions'] = df3
    df5 = df5.append(actual_set, ignore_index = True) #df5 stores the output for all states together   

df5.to_csv('../data/output/base_sir_pred.csv')
                                         
