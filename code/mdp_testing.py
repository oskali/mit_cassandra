# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 23:13:09 2020

@author: Amine, omars
"""

#%% Libraries

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from datetime import timedelta
import math
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

#%% New Exception for Prediction Error
class PredictionError(Exception):
    pass

#%% Helper Functions for Prediction

# get_predictions() takes in a clustered dataframe df_new, and maps each
# CLUSTER to an OG_CLUSTER that has the most elements
# Returns a dataframe of the mappings
def get_predictions(df_new):
    df0 = df_new.groupby(['CLUSTER', 'OG_CLUSTER'])['ACTION'].count()
    df0 = df0.groupby('CLUSTER').idxmax()
    df2 = pd.DataFrame()
    df2['OG_CLUSTER'] = df0.apply(lambda x: x[1])
    return df2


# predict_cluster() takes in a clustered dataframe df_new, the number of
# features pfeatures, and returns a prediction model m that predicts the most
# likely cluster from a datapoint's
def predict_cluster(df_new, # dataframe: trained clusters
                    pfeatures): # int: # of features
    X = df_new.iloc[:, 2:2+pfeatures]
    y = df_new['CLUSTER']

    params = {
    'max_depth': [3, 4, 6, 10,None]
    }

    m = DecisionTreeClassifier()

    m = GridSearchCV(m, params,cv = 5, iid=True) #will return warning if 'idd' param not set to true

#    m = DecisionTreeClassifier(max_depth = 10)
    try:
        m.fit(X, y)
    except ValueError:
        print('ERROR: Feature columns have missing values! Please drop' \
              ' rows or fill in missing data.', flush=True)
        #print('Warning: Feature Columns missing values!', flush=True)
        #df_new.dropna(inplace=True)
        #X = df_new.iloc[:, 2:2+pfeatures]
        #y = df_new['CLUSTER']
        m.fit(X, y)
    return m


# predict_value_of_cluster() takes in MDP parameters, a cluster label, and
# and a list of actions, and returns the predicted value of the given cluster
def predict_value_of_cluster(P_df,R_df, # df: MDP parameters
                             cluster, # int: cluster number
                             actions): # list: list of actions
    s = cluster
    v = R_df.loc[s]
    for a in actions:
        s = P_df.loc[s,a].values[0]
        v = v + R_df.loc[s]
    return v


# get_MDP() takes in a clustered dataframe df_new, and returns dataframes
# P_df and R_df that represent the parameters of the estimated MDP
def get_MDP(df_new):
    # removing None values when counting where clusters go
    df0 = df_new[df_new['NEXT_CLUSTER']!='None']
    transition_df = df0.groupby(['CLUSTER','ACTION','NEXT_CLUSTER'])['RISK'].count()

    # next cluster given how most datapionts transition for the given action
    transition_df = transition_df.groupby(['CLUSTER','ACTION']).idxmax()
    P_df = pd.DataFrame()
    P_df['NEXT_CLUSTER'] = transition_df.apply(lambda x: x[2])
    R_df = df_new.groupby('CLUSTER')['RISK'].mean()
    return P_df,R_df


# Auxiliary function for deployment
# predict_region_date() takes a given state and a date and returns the predicted target_colname
def predict_region_date(self, # MDP_model object
                        region_first_last_dates, # tuple (region, first_date, last_date), e.g (Alabama, Timestamp('2020-03-24 00:00:00'), Timestamp('2020-06-22 00:00:00'))
                        date, # target_colname date for prediciton, e.g. (Timestamp('2020-05-24 00:00:00'))
                        verbose=0):

        region, last_date = region_first_last_dates
        try:
            date = datetime.strptime(date, '%Y-%m-%d')
        except TypeError:
            pass

        # Case 1 : the input date occurs before the first available date for a given region
        try :
            assert date >= last_date
            n_days = (date-last_date).days
            return np.ceil(self.predict_region_ndays(region, n_days))
        except AssertionError:
            if verbose:
                print("Prediction Error type I ('{}', '{}'): the input occurs before the last available ('{}') date of the training set".format(region,
                                                                                                                                          str(date),
                                                                                                                                          str(last_date)                                                                                                                        ))
            raise PredictionError  # test

#%% Function for Plotting

def plot_pred(model, state, df_true, n_days):
    h = int(np.round(n_days/model.days_avg))
    df_true.loc[:, [model.date_colname]]= pd.to_datetime(df_true[model.date_colname])
    date = model.df[model.df[model.region_colname]== state].iloc[-1, 1]
    cases = model.df[model.df[model.region_colname]== state].iloc[-1][model.target_colname]
    dates = [date]
    cases_pred = [cases]

    s = model.df_trained[model.df_trained[model.region_colname]==state].iloc[-1, -2]
    r = 1
    for i in range(h):
        dates.append(date + timedelta((i+1)*model.days_avg))
        r = r*np.exp(model.R_df.loc[s])
        cases_pred.append(cases*r)
        s = model.P_df.loc[s,0].values[0]


    fig, ax = plt.subplots()
    ax.plot(df_true.loc[df_true['state']==state][model.date_colname], \
            df_true.loc[df_true['state']==state][model.target_colname], \
            label = 'True '+model.target_colname)
    ax.plot(dates, cases_pred, label='Predicted '+model.target_colname)
    ax.set_title('%s True vs Predicted '%state + model.target_colname)
    ax.set_xlabel('Date')
    ax.set_ylabel(model.target_colname)
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.show()

#%% Functions for Accuracy and Purity measures

# training_accuracy() takes in a clustered dataframe df_new, and returns the
# average training accuracy of all clusters (float) and a dataframe of
# training accuracies for each OG_CLUSTER
def training_accuracy(df_new):
    clusters = get_predictions(df_new)
#    print('Clusters', clusters)

    # Tallies datapoints where the algorithm correctly classified a datapoint's
    # original cluster to be the OG_CLUSTER mapping of its current cluster
    accuracy = clusters.loc[df_new['CLUSTER']].reset_index()['OG_CLUSTER'] \
                                        == df_new.reset_index()['OG_CLUSTER']
    #print(accuracy)
    tr_accuracy = accuracy.mean()
    accuracy_df = accuracy.to_frame('Accuracy')
    accuracy_df['OG_CLUSTER'] = df_new.reset_index()['OG_CLUSTER']
    accuracy_df = accuracy_df.groupby('OG_CLUSTER').mean()
    return (tr_accuracy, accuracy_df)


# testing_accuracy() takes in a testing dataframe df_test (unclustered),
# a df_new clustered dataset, a model from predict_cluster and
# Returns a float for the testing accuracy measuring how well the model places
# testing data into the right cluster (mapped from OG_CLUSTER), and
# also returns a dataframe that has testing accuracies for each OG_CLUSTER
def testing_accuracy(df_test, # dataframe: testing data
                     df_new, # dataframe: clustered on training data
                     model, # function: output of predict_cluster
                     pfeatures): # int: # of features

    clusters = get_predictions(df_new)

    test_clusters = model.predict(df_test.iloc[:, 2:2+pfeatures])
    df_test['CLUSTER'] = test_clusters

    accuracy = clusters.loc[df_test['CLUSTER']].reset_index()['OG_CLUSTER'] \
                                        == df_test.reset_index()['OG_CLUSTER']
    #print(accuracy)
    tr_accuracy = accuracy.mean()
    accuracy_df = accuracy.to_frame('Accuracy')
    accuracy_df['OG_CLUSTER'] = df_test.reset_index()['OG_CLUSTER']
    accuracy_df = accuracy_df.groupby('OG_CLUSTER').mean()
    return (tr_accuracy, accuracy_df)


# purity() takes a clustered dataframe and returns a dataframe with the purity
# of each cluster
def purity(df):
    su = pd.DataFrame(df.groupby(['CLUSTER'])['OG_CLUSTER']
    .value_counts(normalize=True)).reset_index(level=0)
    su.columns= ['CLUSTER','Purity']
    return su.groupby('CLUSTER')['Purity'].max()

#%% Functions for Value Error

# training_value_error() takes in a clustered dataframe, and computes the
# E((\hat{v}-v)^2) expected error in estimating values (risk) given actions
# Returns a float of average value error per ID
def training_value_error(df_new, #Outpul of algorithm
                         relative=False, #Output Raw error or RMSE ie ((\hat{v}-v)/v)^2
                         h=5 #Length of forcast. The error is computed on v_h = \sum_{t=h}^H v_t
                         ):
    E_v = 0
    P_df,R_df = get_MDP(df_new)
    df2 = df_new.reset_index()
    df2 = df2.groupby(['ID']).first()
    N_train = df2.shape[0]


    for i in range(N_train):
        index = df2['index'].iloc[i]
        # initializing first state for each ID
        cont = True
        H = -1
            # Computing Horizon H of ID i
        while cont:
            H+= 1
            try:
                df_new['ID'].loc[index+H+1]
            except:
                break
            if df_new['ID'].loc[index+H] != df_new['ID'].loc[index+H+1]:
                break

        t = H-h
        s = df_new['CLUSTER'].loc[index + t]
        a = df_new['ACTION'].loc[index + t]
        v_true = df_new['RISK'].loc[index + t]
        v_estim = R_df.loc[s]
        t += 1
        #t = H-h +1
        # predicting path of each ID
        while cont:
            v_true = v_true + df_new['RISK'].loc[index + t]
            try:
                s = P_df.loc[s,a].values[0]
            # error raises in case we never saw a given transition in the data
            except TypeError:
                print('WARNING: In training value evaluation, trying to predict next state from state',s,'taking action',a,', but this transition is never seen in the data. Data point:',i,t)
            a = df_new['ACTION'].loc[index + t]
            v_estim = v_estim + R_df.loc[s]
            try:
                df_new['ID'].loc[index+t+1]
            except:
                break
            if df_new['ID'].loc[index+t] != df_new['ID'].loc[index+t+1]:
                break
            t += 1
        if relative:
            #E_v = E_v + ((math.exp(v_true)-math.exp(v_estim))/math.exp(v_true))**2
            E_v = E_v + np.abs((math.exp(v_true)-math.exp(v_estim))/math.exp(v_true))
        else:
            E_v = E_v + np.abs(math.exp(v_true)-math.exp(v_estim))
    E_v = (E_v/N_train)
    return (E_v)
    #return E_v


# testing_value_error() takes in a dataframe of testing data, and dataframe of
# new clustered data, a model from predict_cluster function, and computes the
# expected value error given actions and a predicted initial cluster
# Returns a float of sqrt average value error per ID
def testing_value_error(df_test, df_new, model, pfeatures,relative=False,h=5):
    try:
        df_test['CLUSTER'] = model.predict(df_test.iloc[:, 2:2+pfeatures])
    except ValueError:
        print('ERROR: Feature columns have missing values! Please drop' \
              ' rows or fill in missing data.', flush=True)
        df_test['CLUSTER'] = model.predict(df_test.iloc[:, 2:2+pfeatures])
    #except ValueError:
        #print('Warning: Feature Columns missing values!')
        #df_test.dropna(inplace=True)
        #model.predict(df_test.iloc[:, 2:2+pfeatures])
        #df_test['CLUSTER'] = model.predict(df_test.iloc[:, 2:2+pfeatures])

    E_v = 0
    P_df,R_df = get_MDP(df_new)
    df2 = df_test.reset_index()
    df2 = df2.groupby(['ID']).first()
    N_test = df2.shape[0]
#    print(df2)

    for i in range(N_test):
#        print('new item')
        # initializing index of first state for each ID
        index = df2['index'].iloc[i]
        cont = True
        H = -1
        # Computing Horizon H of ID i
        while cont:
            H+= 1
            try:
                df_test['ID'].loc[index+H+1]
            except:
                break
            if df_test['ID'].loc[index+H] != df_test['ID'].loc[index+H+1]:
                break

        t = H-h
        s = df_test['CLUSTER'].loc[index + t]
        a = df_test['ACTION'].loc[index + t]
        v_true = df_test['RISK'].loc[index + t]
        v_estim = R_df.loc[s]
        t += 1
        # predicting path of each ID
        while cont:
            v_true = v_true + df_test['RISK'].loc[index + t]
#            print('true', df_test['RISK'].loc[index + t])
            try:
                s = P_df.loc[s,a].values[0]
#                print('new s', s)
            # error raises in case we never saw a given transition in the data
            except TypeError:
                print('WARNING: In training value evaluation, trying to predict next state from state',s,'taking action',a,', but this transition is never seen in the data. Data point:',i,t)
            a = df_test['ACTION'].loc[index + t]
            v_estim = v_estim + R_df.loc[s]
#            print('s', s, 'estim', R_df.loc[s])
            try:
                df_test['ID'].loc[index+t+1]
            except:
                break
            if df_test['ID'].loc[index+t] != df_test['ID'].loc[index+t+1]:
                break


            t += 1
        if relative:
            #E_v = E_v + ((math.exp(v_true)-math.exp(v_estim))/math.exp(v_true))**2
            E_v = E_v + np.abs((math.exp(v_true)-math.exp(v_estim))/math.exp(v_true))
#            print('new E_v', E_v)
        else:
            E_v = E_v + np.abs(math.exp(v_true)-math.exp(v_estim))

    E_v = (E_v/N_test)
#    print('final E_v', E_v)
    #print('sqrt', np.sqrt(E_v))
    #return np.sqrt(E_v)
    return E_v


def error_per_ID(df_test, df_new, model, pfeatures,relative=False,h=5):
    try:
        df_test['CLUSTER'] = model.predict(df_test.iloc[:, 2:2+pfeatures])
    except ValueError:
        print('ERROR: Feature columns have missing values! Please drop' \
              ' rows or fill in missing data.', flush=True)
        df_test['CLUSTER'] = model.predict(df_test.iloc[:, 2:2+pfeatures])
    #except ValueError:
        #print('Warning: Feature Columns missing values!')
        #df_test.dropna(inplace=True)
        #model.predict(df_test.iloc[:, 2:2+pfeatures])
        #df_test['CLUSTER'] = model.predict(df_test.iloc[:, 2:2+pfeatures])
    E_v = 0
    P_df,R_df = get_MDP(df_new)
    df2 = df_test.reset_index()
    df2 = df2.groupby(['ID']).first()
    N_test = df2.shape[0]
#    print(df2)


    V_true,V_estim,V_err,C_err,State = [],[],[],[],[]

    for i in range(N_test):

#        print('new item')
        # initializing index of first state for each ID
        index = df2['index'].iloc[i]
        # print('--------')
        # print(df_test[df_test['ID'] == df_test['ID'].loc[index]]['state'].iloc[0])

        cont = True
        H = -1
        # Computing Horizon H of ID i
        while cont:
            H+= 1
            try:
                df_test['ID'].loc[index+H+1]
            except:
                break
            if df_test['ID'].loc[index+H] != df_test['ID'].loc[index+H+1]:
                break

        t = H-h
        s = df_test['CLUSTER'].loc[index + t]
        a = df_test['ACTION'].loc[index + t]
        v_true = df_test['RISK'].loc[index+t]
        v_estim = R_df.loc[s]
        # print(v_true,df_test['RISK'].loc[index+t], v_estim)
        t += 1
        # predicting path of each ID
        while cont:

            # print(v_true,df_test['RISK'].loc[index+t],R_df.loc[s])
            v_true = v_true + df_test['RISK'].loc[index + t]
#            print('true', df_test['RISK'].loc[index + t])
            try:
                s = P_df.loc[s,a].values[0]
#                print('new s', s)
            # error raises in case we never saw a given transition in the data
            except TypeError:
                print('WARNING: In training value evaluation, trying to predict next state from state',s,'taking action',a,', but this transition is never seen in the data. Data point:',i,t)
            a = df_test['ACTION'].loc[index + t]
            v_estim = v_estim + R_df.loc[s]
#            print('s', s, 'estim', R_df.loc[s])
            try:
                df_test['ID'].loc[index+t+1]
            except:
                break
            if df_test['ID'].loc[index+t] != df_test['ID'].loc[index+t+1]:
                break

            t += 1
        if relative:
            #E_v = E_v + ((math.exp(v_true)-math.exp(v_estim))/math.exp(v_true))**2
            E_v = E_v + np.abs((math.exp(v_true)-math.exp(v_estim))/math.exp(v_true))
#            print('new E_v', E_v)
        else:
            E_v = E_v + np.abs(math.exp(v_true)-math.exp(v_estim))

        State.append(df_test[df_test['ID'] == df_test['ID'].loc[index]]['state'].iloc[0])
        V_true.append(v_true)
        V_estim.append(v_estim)
        V_err.append(np.abs(v_true-v_estim))
        C_err.append(np.abs((math.exp(v_true)-math.exp(v_estim))/math.exp(v_true)))
    df_err = pd.DataFrame()
    df_err['state'] = State
    df_err['v_true'] = V_true
    df_err['v_estim'] = V_estim
    df_err['v_err'] = V_err
    df_err['cases_rel_err'] = C_err

    E_v = (E_v/N_test)
#    print('final E_v', E_v)
    #print('sqrt', np.sqrt(E_v))
    #return np.sqrt(E_v)
    return df_err,E_v

#%% Functions for R2 Values

# R2_value_training() takes in a clustered dataframe, and returns a float
# of the R-squared value between the expected value and true value of samples
def R2_value_training(df_new):
    E_v = 0
    P_df,R_df = get_MDP(df_new)
    df2 = df_new.reset_index()
    df2 = df2.groupby(['ID']).first()
    N = df2.shape[0]
    V_true = []
    for i in range(N):
        # initializing starting cluster and values
        s = df2['CLUSTER'].iloc[i]
        a = df2['ACTION'].iloc[i]
        v_true = df2['RISK'].iloc[i]

        v_estim = R_df.loc[s]
        index = df2['index'].iloc[i]
        cont = True
        t = 1
        # iterating through path of ID
        while cont:
            v_true = v_true + df_new['RISK'].loc[index + t]

            try:
                s = P_df.loc[s,a].values[0]
            # error raises in case we never saw a given transition in the data
            except TypeError:
                print('WARNING: Trying to predict next state from state',s,'taking action',a,', but this transition is never seen in the data. Data point:',i,t)
            a = df_new['ACTION'].loc[index + t]
            v_estim = v_estim + R_df.loc[s]
            try:
                df_new['ID'].loc[index+t+1]
            except:
                break
            if df_new['ID'].loc[index+t] != df_new['ID'].loc[index+t+1]:
                break
            t += 1
        E_v = E_v + (v_true-v_estim)**2
        V_true.append(v_true)
    # defining R2 baseline & calculating the value
    E_v = E_v/N
    V_true = np.array(V_true)
    v_mean = V_true.mean()
    SS_tot = sum((V_true-v_mean)**2)/N
    return max(1- E_v/SS_tot,0)


# R2_value_testing() takes a dataframe of testing data, a clustered dataframe,
# a model outputted by predict_cluster, and returns a float of the R-squared
# value between the expected value and true value of samples in the test set
def R2_value_testing(df_test, df_new, model, pfeatures):
    E_v = 0
    P_df,R_df = get_MDP(df_new)
    df2 = df_test.reset_index()
    df2 = df2.groupby(['ID']).first()
    N = df2.shape[0]

    # predicting clusters based on features
    clusters = model.predict(df2.iloc[:, 2:2+pfeatures])
    df2['CLUSTER'] = clusters

    V_true = []
    for i in range(N):
        s = df2['CLUSTER'].iloc[i]
        a = df2['ACTION'].iloc[i]
        v_true = df2['RISK'].iloc[i]

        v_estim = R_df.loc[s]
        index = df2['index'].iloc[i]
        cont = True
        t = 1
        while cont:
            v_true = v_true + df_test['RISK'].loc[index + t]


            try:
                s = P_df.loc[s,a].values[0]
            # error raises in case we never saw a given transition in the data
            except TypeError:
                print('WARNING: Trying to predict next state from state',s,'taking action',a,', but this transition is never seen in the data. Data point:',i,t)
            a = df_test['ACTION'].loc[index + t]

            v_estim = v_estim + R_df.loc[s]
            try:
                df_test['ID'].loc[index+t+1]
            except:
                break
            if df_test['ID'].loc[index+t] != df_test['ID'].loc[index+t+1]:
                break
            t += 1
        E_v = E_v + (v_true-v_estim)**2
        V_true.append(v_true)
    E_v = E_v/N
    V_true = np.array(V_true)
    v_mean = V_true.mean()
    SS_tot = sum((V_true-v_mean)**2)/N
    return max(1- E_v/SS_tot,0)

#%% Functions for Plotting Features

# plot_features() takes in a dataframe of two features, and plots the data
# to illustrate the noise in each original cluster
def plot_features(df):
    df.plot.scatter(x='FEATURE_1',
                      y='FEATURE_2',
                      c='OG_CLUSTER',
                      colormap='viridis')
#    import seaborn as sns
#    sns.pairplot(x_vars=["FEATURE_1"], y_vars=["FEATURE_2"], data=df, hue="OG_CLUSTER", height=5)
    plt.show()

# plot_path() takes in a trained df_new, state, an h value, and plots the path
# (by ratio: e^v) of the MDP versus the actual state, given a horizon of prediction h
def plot_path(df_new, df, state, h, pfeatures, plot=True):
    state_df = show_state(df_new, df, state, pfeatures)
    H = state_df.shape[0]
    P_df,R_df = get_MDP(df_new)

    t = H-h

    v_true = state_df['r_t'][t:]
    v_estim = []

    s = state_df['CLUSTER'].iloc[t]
    s_seq = [s]
    #print('initial state', s)
    #a = state_df['ACTION'].iloc[i]
    v_estim.append(math.exp(R_df.loc[s]))
    t += 1
    while t < H:
        try:
            s = P_df.loc[s,0].values[0]
            s_seq.append(s)
        # except TypeError:
        #         print('WARNING: Trying to predict next state from state',s,'taking action',a,', but this transition is never seen in the data. Data point:',i,t)
        except TypeError:
                print('WARNING: Unobserved transition')
        #a = df_test['ACTION'].loc[index + t]
        v_estim.append(math.exp(R_df.loc[s]))
        t += 1

    v_estim = np.array(v_estim)
    #plt.plot(v_true)
    if plot:
        fig1, ax1 = plt.subplots()
        its = np.arange(-h, 0)
        ax1.plot(its, v_true, label= "True Ratio")
        ax1.plot(its, v_estim, label = "Predicted Ratio")
        ax1.set_title('%s True vs Predicted Ratios of Cases' %state)
        ax1.set_xlabel('Time Before Present')
        ax1.set_ylabel('Ratio of Cases')
        plt.legend()
        plt.show()
        # print('state sequence:', s_seq)

    E_v = sum(np.abs((v_estim - v_true)/v_estim))/h
    # print("error", E_v)
    return E_v, v_true, v_estim, s_seq

# plot_path_all() returns the ratios and sequence of states for an optimal longest path
# if opt = True: Find optimal path and stop there
# if opt = False: plot the error horizon over different time horizons of predictions
def plot_path_all(df_new, df, state, pfeatures, opt=True, plot=True):
    state_df = show_state(df_new, df, state, pfeatures)
    H = state_df.shape[0] # of datapoints
    errors = []
    prev = float('inf')

    for h in range(H, 0, -1):
        E_v, v_true, v_estim, s_seq = plot_path(df_new, df, state, h, pfeatures, plot)

        if opt and E_v > prev and h < 16: # arbitrary threshold for a decent prediction
            break

        v_true_prev = v_true
        v_estim_prev = v_estim
        s_seq_prev = s_seq
        errors.append(E_v)
        prev = E_v

    if opt!= True:
        fig2, ax2 = plt.subplots()
        its = np.arange(H, 0, -1)
        ax2.plot(its, errors)
        ax2.set_title('%s MAPE over different time horizons' %state)
        ax2.set_xlabel('Horizon of Prediction')
        ax2.set_ylabel('Error')
        plt.show()
        df_errors = pd.DataFrame(list(zip(its, errors)),
               columns =['h', 'Error'])
        # print(df_errors)
    return v_estim_prev, s_seq_prev, prev

# go through each state, aggregate their optimal paths, and return as a dataframe
# which also includes each state's sequence, ratios, and error
def all_paths(df, df_new, pfeatures, opt=True, plot=True):
    # sort paths by num of unique elements
    # plot them if their starting node hasn't been seen before
    states = df['state'].unique()
    sequences = []
    ratios = []
    errors = []
    for i in states:
        v_estim, s_seq, error = plot_path_all(df_new, df, i, pfeatures, opt, plot)
        errors.append(error)
        ratios.append(v_estim)
        sequences.append(s_seq)
    df_seq = pd.DataFrame(list(zip(states, sequences, ratios, errors)),
                          columns = ['state', 'sequence', 'ratios', 'error'])
    return df_seq


# plot_pred() takes a trained model, a specific US state name, and the df_true
# (sorted by TIME), and plots the predicted versus true cases for n_days
def plot_pred(model, state, df_true, n_days):
    h = int(np.round(n_days/model.days_avg))
    df_true.loc[:, [model.date_colname]]= pd.to_datetime(df_true[model.date_colname])
    date = model.df[model.df[model.region]== state].iloc[-1, 1]
    cases = model.df[model.df[model.region]== state].iloc[-1, 2]
    dates = [date]
    cases_pred = [cases]

    s = model.df_trained[model.df_trained[model.region]==state].iloc[-1, -2]
    r = 1
    for i in range(h):
        dates.append(date + timedelta((i+1)*model.days_avg))
        r = r*np.exp(model.R_df.loc[s])
        cases_pred.append(cases*r)
        s = model.P_df.loc[s,0].values[0]


    fig, ax = plt.subplots()
    ax.plot(df_true.loc[df_true['state']==state][model.date_colname], \
            df_true.loc[df_true['state']==state][model.target_colname], \
            label = 'True '+model.target_colname)
    ax.plot(dates, cases_pred, label='Predicted '+model.target_colname)
    ax.set_title('%s True vs Predicted '%state + model.target_colname)
    ax.set_xlabel('Date')
    ax.set_ylabel(model.target_colname)
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.show()
    return

#%% Post Hoc Analysis Functions

def cluster_size(df):
    return df.groupby('CLUSTER')['RISK'].agg(['count','mean','std','min','max'])

def show_state(df_new,df,state,pfeatures):
    model = predict_cluster(df_new,pfeatures)
    st = df[df['state'] == state]
    st['CLUSTER'] = model.predict(st.iloc[:,2:pfeatures+2])
    return st[['TIME','cases','RISK','CLUSTER', 'r_t']]

def mape(df_pred,df_true, target_colname):
    df_pred['real '+target_colname] = df_true[target_colname]
    df_pred['rel_error'] = abs(df_pred[target_colname]-df_true[target_colname])/df_true[target_colname]
    return df_pred
