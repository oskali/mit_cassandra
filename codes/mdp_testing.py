# -*- coding: utf-8 -*-
"""
This file is intended to perform various testing measurements on the output of

the MDP Clustering Algorithm.

Created on Sun Apr 26 23:13:09 2020

@author: Amine
"""
#############################################################################
# Load Libraries

# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 23:13:09 2020

@author: Amine, omars
"""

# %% Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from datetime import timedelta
import math
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from itertools import product
from collections import Counter

# %% New Exception for MDP learning
class MDPPredictionError(Exception):
    pass


class MDPTrainingError(Exception):
    pass


# %% Helper Functions for Prediction

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
def predict_cluster(df_new,  # dataframe: trained clusters
                    pfeatures,
                    cv=5):  # int: # of features
    X = df_new.iloc[:, 2:2 + pfeatures]
    y = df_new['CLUSTER']

    params = {
        'max_depth': [3, 4, 6, 10, 20, None]
    }

    m = DecisionTreeClassifier()
    # m = RandomForestClassifier()

    m = GridSearchCV(m, params, cv=cv, iid=True)  # will return warning if 'idd' param not set to true

    try:
        m.fit(X, y)
    except ValueError:
        try:
            m = GridSearchCV(m, params, cv=2, iid=True)  # will return warning if 'idd' param not set to true
            m.fit(X, y)
            print(
                'ERROR SOLVED: n_splits=5 cannot be greater than the number of members in each class, then cv_split = 2',
                flush=True)
        except ValueError:
            print('ERROR: Feature columns have missing values! Please drop' \
                  ' rows or fill in missing data.', flush=True)
            # print('Warning: Feature Columns missing values!', flush=True)
            # df_new.dropna(inplace=True)
            # X = df_new.iloc[:, 2:2+pfeatures]
            # y = df_new['CLUSTER']
            # m = GridSearchCV(m, params, cv=1, iid=True) #will return warning if 'idd' param not set to true
            # m.fit(X, y)
    return m

def compute_state_target(test_init_state,
                         current_state_target,
                         transitions,
                         rewards,
                         all_data,
                         days_avg):
    test_init_state = test_init_state.to_frame()
    for region_id in test_init_state.index:
        last_date = test_init_state.loc[region_id, "TIME"]
        try:
            current_cluster = current_state_target.loc[region_id, "CLUSTER"]
            current_target = current_state_target.loc[region_id, "TARGET"]
            current_date = current_state_target.loc[region_id, "TIME"]
        except KeyError:
            continue  # to update
        previous_cluster = current_cluster
        previous_target = current_target
        for date in pd.date_range(start=current_date, end=last_date, freq='%sD' %days_avg):
            previous_cluster = current_cluster
            previous_target = current_target
            action = all_data.loc[(region_id, date), "ACTION"]
            current_cluster = transitions.loc[(current_cluster, action), "TRANSITION_CLUSTER"]
            current_target *= np.exp(rewards.loc[current_cluster, "EST_RISK"])

        test_init_state.loc[region_id, "INIT_TARGET"] = previous_target
        test_init_state.loc[region_id, "EST_H_NEXT_CLUSTER"] = previous_cluster

    return test_init_state

# predict_value_of_cluster() takes in MDP parameters, a cluster label, and
# and a list of actions, and returns the predicted value of the given cluster
def predict_value_of_cluster(P_df, R_df,  # df: MDP parameters
                             cluster,  # int: cluster number
                             actions):  # list: list of actions
    s = cluster
    v = R_df.loc[s]
    for a in actions:
        s = P_df.loc[s, a].values[0]
        v = v + R_df.loc[s]
    return v


def complete_P_df(df, P_df, actions, pfeatures, OutputFlag=0):

    risk_model_dict = dict()
    n_nan_actions = P_df["TRANSITION_CLUSTER"].isnull().groupby("ACTION").sum()

    for action in actions:
        if n_nan_actions[action] == 0:
            continue
        X = df[(df.ACTION == action) & (df.NEXT_CLUSTER.notnull())].iloc[:, 2: pfeatures + 2].values
        y = df[(df.ACTION == action) & (df.NEXT_CLUSTER.notnull())]["NEXT_CLUSTER"].values

        params = {
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        }

        m = KNeighborsClassifier(n_neighbors=1)
        m = GridSearchCV(m, params, cv=3, iid=True)  # will return warning if 'idd' param not set to true

        try:
            m.fit(X, y)
            risk_model_dict[action] = (m, Counter(y).most_common(1)[0][0])
        except ValueError:
            try:
                m = GridSearchCV(m, params, cv=2, iid=True)  # will return warning if 'idd' param not set to true
                m.fit(X, y)
                risk_model_dict[action] = (m, Counter(y).most_common(1)[0][0])
            except ValueError:
                risk_model_dict[action] = (None, Counter(y).most_common(1)[0][0])
                if OutputFlag >=2:
                    print('ERROR: Fitting KNN for action {}'.format(action), flush=True)

    P_df.reset_index(inplace=True)
    for action in actions:
        if n_nan_actions[action] == 0:
            continue
        missing_clusters = P_df[(P_df.ACTION == action) & (P_df.TRANSITION_CLUSTER.isna())].CLUSTER
        for idx, cluster in missing_clusters.items():
            try:
                nc = df[(df.ACTION == action) & (df.CLUSTER == cluster)].iloc[:, 2: pfeatures + 2].values
                nc = np.argmax(risk_model_dict[action][0].predict(nc))
            except:
                nc = risk_model_dict[action][1]
            P_df.iloc[idx, 2] = nc

    return P_df.set_index(["CLUSTER", "ACTION"])


# get_MDP() takes in a clustered dataframe df_new, and returns dataframes
# P_df and R_df that represent the parameters of the estimated MDP
def get_MDP(df_new, actions, pfeatures, n_cluster, complete=True, OutputFlag=0):
    # removing None values when counting where clusters go
    # df0 = df_new[df_new['NEXT_CLUSTER'] != 'None']
    try:
        # Robust transition cluster : best worst case error (existing cluster)
        transition_df = df_new[~(df_new["CLUSTER"] == (n_cluster-1))].dropna(subset=['NEXT_CLUSTER']).groupby(['CLUSTER', 'ACTION', 'NEXT_CLUSTER'])[
        'EST_H_ERROR'].max().groupby(['CLUSTER', 'ACTION']).idxmin()

        # Max next cluster appearance (new cluster)
        transition_df_ = df_new[(df_new["CLUSTER"] == (n_cluster-1))].dropna(subset=['NEXT_CLUSTER']).groupby(['CLUSTER', 'ACTION', 'NEXT_CLUSTER'])[
            'RISK'].count().groupby(['CLUSTER', 'ACTION']).idxmax()

        transition_df = pd.concat([transition_df, transition_df_]).copy()
        transition_df.name = "RISK"

    except KeyError:
        # next cluster given how most datapoints transition for the given action
        transition_df = df_new.dropna(subset=['NEXT_CLUSTER']).groupby(['CLUSTER', 'ACTION', 'NEXT_CLUSTER'])[
            'RISK'].count()
        transition_df = transition_df.groupby(['CLUSTER', 'ACTION']).idxmax()

    transition_df = transition_df.dropna()

    # P_df = pd.DataFrame()
    P_df = pd.DataFrame(index=pd.MultiIndex.from_tuples(product(range(n_cluster), actions),
                                                        names=['CLUSTER', 'ACTION']))
    # DEBUG 07132020 : NaN in P_df
    P_df['NEXT_CLUSTER'] = P_df.join(transition_df.apply(lambda x: np.nan if pd.isna(x) else x[2]),
                                     how="left").values
    P_df.columns = ['TRANSITION_CLUSTER']
    # P_df['NEXT_CLUSTER'] = P_df.apply(lambda x: complete_p_df(x, P_df), axis=1,)
    R_df = pd.DataFrame(index=pd.Index(range(n_cluster), name="CLUSTER"))

    # DEBUD mean --> min (conservative predictions)
    R_df = R_df.join(df_new.groupby('CLUSTER')['RISK'].mean())
    R_df.columns = ['EST_RISK']

    if complete:
        P_df = complete_P_df(df_new, P_df, actions, pfeatures, OutputFlag).copy()
    return P_df, R_df


# def get_MDP(df_new, n_cluster, actions, pfeatures, complete=True):
#     # removing None values when counting where clusters go
#     # df0 = df_new[df_new['NEXT_CLUSTER'] != 'None']
#     transition_df = df_new.dropna(subset=['NEXT_CLUSTER']).groupby(['CLUSTER', 'ACTION', 'NEXT_CLUSTER'])[
#         'RISK'].count()
#
#     # next cluster given how most datapoints transition for the given action
#     transition_df = transition_df.groupby(['CLUSTER', 'ACTION']).idxmax()
#     transition_df = transition_df.dropna()
#     # P_df = pd.DataFrame()
#     P_df = pd.DataFrame(index=pd.MultiIndex.from_tuples(product(range(n_cluster+1), actions),
#                                                         names=['CLUSTER', 'ACTION']))
#     # DEBUG 07132020 : NaN in P_df
#     P_df['NEXT_CLUSTER'] = P_df.join(transition_df.apply(lambda x: np.nan if pd.isna(x) else x[2]),
#                                      how="left").values
#     # P_df['NEXT_CLUSTER'] = P_df.apply(lambda x: complete_p_df(x, P_df), axis=1,)
#     R_df = pd.DataFrame(index=pd.Index(range(n_cluster+1), name="CLUSTER"))
#     R_df = R_df.join(df_new.groupby('CLUSTER')['RISK'].mean())
#
#     # complete the P_df
#     P_df = complete_P_df(df_new, P_df, actions, pfeatures).copy()
#     return P_df, R_df

# Auxiliary function for deployment
# predict_region_date() takes a given state and a date and returns the predicted target_colname
def predict_region_date(mdp,  # MDP_model object
                        region_last_date,  # tuple (region, last_date), e.g (Alabama, Timestamp('2020-03-24 00:00:00'), Timestamp('2020-06-22 00:00:00'))
                        date,  # target_colname date for prediciton, e.g. (Timestamp('2020-05-24 00:00:00'))
                        model_key="k_opt",
                        from_first=False,
                        verbose=0):
    region, last_date = region_last_date
    try:
        date = datetime.strptime(date, '%Y-%m-%d')
    except TypeError:
        pass

    # Case 1 : the input date occurs before the first available date for a given region
    try:
        assert date >= last_date
        n_days = (date - last_date).days
        return mdp.predict_region_ndays(region, n_days, from_first=from_first, model_key=model_key)
    except AssertionError:
        if verbose >= 1:
            print(
                "Prediction Error type I ('{}', '{}'): the input occurs before the last available ('{}') date of the training set".format(
                    region,
                    str(date),
                    str(last_date)))
        raise MDPPredictionError  # test


def prediction_score(mdp, testing_data):
    mdp.verbose = 0
    testing_data["{}_pred".format(mdp.target_colname)] = \
        testing_data.apply(lambda row: predict_region_date(mdp,
                                                           (row[mdp.region_colname],
                                                            mdp.df_trained.loc[row[mdp.region_colname], "TIME"]),
                                                           row[mdp.date_colname]),
                           axis=0)
    errors = mape_(y_pred=testing_data[mdp.target_colname], y_true=testing_data[mdp.target_colname + "_pred"])

    # evaluate the 3rd quantile
    return errors.groupby(mdp.region_colname).mean().quantile(0.75)


#############################################################################
# Functions for Accuracy and Purity

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
    # print(accuracy)
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
def testing_accuracy(df_test,  # dataframe: testing data
                     df_new,  # dataframe: clustered on training data
                     model,  # function: output of predict_cluster
                     pfeatures):  # int: # of features

    clusters = get_predictions(df_new)

    test_clusters = model.predict(df_test.iloc[:, 2:2 + pfeatures])
    df_test['CLUSTER'] = test_clusters

    accuracy = clusters.loc[df_test['CLUSTER']].reset_index()['OG_CLUSTER'] \
               == df_test.reset_index()['OG_CLUSTER']
    # print(accuracy)
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
    su.columns = ['CLUSTER', 'Purity']
    return su.groupby('CLUSTER')['Purity'].max()


#############################################################################


#############################################################################
# Functions for Error

# training_value_error() takes in a clustered dataframe, and computes the
# E((\hat{v}-v)^2) expected error in estimating values (risk) given actions
# Returns a float of average value error per ID
# NEW VERSION
def training_value_error(splitter_dataframe,  # Outpul of algorithm
                         ):

    # TRANSITION ERROR FUNCTION

    # COMPUTE ERROR

    if splitter_dataframe.error_computing == "horizon":
        error_train = splitter_dataframe.df.dropna(subset=["EST_H_ERROR"]).groupby("ID").tail(splitter_dataframe.horizon).reset_index()
        error_train = error_train.groupby("ID")["EST_H_ERROR"].mean().mean() / splitter_dataframe.horizon

    elif splitter_dataframe.error_computing == "id":
        error_train = splitter_dataframe.df.dropna(subset=["EST_H_ERROR"]).groupby("ID")["EST_H_ERROR"].mean().mean() / splitter_dataframe.horizon

    else:  # exponential
        error_train = splitter_dataframe.df.dropna(subset=["EST_H_ERROR"]).groupby("ID").apply(lambda group: group["EST_H_ERROR"].ewm(alpha=splitter_dataframe.alpha).mean()).reset_index()
        error_train = error_train.groupby("ID")["EST_H_ERROR"].last().mean() / splitter_dataframe.horizon

    return error_train


# testing_value_error() takes in a dataframe of testing data, and dataframe of
# new clustered data, a model from predict_cluster function, and computes the
# expected value error given actions and a predicted initial cluster
# Returns a float of sqrt average value error per ID
def testing_value_error(test_splitter_dataframe):

    error_test = test_splitter_dataframe.df.dropna(subset=["EST_H_ERROR"]).groupby("ID").last().reset_index()
    return error_test.groupby("ID")["EST_H_ERROR"].mean().mean() / test_splitter_dataframe.horizon


def error_per_ID(df_test, model, pfeatures, P_df, R_df, OutputFlag=0, relative=False, h=5):
    try:
        df_test['CLUSTER'] = model.predict(df_test.iloc[:, 2:2 + pfeatures])
    except ValueError:
        print('ERROR: Feature columns have missing values! Please drop' \
              ' rows or fill in missing data.', flush=True)
        df_test['CLUSTER'] = model.predict(df_test.iloc[:, 2:2 + pfeatures])
    # except ValueError:
    # print('Warning: Feature Columns missing values!')
    # df_test.dropna(inplace=True)
    # model.predict(df_test.iloc[:, 2:2+pfeatures])
    # df_test['CLUSTER'] = model.predict(df_test.iloc[:, 2:2+pfeatures])
    E_v = 0
    df2 = df_test.reset_index()
    df2 = df2.groupby(['ID']).first()
    N_test = df2.shape[0]
    #    print(df2)

    V_true, V_estim, V_err, C_err, State = [], [], [], [], []

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
            H += 1
            try:
                df_test['ID'].loc[index + H + 1]
            except:
                break
            if df_test['ID'].loc[index + H] != df_test['ID'].loc[index + H + 1]:
                break

        t = max(0, H - h)
        s = df_test['CLUSTER'].loc[index + t]
        a = df_test['ACTION'].loc[index + t]
        v_true = df_test['RISK'].loc[index + t]
        v_estim = R_df.loc[s]
        # print(v_true,df_test['RISK'].loc[index+t], v_estim)
        t += 1
        # predicting path of each ID
        while cont:
            try:
                risk = df_test['RISK'].loc[index + t]
                v_true = v_true + risk
            except KeyError:
                if OutputFlag >= 3:
                    print('WARNING: In training value evaluation, for a given region there is only one observation')
                break
            try:
                s = P_df.loc[s, a].values[0]
            # error raises in case we never saw a given transition in the data
            except(TypeError, KeyError):
                if OutputFlag >= 3:
                    print('WARNING: In training value evaluation, trying to predict next state from state', s,
                          'taking action', a, ', but this transition is never seen in the data. Data point:', i, t)

            a = df_test['ACTION'].loc[index + t]
            v_estim = v_estim + R_df.loc[s]

            try:
                df_test['ID'].loc[index + t + 1]
            except:
                break
            if df_test['ID'].loc[index + t] != df_test['ID'].loc[index + t + 1]:
                break

            t += 1
        if relative:
            # E_v = E_v + ((math.exp(v_true)-math.exp(v_estim))/math.exp(v_true))**2
            E_v = E_v + np.abs((math.exp(v_true) - math.exp(v_estim)) / math.exp(v_true))
        #            print('new E_v', E_v)
        else:
            E_v = E_v + np.abs(math.exp(v_true) - math.exp(v_estim))

        State.append(df_test[df_test['ID'] == df_test['ID'].loc[index]]['ID'].iloc[0])
        V_true.append(v_true)
        V_estim.append(v_estim)
        V_err.append(np.abs(v_true - v_estim))
        C_err.append(np.abs((math.exp(v_true) - math.exp(v_estim)) / math.exp(v_true)))
    df_err = pd.DataFrame()
    df_err['ID'] = State
    df_err['v_true'] = V_true
    df_err['v_estim'] = V_estim
    df_err['v_err'] = V_err
    df_err['cases_rel_err'] = C_err

    E_v = (E_v / N_test)
    #    print('final E_v', E_v)
    # print('sqrt', np.sqrt(E_v))
    # return np.sqrt(E_v)
    return df_err, E_v


#############################################################################


#############################################################################
# Functions for R2 Values


# splitter version
# R2_value() takes a splitting_dataframe instance and return the R2
def R2_value(splitting_dataframe):

    risk_est_true = splitting_dataframe.df[["ID", "EST_H_NEXT_RISK", "TRUE_H_NEXT_RISK"]].dropna().groupby("ID").last()
    risk_est = risk_est_true["EST_H_NEXT_RISK"].values
    risk_true = risk_est_true["TRUE_H_NEXT_RISK"].values
    N = risk_est.shape[0]

    # return max(1- E_v/SS_tot,0)
    return 1 - ((risk_est-risk_true) ** 2).sum() / N / np.var(risk_true)


#############################################################################


#############################################################################
# Functions for Plotting

# plot_features() takes in a dataframe of two features, and plots the data
# to illustrate the noise in each original cluster
def plot_features(df):
    df.plot.scatter(x='FEATURE_1',
                    y='FEATURE_2',
                    c='OG_CLUSTER',
                    colormap='viridis')
    #    import seaborn as sns
    #    sns.pairplot(x_vars=["FEATURE_1"], y_vars=["FEATURE_2"], data=df, hue="OG_CLUSTER", height=5)
    plt.show(block=False)


# plot_path() takes in a trained df_new, state, an h value, and plots the path
# (by ratio: e^v) of the MDP versus the actual state, given a horizon of prediction h
def plot_path(df_new, df, state, h, pfeatures, plot=True, OutputFlag=0):
    state_df = show_state(df_new, df, state, pfeatures)
    H = state_df.shape[0]
    P_df, R_df = get_MDP(df_new)

    t = max(H - h, 0)

    v_true = state_df['r_t'][t:]
    v_estim = []

    s = state_df['CLUSTER'].iloc[t]
    s_seq = [s]
    # print('initial state', s)
    # a = state_df['ACTION'].iloc[i]
    v_estim.append(math.exp(R_df.loc[s]))
    t += 1
    while t < H:
        try:
            s = P_df.loc[s, 0].values[0]
            s_seq.append(s)
        except TypeError:
            if OutputFlag >= 2:
                print('WARNING: Trying to predict next state from state', s, 'taking action', a,
                      ', but this transition is never seen in the data. Data point:', i, t)
        # a = df_test['ACTION'].loc[index + t]
        v_estim.append(math.exp(R_df.loc[s]))
        t += 1

    v_estim = np.array(v_estim)
    # plt.plot(v_true)
    if plot:
        fig1, ax1 = plt.subplots()
        its = np.arange(-h, 0)
        ax1.plot(its, v_true, label="True Ratio")
        ax1.plot(its, v_estim, label="Predicted Ratio")
        ax1.set_title('%s True vs Predicted Ratios of Cases' % state)
        ax1.set_xlabel('Time Before Present')
        ax1.set_ylabel('Ratio of Cases')
        plt.legend()
        plt.show(block=False)
        # print('state sequence:', s_seq)

    E_v = sum(np.abs((v_estim - v_true) / v_estim)) / h
    # print("error", E_v)
    return E_v, v_true, v_estim, s_seq


# plot_path_all() returns the ratios and sequence of states for an optimal longest path
# if opt = True: Find optimal path and stop there
# if opt = False: plot the error horizon over different time horizons of predictions
def plot_path_all(df_new, df, state, pfeatures, opt=True, plot=True):
    state_df = show_state(df_new, df, state, pfeatures)
    H = state_df.shape[0]  # of datapoints
    errors = []
    prev = float('inf')

    for h in range(H, 0, -1):
        E_v, v_true, v_estim, s_seq = plot_path(df_new, df, state, h, pfeatures, plot)

        if opt and E_v > prev and h < 16:  # arbitrary threshold for a decent prediction
            break

        v_true_prev = v_true
        v_estim_prev = v_estim
        s_seq_prev = s_seq
        errors.append(E_v)
        prev = E_v

    if opt != True:
        fig2, ax2 = plt.subplots()
        its = np.arange(H, 0, -1)
        ax2.plot(its, errors)
        ax2.set_title('%s MAPE over different time horizons' % state)
        ax2.set_xlabel('Horizon of Prediction')
        ax2.set_ylabel('Error')
        plt.show(block=False)
        df_errors = pd.DataFrame(list(zip(its, errors)),
                                 columns=['h', 'Error'])
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
                          columns=['state', 'sequence', 'ratios', 'error'])
    return df_seq


# plot_pred() takes a trained model, a specific US state name, and the df_true
# (sorted by TIME), and plots the predicted versus true cases for n_days
def plot_pred(model, state, df_true, n_days, from_first=False, model_key="median"):
    if from_first:
        init_df_trained = model.df_trained_first
    else:
        init_df_trained = model.df_trained
    last_training_date = init_df_trained.loc[state, "TIME"]
    state_date_range = pd.date_range(start=last_training_date+timedelta(days=1), periods=n_days, freq="1D")
    state_predictions = model.predict([state], state_date_range, from_first=from_first, model_key=model_key)
    try:
        state_predictions = state_predictions[state]
        state_target = df_true[df_true[model.region_colname] == state][[model.date_colname, model.target_colname]].set_index(model.date_colname)

        fig, ax = plt.subplots()
        state_target.plot(label='True ' + model.target_colname, ax=ax)
        state_predictions.plot(label='Predicted ' + model.target_colname, ax=ax)
        ax.set_title('%s True vs Predicted ' % state + model.target_colname)
        ax.set_xlabel('Date')
        ax.set_ylabel(model.target_colname)
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.show(block=False)
    except:
        pass


# plot_pred() takes a trained model, a specific US state name, and the df_true
# (sorted by TIME), and plots the predicted versus true cases for n_days
def plot_pred_action(model, state, df_true, n_days, action_day=0, from_first=False):
    h = int(np.round(n_days / model.days_avg))
    action_adj = int(np.floor(action_day / model.days_avg))
    df_true.loc[:, [model.date_colname]] = pd.to_datetime(df_true[model.date_colname])
    if from_first:
        date = model.df_trained_first.loc[state, "TIME"]
        target = model.df_trained_first.loc[state, model.target_colname]
        s_init = model.df_trained_first.loc[state, "CLUSTER"]
    else:
        date = model.df_trained.loc[state, "TIME"]
        target = model.df_trained.loc[state, model.target_colname]
        s_init = model.df_trained.loc[state, "CLUSTER"]

    actions = [a - model.action_thresh[1] for a in range(len(model.action_thresh[0]) + 1)]
    fig, ax = plt.subplots()

    # prediction 0
    for a in actions:
        s = s_init
        dates = [date]
        targets_pred = [target]
        r = 1
        for i in range(h):
            dates.append(date + timedelta((i + 1) * model.days_avg))
            r = r * np.exp(model.R_df.loc[s])
            targets_pred.append(target * r)
            try:
                if i == action_adj:
                    s_bf = s
                    s = model.P_df.loc[s, a].values[0]
                    print("with action {}".format(a), " STATE bef:", s_bf, " STATE aft:", s)
                else:
                    s = model.P_df.loc[s, 0].values[0]
            except TypeError:
                print("Transition not found:", (s, a))
                break

        ax.plot(dates, targets_pred,
                label='Predicted ' + model.target_colname + ' with ACTION {} after {} days'.format(a, action_day))

    ax.plot(df_true.loc[df_true[model.region_colname] == state][model.date_colname], \
            df_true.loc[df_true[model.region_colname] == state][model.target_colname], \
            label='True ' + model.target_colname)
    ax.set_title('%s True vs Predicted ' % state + model.target_colname)
    ax.set_xlabel('Date')
    ax.set_ylabel(model.target_colname)
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.show(block=False)


def plot_pred_fact(model, state, df_w_act, starting_date, n_days=30, OutputFlag=0):

    fig, ax = plt.subplots()

    try :
        starting_date = datetime.strptime(starting_date, "%Y%m%d")
    except:
        pass

    df_state = df_w_act[df_w_act[model.region_colname] == state].reset_index().drop("index", axis=1)

    first_date = df_state["TIME"].min()

    starting_n_days = int((starting_date - first_date).days)
    try:
        assert starting_n_days >= 0
    except AssertionError:
        print(" the given starting_date {} occurs before the first date {} of the given data set".format(str(starting_date),
                                                                                                         str(first_date)))
        raise AssertionError

    h = int(np.round(n_days/model.days_avg))
    h_start = int(np.floor(starting_n_days/model.days_avg))

    date = df_state.iloc[h_start]["TIME"]
    target = df_state.iloc[h_start][model.target_colname]

    # predict the current state
    date_features = df_state.iloc[h_start, 2:(2+model.pfeatures)].values.reshape((1, -1))
    s_init = model.classifier.predict(date_features)[0]

    # prediction ACTION = 0
    s = s_init
    dates = [date]
    targets_pred = [target]
    r = 1.
    for i in range(h):
        dates.append(date + timedelta((i+1)*model.days_avg))
        r = r*np.exp(model.R_df.loc[s])
        targets_pred.append(target*r)
        try:
            s = model.P_df.loc[s, 0].values[0]
        except TypeError:
            print("Transition not found:", (s, 0))
            break

    ax.plot(dates, targets_pred, label='Predicted '+model.target_colname+ ' with NO ACTION')

    # prediction with adaptive ACTION
    s = s_init
    dates = [date]
    targets_pred = [target]
    r = 1.
    for i in range(h):
        dates.append(date + timedelta((i+1)*model.days_avg))
        r = r*np.exp(model.R_df.loc[s])
        targets_pred.append(target*r)

        try:
            a = df_state.iloc[(h_start+i)]["ACTION"]

        except:
            if OutputFlag >= 1:
                print("The next action at {} (after {} days) is not defined".format(date + timedelta((i+1)*model.days_avg), i))
            break

        if a != 0:
            print("{} : Action {}".format(dates[-1], a))
            ax.axvline(dates[-1], color='k', linestyle='--')
        try:
            s_bf = s
            s = model.P_df.loc[s, a].values[0]
            if a != 0:
                print("with action {}".format(a)," STATE bef:", s_bf, " STATE aft:", s)
        except TypeError:
            print("Transition not found:", (s, a))
            break

    ax.plot(dates, targets_pred, label='Predicted '+model.target_colname+ ' with ACTION')

    ax.plot(df_state["TIME"], \
            df_state[model.target_colname], \
            label = 'True '+model.target_colname)
    ax.set_title('%s True vs Predicted '%state + model.target_colname)
    ax.set_xlabel('Date')
    ax.set_ylabel(model.target_colname)
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.show(block=False)
#############################################################################


#############################################################################
def cluster_size(df):
    return df.groupby('CLUSTER')['RISK'].agg(['count', 'mean', 'std', 'min', 'max'])


def show_state(df_new, df, state, pfeatures):
    model = predict_cluster(df_new, pfeatures)
    st = df[df['state'] == state]
    st['CLUSTER'] = model.predict(st.iloc[:, 2:pfeatures + 2])
    return st[['TIME', 'cases', 'RISK', 'CLUSTER', 'r_t']]


def mape(df_pred, df_true, target_colname):
    df_pred['real ' + target_colname] = df_true[target_colname]
    df_pred['rel_error'] = abs(df_pred[target_colname] - df_true[target_colname]) / df_true[target_colname]
    return df_pred


def mape_(y_pred, y_true):
    return abs(y_pred - y_true) / y_true
#############################################################################


if __name__ == "__main__":
    from codes.data_utils import (load_data, load_model)
    from codes.params import validation_cutoff
    from codes.mdp_model import MDPModel, MDPGridSearch
    import warnings
    warnings.filterwarnings("ignore")
    df, _, _ = load_data(validation_cutoff=validation_cutoff)
    n_days = 15

    mdp_file = r"C:\Users\david\Desktop\MIT\Courses\Research internship\results\22 - 20200822 - Massachusetts with Boosted MDP new pred\MDPs_without_actions\TIME_CV\mdp__target_cases__h5__davg7__cdt_8pct__n_iter500__ClAlg_Rando__errhoriz_cv4_nbfs3\mdp_20200801_cases_state.pkl"
    mdp = load_model(mdp_file)
    plot_pred(mdp, "California", df, n_days, model_key="median")
