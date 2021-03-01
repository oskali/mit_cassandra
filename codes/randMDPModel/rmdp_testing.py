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
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
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
                    features,
                    cv=4):  # int: # of features
    X = df_new.loc[~df_new.NEXT_CLUSTER.isnull(), features]
    y = df_new.loc[~df_new.NEXT_CLUSTER.isnull(), 'CLUSTER']

    params = {
        'max_depth': [6, 7, 10, 15, 20, 30, 40],
        # 'ccp_alpha': np.logspace(-2, 0, 30)
    }

    m = DecisionTreeClassifier()
    # m = RandomForestClassifier()

    m = RandomizedSearchCV(m, params, cv=cv, iid=True, n_iter=30)  # will return warning if 'idd' param not set to true

    try:
        m.fit(X, y)
    except ValueError:
        try:
            m = DecisionTreeClassifier()
            m = RandomizedSearchCV(m, params, cv=2, iid=True)  # will return warning if 'idd' param not set to true
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


def compute_state_target_risk(test_init_state,
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

            for date in pd.date_range(start=current_date, end=last_date, freq='%sD' %days_avg)[1:]:
                action = all_data.loc[(region_id, date), "ACTION"]
                current_cluster = transitions.loc[(current_cluster, action), "TRANSITION_CLUSTER"]
                current_target *= np.exp(rewards.loc[current_cluster, "EST_RISK"])

            test_init_state.loc[region_id, "INIT_TARGET"] = current_target
            test_init_state.loc[region_id, "EST_H_NEXT_CLUSTER"] = current_cluster

        except KeyError:
            continue  # to update

    return test_init_state


def compute_state_target_alpha(test_init_state,
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
            current_risk = current_state_target.loc[region_id, "RISK"]
            current_date = current_state_target.loc[region_id, "TIME"]
        except KeyError:
            continue  # to update
        for date in pd.date_range(start=current_date, end=last_date, freq='%sD' %days_avg)[1:]:
            try:
                action = all_data.loc[(region_id, date), "ACTION"]
            except:
                break
            current_cluster = transitions.loc[(current_cluster, action), "TRANSITION_CLUSTER"]
            current_risk *= np.exp(rewards.loc[current_cluster, "EST_AlphaRISK"])
            current_target *= np.exp(current_risk)

        test_init_state.loc[region_id, "INIT_TARGET"] = current_target
        test_init_state.loc[region_id, "INIT_RISK"] = current_risk
        test_init_state.loc[region_id, "EST_H_NEXT_CLUSTER"] = current_cluster

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

# def complete_P_df(df, P_df, actions, features, OutputFlag=0):

#     P_df.reset_index(inplace=True)
#     for _ in 
#     for action in actions:
#         if n_nan_actions[action] == 0:
#             continue
#         missing_clusters = P_df[(P_df.ACTION == action) & (P_df.TRANSITION_CLUSTER.isna())].CLUSTER
#         for idx, cluster in missing_clusters.items():
#             try:
#                 nc = df[(df.ACTION == action) & (df.CLUSTER == cluster)].loc[:, features].values
#                 nc = np.argmax(risk_model_dict[action][0].predict(nc))
#             except:
#                 nc = risk_model_dict[action][1]
#             P_df.iloc[idx, 2] = nc

#     return P_df.set_index(["CLUSTER", "ACTION"])

def complete_P_df(df, P_df, actions, features, OutputFlag=0):

    for _ in P_df[P_df.TRANSITION_CLUSTER.isna()].index:
        P_df.loc[_, "TRANSITION_CLUSTER"] = _[0]

    return P_df

# def complete_P_df(df, P_df, actions, features, OutputFlag=0):

#     risk_model_dict = dict()
#     n_nan_actions = P_df["TRANSITION_CLUSTER"].isnull().groupby("ACTION").sum()

#     for action in actions:
#         if n_nan_actions[action] == 0:
#             continue
#         X = df[(df.ACTION == action) & (df.NEXT_CLUSTER.notnull())].loc[:, features].values
#         y = df[(df.ACTION == action) & (df.NEXT_CLUSTER.notnull())]["NEXT_CLUSTER"].values

#         params = {
#             'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
#         }

#         m = KNeighborsClassifier(n_neighbors=1)
#         m = GridSearchCV(m, params, cv=3, iid=True)  # will return warning if 'idd' param not set to true

#         try:
#             m.fit(X, y)
#             risk_model_dict[action] = (m, Counter(y).most_common(1)[0][0])
#         except ValueError:
#             try:
#                 m = GridSearchCV(m, params, cv=2, iid=True)  # will return warning if 'idd' param not set to true
#                 m.fit(X, y)
#                 risk_model_dict[action] = (m, Counter(y).most_common(1)[0][0])
#             except ValueError:
#                 risk_model_dict[action] = (None, Counter(y).most_common(1)[0][0])
#                 if OutputFlag >=2:
#                     print('ERROR: Fitting KNN for action {}'.format(action), flush=True)

#     P_df.reset_index(inplace=True)
#     for action in actions:
#         if n_nan_actions[action] == 0:
#             continue
#         missing_clusters = P_df[(P_df.ACTION == action) & (P_df.TRANSITION_CLUSTER.isna())].CLUSTER
#         for idx, cluster in missing_clusters.items():
#             try:
#                 nc = df[(df.ACTION == action) & (df.CLUSTER == cluster)].loc[:, features].values
#                 nc = np.argmax(risk_model_dict[action][0].predict(nc))
#             except:
#                 nc = risk_model_dict[action][1]
#             P_df.iloc[idx, 2] = nc

#     return P_df.set_index(["CLUSTER", "ACTION"])


# get_MDP() takes in a clustered dataframe df_new, and returns dataframes
# P_df and R_df that represent the parameters of the estimated MDP
def get_MDP(df_new, actions, features, n_cluster, complete=True, reward='AlphaRISK', OutputFlag=0):

    transition_df = df_new.dropna(subset=['NEXT_CLUSTER']).groupby(['CLUSTER', 'ACTION', 'NEXT_CLUSTER']).size()
    transition_df = transition_df.groupby(['CLUSTER', 'ACTION']).idxmax()
    transition_df = transition_df.dropna()
    transition_df.name = "TRANSITION"

    # P_df = pd.DataFrame()
    P_df = pd.DataFrame(index=pd.MultiIndex.from_tuples(product(range(n_cluster), actions),
                                                        names=['CLUSTER', 'ACTION']))
    # DEBUG 07132020 : NaN in P_df
    P_df['TRANSITION_CLUSTER'] = P_df.join(transition_df.apply(lambda x: np.nan if pd.isna(x) else x[2]),
                                     how="left").values

    R_df = pd.DataFrame(index=pd.Index(range(n_cluster), name="CLUSTER"))

    # DEBUG mean --> min (conservative predictions)
    # df_new[reward] = np.exp(df_new[reward])
    # R_df = R_df.join(np.log(df_new.groupby('CLUSTER')[reward].mean()))
    # df_new[reward] = np.log(df_new[reward])
    R_df = R_df.join(df_new.groupby('CLUSTER')[reward].mean())
    R_df.columns = ['EST_{}'.format(reward)]

    if complete:
        P_df = complete_P_df(df_new, P_df, actions, features, OutputFlag).copy()
    return P_df, R_df


# Auxiliary function for deployment
# predict_region_date() takes a given state and a date and returns the predicted target_colname
def predict_region_date(mdp,  # MDP_model object
                        region_last_date,  # tuple (region, last_date), e.g (Alabama, Timestamp('2020-03-24 00:00:00'), Timestamp('2020-06-22 00:00:00'))
                        date,  # target_colname date for prediciton, e.g. (Timestamp('2020-05-24 00:00:00'))
                        model_key="k_opt",
                        from_first=False,
                        agg=False,
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
        if mdp.reward_name == "RISK":
            return mdp.predict_region_ndays(region, n_days, from_first=from_first, agg=agg, model_key=model_key)
        elif mdp.reward_name == "AlphaRISK":
            return mdp.predict_region_ndays_alpha(region, n_days, from_first=from_first, agg=agg, model_key=model_key)
        else:
            raise MDPPredictionError("Unknown reward name : {}".format(mdp.reward_name))
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


# [Deprecated]
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

# Additional functions
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


def compute_exp_mean_alpha(v, alphas):
   v = np.array([_[1] for _ in v])
   v = v[~np.isnan(v)]
   std_v = np.std(v)
   mean_v = np.mean(v)
   aux = lambda alpha: np.average(v, weights=np.maximum(np.exp(- alpha * (v - mean_v) / std_v), 1e-6))
   aux = np.vectorize(aux)
   try:
       return aux(alphas)
   except:
       _ = np.empty_like(alphas)
       _[:] = np.nan
       return _


def compute_exp_sigmoid_alpha(v, alphas):
   v = np.array([_[1] for _ in v])
   v = v[~np.isnan(v)]
   std_v = np.std(v)
   mean_v = np.mean(v)
   aux = lambda alpha: np.average(v, weights=np.maximum(np.exp(- alpha * (v - mean_v) / std_v) / (1. + np.exp(- alpha * (v - mean_v) / std_v)), 1e-6))
   aux = np.vectorize(aux)
   return aux(alphas)

#############################################################################


if __name__ == "__main__":
    pass
