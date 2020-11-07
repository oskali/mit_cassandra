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

@author: Amine, omars, david
"""

#%% Libraries

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from datetime import timedelta
import math
from sklearn.isotonic import IsotonicRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from itertools import product
from collections import Counter
from itertools import combinations, permutations
import statsmodels.api as sm
import statsmodels.formula.api as smf

#%% New Exception for MDP learning
class MDPPredictionError(Exception):
    pass


class MDPTrainingError(Exception):
    pass


def complete_P_df_(df, P_df, R_df, actions, pfeatures, OutputFlag=0, relative=False):

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
                try:
                    risk_model_dict[action] = (None, Counter(y).most_common(1)[0][0])
                except IndexError:
                    if OutputFlag >= 2:
                        print('ERROR: Unobserved action in the dataset ({})'.format(action), flush=True)
                    risk_model_dict[action] = (None, np.nan)
                    continue
                if OutputFlag >= 2:
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

    #
    clusters = P_df.CLUSTER.unique()
    P_df = P_df.set_index(["CLUSTER", "ACTION"])

    if relative:
        for cluster in clusters:

            if R_df.loc[P_df.loc[(cluster, 1)]].values[0, 0] < R_df.loc[P_df.loc[(cluster, 0)]].values[0,0]:
                P_df.loc[(cluster, 1)] = P_df.loc[(cluster, 0)]

            if R_df.loc[P_df.loc[(cluster, -1)]].values[0, 0] > R_df.loc[P_df.loc[(cluster, 0)]].values[0,0]:
                P_df.loc[(cluster, -1)] = P_df.loc[(cluster, 0)]
    return P_df


def complete_P_df(df, P_df, R_df, actions, pfeatures, OutputFlag=0, relative=False):

    risk_model_dict = dict()
    n_nan_actions = P_df["TRANSITION_CLUSTER"].isnull().groupby("ACTION").sum()

    # action 0
    for s in R_df.index:
        try:
            assert not np.isnan(P_df.loc[(s, 0)].values[0])
        except:
            P_df.loc[(s, 0)] = s

    l = 4
    R_df_l = pd.DataFrame()

    for init_c in R_df.index:
        s = init_c
        r = 0
        try:
            for i in range(l+1):
                rew = R_df.loc[s].values[0]
                assert not np.isnan(rew)
                r += rew
                s = P_df.loc[(s, 0)].values[0]
                assert not np.isnan(s)
                R_df_l.at[init_c, "R_{}".format(i)] = r
        except:
            print(init_c)

    action_clusters = P_df.loc[P_df["TRANSITION_CLUSTER"].isnull()]
    for state, action in action_clusters.index:
        if action > 0:
            potential_nc = set(R_df_l[R_df_l >= R_df_l.loc[P_df.loc[(state, 0)].values[0]]].dropna(axis=0).index.tolist())
            if not potential_nc:
                risk_model_dict[(state, action)] = (None, np.nan)
                continue
            X = df[((df.ACTION == action) | (df.ACTION == action-1)) & (df.NEXT_CLUSTER.isin(potential_nc))].iloc[:, 2: pfeatures + 3].values
            y = df[((df.ACTION == action) | (df.ACTION == action-1)) & (df.NEXT_CLUSTER.isin(potential_nc))]["NEXT_CLUSTER"].values

        else:
            potential_nc = set(R_df_l[R_df_l <= R_df_l.loc[P_df.loc[(s, 0)].values[0]]].dropna(axis=0).index.tolist())
            if not potential_nc:
                risk_model_dict[(state, action)] = (None, np.nan)
                continue
            X = df[((df.ACTION == action) | (df.ACTION == action+1)) & (df.NEXT_CLUSTER.isin(potential_nc))].iloc[:, 2: pfeatures + 3].values
            y = df[((df.ACTION == action) | (df.ACTION == action+1)) & (df.NEXT_CLUSTER.isin(potential_nc))]["NEXT_CLUSTER"].values

        params = {
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        }

        m = KNeighborsClassifier(n_neighbors=1)
        m = GridSearchCV(m, params, cv=3, iid=True)  # will return warning if 'idd' param not set to true

        try:
            m.fit(X, y)
            risk_model_dict[(state, action)] = (m, Counter(y).most_common(1)[0][0])
        except ValueError:
            try:
                m = GridSearchCV(m, params, cv=2, iid=True)  # will return warning if 'idd' param not set to true
                m.fit(X, y)
                risk_model_dict[(state, action)] = (m, Counter(y).most_common(1)[0][0])
            except ValueError:
                try:
                    risk_model_dict[(state, action)] = (None, Counter(y).most_common(1)[0][0])
                except IndexError:
                    if OutputFlag >= 2:
                        print('ERROR: Unobserved action in the dataset ({})'.format(action), flush=True)
                    risk_model_dict[(state, action)] = (None, np.nan)
                    continue
                if OutputFlag >= 2:
                    print('ERROR: Fitting KNN for action {}'.format(action), flush=True)

    for cluster, action in action_clusters.index:
        nc = risk_model_dict[(cluster, action)][1]
        if np.isnan(nc):
            nc = P_df.loc[(cluster, 0)].values[0]

        P_df.loc[(cluster, action)] = nc

    if relative:
        #
        clusters = P_df.reset_index().CLUSTER.unique()
        for cluster in clusters:

            if R_df.loc[P_df.loc[(cluster, 1)]].values[0, 0] < R_df.loc[P_df.loc[(cluster, 0)]].values[0, 0]:
                P_df.loc[(cluster, 1)] = P_df.loc[(cluster, 0)]

            if R_df.loc[P_df.loc[(cluster, -1)]].values[0, 0] > R_df.loc[P_df.loc[(cluster, 0)]].values[0, 0]:
                P_df.loc[(cluster, -1)] = P_df.loc[(cluster, 0)]
    return P_df


def compute_relative_R_df(df, actions, OutputFlag=0, lag=5):

    action_risk = df[["ID", "RISK", "ACTION"]].copy()
    action_risk["RISK_RATIO"] = action_risk.groupby("ID")["RISK"].rolling(window=lag, min_period=lag).mean().values
    action_risk["RISK_RATIO"] = action_risk.groupby("ID")["RISK_RATIO"].shift(-lag).values / action_risk["RISK"]
    action_risk["RISK_RATIO"] = action_risk["RISK_RATIO"].replace([np.inf, -np.inf], np.nan)
    action_risk.dropna(subset=["RISK_RATIO"], inplace=True)

    action_ratio_up = action_risk.groupby(by=["ACTION", "ID"])["RISK_RATIO"].quantile(0.6).reset_index()
    action_ratio_up = action_ratio_up.pivot(index="ID", columns=["ACTION"], values=["RISK_RATIO"])
    action_ratio_down = action_risk.groupby(by=["ACTION", "ID"])["RISK_RATIO"].quantile(0.4).reset_index()
    action_ratio_down = action_ratio_down.pivot(index="ID", columns=["ACTION"], values=["RISK_RATIO"])

    action_ratio = pd.DataFrame()
    # and length 2
    comb = permutations(actions, 2)

    # Print the obtained combinations
    for a1, a2 in list(comb):
        try:
            sample = pd.DataFrame()
            sample["Y"] = action_ratio_up[("RISK_RATIO", a1)]
            sample["X"] = action_ratio_down[("RISK_RATIO", a2)]
            sample.dropna(inplace=True)
            sample = sample[(a1 - a2) * (sample.Y / sample.X - 1.) > 0]
            mod = smf.quantreg('Y ~ X -1', sample).fit(q=0.5)
            action_ratio.at[a1, a2] = mod.params[0]
            # R(s, a1) = default_R(a1, a2) * R(s, a2)
        except:
            action_ratio.at[a1, a2] = 1.
            continue

    action_ratio.sort_index(inplace=True)
    action_ratio = action_ratio.loc[:, actions]

    # Isotonic Regression
    for action in actions:
        est_ratio = action_ratio[action].reset_index().dropna()
        mod = IsotonicRegression()
        mod.fit(est_ratio["index"], est_ratio[action])
        est_ratio_iso = mod.predict(est_ratio["index"])
        for idx, action_bis in enumerate(est_ratio["index"]):
            action_ratio.at[action_bis, action] = est_ratio_iso[idx]

    return action_ratio

#%% MDP Transition functions and class
def CompletedTransition(mdp_trans, s, a=None):
    try:
        assert not (a is None)
        s_next = mdp_trans.P_df.loc[(s, a)].values[0]
        r_next = mdp_trans.R_df.loc[s_next].values[0]
        return s_next, r_next
    except AssertionError:
        return mdp_trans.R_df.loc[s].values[0]


def kNNTransition(mdp_trans, s, a=None):
    l = 4
    R_df_l = pd.DataFrame()

    for init_c in mdp_trans.index:
        s = init_c
        r = 0
        try:
            for i in range(l+1):
                rew = mdp_trans.loc[s].values[0]
                assert not np.isnan(rew)
                r += rew
                s = mdp_trans.loc[(s, 0)].values[0]
                assert not np.isnan(s)
                R_df_l.at[init_c, "R_{}".format(i)] = r
        except:
            print(init_c)

    try:
        assert not (a is None)
        try:
            s_next = mdp_trans.P_df.loc[(s, a)].values[0]
            r_next = mdp_trans.R_df.loc[s_next].values[0]
            return s_next, r_next
        except(KeyError, IndexError):
            a_actions = [_ for _ in mdp_trans.actions if _ != a]
            a_actions = sorted(a_actions, key=lambda x: (np.abs(x - a), a - x))
            for action in a_actions:
                try:
                    s_next_alt = mdp_trans.P_df.loc[(s, action)].values[0]
                    r_next_alt = mdp_trans.R_df.loc[s_next_alt].values[0]
                    assert not (np.isnan(s_next_alt) | np.isnan(r_next_alt))

                    adj_r = mdp_trans.relative_R_df.loc[a, action]
                    return s_next_alt, adj_r * r_next_alt  # or s?
                except(KeyError, IndexError, AssertionError):
                    continue
            raise MDPTrainingError("No existing transition found : ({}, {})".format(s, a))
    except AssertionError:
        return mdp_trans.R_df.loc[s].values[0]


def RelativeTransition(mdp_trans, s, a=None):
    try:
        assert not (a is None)
        try:
            s_next = mdp_trans.P_df.loc[(s, a)].values[0]
            r_next = mdp_trans.R_df.loc[s_next].values[0]
            return s_next, r_next
        except(KeyError, IndexError):
            a_actions = [_ for _ in mdp_trans.actions if _ != a]
            a_actions = sorted(a_actions, key=lambda x: (np.abs(x - a), a - x))
            for action in a_actions:
                try:
                    s_next_alt = mdp_trans.P_df.loc[(s, action)].values[0]
                    r_next_alt = mdp_trans.R_df.loc[s_next_alt].values[0]
                    assert not (np.isnan(s_next_alt) | np.isnan(r_next_alt))

                    adj_r = mdp_trans.relative_R_df.loc[a, action]
                    return s_next_alt, adj_r * r_next_alt  # or s?
                except(KeyError, IndexError, AssertionError):
                    continue
            raise MDPTrainingError("No existing transition found : ({}, {})".format(s, a))
    except AssertionError:
        return mdp_trans.R_df.loc[s].values[0]


class MDPTransition:
    def __init__(self, pfeatures, actions, completion_algorithm="bias_completion", verbose=0):
        self.P_df = None
        self.R_df = None
        self.completion_algorithm = completion_algorithm
        self.n_cluster = None
        self.relative_R_df = None
        self.pfeatures = pfeatures
        self.actions = actions
        self.verbose = verbose

        if self.completion_algorithm == "bias_completion":
            self.completion_function = CompletedTransition
        elif self.completion_algorithm == "relative_completion":
            self.completion_function = RelativeTransition
        elif self.completion_algorithm == "knn_completion":
            self.completion_function = CompletedTransition
        else:
            self.completion_function = CompletedTransition

    def __call__(self, s, a=None):
        # return the next state and reward if action != None
        # return current reward if action == None
        return self.completion_function(self, s, a)

    def update(self, df_new, n_cluster, adjust=False, pre_complete=False):
        P_df, R_df = get_MDP(df_new, actions=self.actions, n_cluster=n_cluster,  OutputFlag=self.verbose)
        if pre_complete:
            P_df = precomplete_transition(P_df, self.actions, n_cluster, window=2).copy()
        if adjust:
            for cluster in range(n_cluster):

                # find a default action
                actions_list = [_ for _ in self.actions]
                a_0 = 0
                while True:
                    try:
                        r1 = R_df.loc[P_df.loc[(cluster, a_0)]].values[0, 0]
                        nc1 = P_df.loc[(cluster, 0)].copy()
                        break
                    except:
                        actions_list.remove(a_0)
                        a_0 = actions_list[0]

                # complete actions that are above the default action
                a_0up = a_0
                for a_up in range(a_0+1, self.actions[-1]+1):
                    # a1 < a2 anyway
                    try:
                        if R_df.loc[P_df.loc[(cluster, a_up)]].values[0, 0] < R_df.loc[P_df.loc[(cluster, a_0up)]].values[0, 0]:
                                P_df.loc[(cluster, a_up)] = P_df.loc[(cluster, a_0up)].copy()
                        else:
                            a_0up = a_up
                    except:
                        continue

                # complete actions that are below the default action
                a_0down = a_0
                for a_down in range(a_0-1, self.actions[0]-1, -1):
                    # a1 < a2 anyway
                    try:
                        if R_df.loc[P_df.loc[(cluster, a_0down)]].values[0, 0] < R_df.loc[P_df.loc[(cluster, a_down)]].values[0, 0]:
                                P_df.loc[(cluster, a_down)] = P_df.loc[(cluster, a_0down)].copy()
                        else:
                            a_0down = a_down
                    except:
                        continue

        self.P_df = P_df
        self.R_df = R_df

        if self.completion_algorithm == "bias_completion":
            self.P_df = complete_P_df(df_new, self.P_df, self.R_df, self.actions, self.pfeatures, OutputFlag=self.verbose, relative=True).copy()

        if self.completion_algorithm == "knn_completion":
            self.P_df = complete_P_df(df_new, self.P_df, self.R_df, self.actions, self.pfeatures, OutputFlag=self.verbose, relative=False).copy()

        if self.completion_algorithm == "unbias_completion":
            self.P_df = complete_P_df(df_new, self.P_df, self.R_df, self.actions, self.pfeatures, OutputFlag=self.verbose, relative=False).copy()

        if self.completion_algorithm == "relative_completion":
            if self.relative_R_df is None:
                self.relative_R_df = compute_relative_R_df(df_new, self.actions, OutputFlag=self.verbose).copy()


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
                    pfeatures,
                    cv=5): # int: # of features
    X = df_new.iloc[:, 2:2+pfeatures]
    y = df_new['CLUSTER']

    params = {
    'max_depth': [3, 4, 6, 10, None]
    }

    m = DecisionTreeClassifier()

    m = GridSearchCV(m, params, cv=cv, iid=True) #will return warning if 'idd' param not set to true

    try:
        m.fit(X, y)
    except ValueError:
        try:
            m = GridSearchCV(m, params, cv=2, iid=True) #will return warning if 'idd' param not set to true
            m.fit(X, y)
            print('ERROR SOLVED: n_splits=5 cannot be greater than the number of members in each class, then cv_split = 2', flush=True)
        except ValueError:
            print('ERROR: Feature columns have missing values! Please drop' \
                  ' rows or fill in missing data.', flush=True)
            #print('Warning: Feature Columns missing values!', flush=True)
            #df_new.dropna(inplace=True)
            #X = df_new.iloc[:, 2:2+pfeatures]
            #y = df_new['CLUSTER']
            # m = GridSearchCV(m, params, cv=1, iid=True) #will return warning if 'idd' param not set to true
            # m.fit(X, y)
    return m


# predict_value_of_cluster() takes in MDP parameters, a cluster label, and
# and a list of actions, and returns the predicted value of the given cluster
def predict_value_of_cluster(P_df, R_df, # df: MDP parameters
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
def get_MDP(df_new, actions, n_cluster, OutputFlag=0):
    # removing None values when counting where clusters go
    # df0 = df_new[df_new['NEXT_CLUSTER'] != 'None']
    df_new["NEXT_RISK"] = df_new.groupby(['ID'])['RISK'].shift(-1).values
    transition_df = df_new.dropna(subset=['NEXT_CLUSTER']).groupby(['CLUSTER', 'ACTION', 'NEXT_CLUSTER'])['RISK'].count()
    transition_df = transition_df[transition_df != 0].copy()

    # next cluster given how most datapoints transition for the given action
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
    df_group = []
    df_actions = df_new[["ID", "CLUSTER", "NEXT_CLUSTER", "ACTION", "NEXT_RISK"]].dropna()
    for group_idx, group in df_actions.groupby("CLUSTER"):
        mod = IsotonicRegression()
        mod.fit(group["ACTION"], group["NEXT_RISK"])
        group["NEXT_RISK_ISO"] = mod.predict(group["ACTION"])
        df_group.append(group)
    df_actions = pd.concat(df_group).sort_index()
    df_actions["RISK_ISO"] = df_actions.groupby(['ID'])['NEXT_RISK_ISO'].shift(1).values

    R_df = R_df.join(df_actions.groupby('CLUSTER')['RISK_ISO'].mean())
    R_df.columns = ['EST_RISK']

    return P_df, R_df


def precomplete_transition(P_df, actions, nc, window=2):
    for cluster in range(nc):
        for action in actions:
            if action != 0:
                if np.isnan(P_df.loc[(cluster, action)][0]):
                    cur_cluster = cluster
                    it = 0
                    while it < window:
                        it += 1
                        cur_cluster = P_df.loc[(cur_cluster, 0)][0]
                        if np.isnan(cur_cluster):
                            break
                        elif not np.isnan(P_df.loc[(cur_cluster, action)][0]):
                            P_df.loc[(cluster, action)] = P_df.loc[(cur_cluster, action)][0]
                            break
    return P_df


# Auxiliary function for deployment
# predict_region_date() takes a given state and a date and returns the predicted target_colname
def predict_region_date(mdp,  # MDP_model object
                        region_last_date,  # tuple (region, first_date, last_date), e.g (Alabama, Timestamp('2020-03-24 00:00:00'), Timestamp('2020-06-22 00:00:00'))
                        date,  # target_colname date for prediciton, e.g. (Timestamp('2020-05-24 00:00:00'))
                        verbose=0):

        region, last_date = region_last_date
        try:
            date = datetime.strptime(date, '%Y-%m-%d')
        except TypeError:
            pass

        # Case 1 : the input date occurs before the first available date for a given region
        try :
            assert date >= last_date
            n_days = (date-last_date).days
            return np.ceil(mdp.predict_region_ndays(region, n_days))
        except AssertionError:
            if verbose >= 1:
                print("Prediction Error type I ('{}', '{}'): the input occurs before the last available ('{}') date of the training set".format(region,
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
#############################################################################


#############################################################################
# Functions for Error

# training_value_error() takes in a clustered dataframe, and computes the
# E((\hat{v}-v)^2) expected error in estimating values (risk) given actions
# Returns a float of average value error per ID
def training_value_error(df_new, #Outpul of algorithm
                         mdp_transition,
                         relative=False, #Output Raw error or RMSE ie ((\hat{v}-v)/v)^2
                         h=5, #Length of forcast. The error is computed on v_h = \sum_{t=h}^H v_t
                         OutputFlag=0,
                         ):
    E_v = 0
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

        t = max(H-h, 0)
        s = df_new['CLUSTER'].loc[index + t]
        a = df_new['ACTION'].loc[index + t]
        v_true = df_new['RISK'].loc[index + t]
        v_estim = mdp_transition(s)
        t += 1
        #t = H-h +1
        # predicting path of each ID
        while cont:
            try:
                risk = df_new['RISK'].loc[index + t]
                v_true = v_true + risk
            except KeyError:
                if OutputFlag >= 1:
                    print('WARNING: In training value evaluation, for a given region there is only one observation')
                break
            try:
                s, r = mdp_transition(s, a)
            # error raises in case we never saw a given transition in the data
            except(TypeError, KeyError):
                if OutputFlag >= 1:  # DEBUG
                    print('WARNING: In training value evaluation, trying to predict next state from state',s,'taking action',a,', but this transition is never seen in the data. Data point:',i,t)
                # by default : we stay in the same state
                r = mdp_transition(s)

            a = df_new['ACTION'].loc[index + t]
            v_estim = v_estim + r

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


def populate_cluster(df_test, df_new, mdp_transition):
    starting_clusters = df_new.groupby("ID").first()[["CLUSTER", "ACTION"]].to_dict()
    df2 = df_test.reset_index().groupby("ID").first()
    N_test = df_test.ID.nunique()
    for i in range(N_test):
        # initializing index of first state for each ID
        index = df2['index'].iloc[i]
        region_id = df2.index[i]

        # first cluster
        try:
            previous_cluster = starting_clusters["CLUSTER"][region_id]
        except:
            continue
        previous_action = starting_clusters["ACTION"][region_id]
        cluster, _ = mdp_transition(previous_cluster, previous_action)
        action = df_test.loc[index, "ACTION"]
        df_test.at[index, "CLUSTER"] = cluster

        cont = True
        H = -1
        # Computing Horizon H of ID i
        while cont:
            H += 1
            try:
                df_test['ID'].loc[index+H+1]
            except:
                break
            if df_test['ID'].loc[index+H] != df_test['ID'].loc[index+H+1]:
                break
            else:
                cluster, action = mdp_transition(cluster, action)[0], df_test['ACTION'].loc[index+H+1]
                df_test.at[index+H+1, "CLUSTER"] = cluster
    return df_test


# testing_value_error() takes in a dataframe of testing data, and dataframe of
# new clustered data, a model from predict_cluster function, and computes the
# expected value error given actions and a predicted initial cluster
# Returns a float of sqrt average value error per ID
def testing_value_error(df_test, df_new, model, mdp_transition, OutputFlag=0, relative=False, h=5):
    # try:
    #     df_test['CLUSTER'] = model.predict(df_test.iloc[:, 2:2+pfeatures])
    # except ValueError:
    #     print('ERROR: Feature columns have missing values! Please drop' \
    #           ' rows or fill in missing data.', flush=True)
    #     df_test['CLUSTER'] = model.predict(df_test.iloc[:, 2:2+pfeatures])

    E_v = 0

    # compute initial clusters
    df_test = populate_cluster(df_test, df_new, mdp_transition)
    df2 = df_test.reset_index().dropna(subset=["CLUSTER"]).groupby(['ID']).first().copy()
    N_test = df2.shape[0]

    for i in range(N_test):
        # initializing index of first state for each ID
        index = df2['index'].iloc[i]
        cont = True
        H = -1
        # Computing Horizon H of ID i
        while cont:
            H += 1
            try:
                df_test['ID'].loc[index+H+1]
            except:
                break
            if df_test['ID'].loc[index+H] != df_test['ID'].loc[index+H+1]:
                break

        t = max(H-h, 0)
        s = df_test['CLUSTER'].loc[index + t]
        a = df_test['ACTION'].loc[index + t]
        v_true = df_test['RISK'].loc[index + t]
        v_estim = mdp_transition(s)
        t += 1
        while cont:
            try:
                risk = df_test['RISK'].loc[index + t]
                v_true = v_true + risk
            except KeyError:
                if OutputFlag >= 3:
                    print('WARNING: In training value evaluation, for a given region there is only one observation')
                break
            try:
                s, r = mdp_transition(s, a)
            # error raises in case we never saw a given transition in the data
            except(TypeError, KeyError):
                if OutputFlag >= 1:
                    print('WARNING: In training value evaluation, trying to predict next state from state',s,'taking action',a,', but this transition is never seen in the data. Data point:',i,t)
                r = mdp_transition(s)

            a = df_test['ACTION'].loc[index + t]
            v_estim = v_estim + r

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


def error_per_ID(df_test, df_new, model, actions, pfeatures, n_cluster, OutputFlag=0, relative=False, h=5):
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
    P_df,R_df = get_MDP(df_new, actions=actions, n_cluster=n_cluster)
    df2 = df_test.reset_index()
    df2 = df2.groupby(['ID']).first()
    N_test = df2.shape[0]
    # print(df2)

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

        t = max(0, H-h)
        s = df_test['CLUSTER'].loc[index + t]
        a = df_test['ACTION'].loc[index + t]
        v_true = df_test['RISK'].loc[index+t]
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
            except(TypeError, KeyError) :
                if OutputFlag >= 3:
                    print('WARNING: In training value evaluation, trying to predict next state from state',s,'taking action',a,', but this transition is never seen in the data. Data point:',i,t)

            a = df_test['ACTION'].loc[index + t]
            v_estim = v_estim + R_df.loc[s]

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

        State.append(df_test[df_test['ID'] == df_test['ID'].loc[index]]['ID'].iloc[0])
        V_true.append(v_true)
        V_estim.append(v_estim)
        V_err.append(np.abs(v_true-v_estim))
        C_err.append(np.abs((math.exp(v_true)-math.exp(v_estim))/math.exp(v_true)))
    df_err = pd.DataFrame()
    df_err['ID'] = State
    df_err['v_true'] = V_true
    df_err['v_estim'] = V_estim
    df_err['v_err'] = V_err
    df_err['cases_rel_err'] = C_err

    E_v = (E_v/N_test)
#    print('final E_v', E_v)
    #print('sqrt', np.sqrt(E_v))
    #return np.sqrt(E_v)
    return df_err, E_v


#############################################################################


#############################################################################
# Functions for R2 Values

# R2_value_training() takes in a clustered dataframe, and returns a float
# of the R-squared value between the expected value and true value of samples
def R2_value_training(df_new, mdp_transition, OutputFlag=0):
    E_v = 0
    # P_df, R_df = get_MDP(df_new, actions=actions, pfeatures=pfeatures, n_cluster=n_cluster, complete=True)
    df2 = df_new.reset_index()
    df2 = df2.groupby(['ID']).first()
    N = df2.shape[0]
    V_true = []
    for i in range(N):
        # initializing starting cluster and values
        s = df2['CLUSTER'].iloc[i]
        a = df2['ACTION'].iloc[i]
        v_true = df2['RISK'].iloc[i]

        v_estim = mdp_transition(s)
        index = df2['index'].iloc[i]
        cont = True
        t = 1
        # iterating through path of ID
        while cont:
            try:
                risk = df_new['RISK'].loc[index + t]
                v_true = v_true + risk
            except KeyError:
                if OutputFlag >= 3:
                    print('WARNING: In training value evaluation, for a given region there is only one observation')
                break
            try:
                s, r = mdp_transition(s, a)

            # error raises in case we never saw a given transition in the data
            except(TypeError, KeyError):
                if OutputFlag >= 1:
                    print('WARNING: In training value evaluation, trying to predict next state from state',s,'taking action',a,', but this transition is never seen in the data. Data point:',i,t)
                r = mdp_transition(s)

            a = df_new['ACTION'].loc[index + t]
            v_estim = v_estim + r
            try:
                df_new['ID'].loc[index+t+1]
            except KeyError:
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
    return max(1 - E_v/SS_tot, 0)


# R2_value_testing() takes a dataframe of testing data, a clustered dataframe,
# a model outputted by predict_cluster, and returns a float of the R-squared
# value between the expected value and true value of samples in the test set
def R2_value_testing(df_test, df_new, model, mdp_transition, OutputFlag=0):
    E_v = 0
    df2 = df_test.reset_index()
    df2 = df2.groupby(['ID']).first()

    # predicting clusters based on features
    # clusters = model.predict(df2.iloc[:, 2:2+pfeatures])
    # df2['CLUSTER'] = clusters

    # update the clusters (test)
    for id_cluster in df2.index:
        try:

            # end_date = df2.loc[id_cluster, "TIME"]
            start_date, start_cluster, action = df_new.loc[df_new.ID == id_cluster, ["TIME", "CLUSTER", "ACTION"]].tail(1).values[0]
            next_cluster, _ = mdp_transition(start_cluster, action)
            df2.at[id_cluster, 'CLUSTER'] = next_cluster
        except:
            # raise MDPTrainingError("Transition not found")
            pass

    df2 = df2.dropna(subset=["CLUSTER"]).copy()
    N = df2.shape[0]
    V_true = []
    for i in range(N):
        s = df2['CLUSTER'].iloc[i]
        a = df2['ACTION'].iloc[i]
        v_true = df2['RISK'].iloc[i]

        try:
            v_estim = mdp_transition(s)
        except:
            pass
        index = df2['index'].iloc[i]
        cont = True
        t = 1
        while cont:
            try:
                risk = df_test['RISK'].loc[index + t]
                v_true = v_true + risk
            except KeyError:
                if OutputFlag >= 3:
                    print('WARNING: In training value evaluation, for a given region there is only one observation')
                break
            try:
                s, r = mdp_transition(s, a)

            # error raises in case we never saw a given transition in the data
            except(TypeError, KeyError):
                if OutputFlag >= 1:
                    print('WARNING: In training value evaluation, trying to predict next state from state', s, 'taking action', a, ', but this transition is never seen in the data. Data point:',i,t)
                r = mdp_transition(s)
            a = df_test['ACTION'].loc[index + t]
            v_estim = v_estim + r

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
    return max(1 - E_v/SS_tot, 0)
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
    P_df,R_df = get_MDP(df_new)

    t = max(H-h, 0)

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
            s = P_df.loc[s, 0].values[0]
            s_seq.append(s)
        except TypeError:
            if OutputFlag >= 2:
                print('WARNING: Trying to predict next state from state',s,'taking action',a,', but this transition is never seen in the data. Data point:',i,t)
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
        plt.show(block=False)
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
        plt.show(block=False)
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
                          columns=['state', 'sequence', 'ratios', 'error'])
    return df_seq


# plot_pred() takes a trained model, a specific US state name, and the df_true
# (sorted by TIME), and plots the predicted versus true cases for n_days
def plot_pred(model, state, df_true, n_days, from_first=False):
    h = int(np.floor(n_days/model.days_avg))
    delta = n_days - model.days_avg*h
    df_true.loc[:, [model.date_colname]] = pd.to_datetime(df_true[model.date_colname])
    if from_first:
        date = model.df_trained_first.loc[state, "TIME"]
        target = model.df_trained_first.loc[state, model.target_colname]
        s = model.df_trained_first.loc[state, "CLUSTER"]
    else:
        date = model.df_trained.loc[state, "TIME"]
        target = model.df_trained.loc[state, model.target_colname]
        s = model.df_trained.loc[state, "CLUSTER"]
    dates = [date]
    targets_pred = [target]

    r = 1
    r_next = model.mdp_transition(s)
    for i in range(h):
        dates.append(date + timedelta((i+1)*model.days_avg))
        r *= np.exp(r_next)
        targets_pred.append(target*r)
        s, r_next = model.mdp_transition(s, 0)

    fig, ax = plt.subplots()
    ax.plot(df_true.loc[df_true[model.region_colname]==state][model.date_colname], \
            df_true.loc[df_true[model.region_colname]==state][model.target_colname], \
            label = 'True '+model.target_colname)
    ax.plot(dates, targets_pred, label='Predicted '+model.target_colname)
    ax.set_title('%s True vs Predicted '%state + model.target_colname)
    ax.set_xlabel('Date')
    ax.set_ylabel(model.target_colname)
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.show(block=False)


# plot_pred() takes a trained model, a specific US state name, and the df_true
# (sorted by TIME), and plots the predicted versus true cases for n_days
def plot_pred_action(model, state, df_true, n_days, action_day=0, from_first=False):
    h = int(np.floor(n_days/model.days_avg))
    action_adj = int(np.floor(action_day/model.days_avg))
    df_true.loc[:, [model.date_colname]] = pd.to_datetime(df_true[model.date_colname])
    if from_first:
        date = model.df_trained_first.loc[state, "TIME"]
        target = model.df_trained_first.loc[state, model.target_colname]
        s_init = model.df_trained_first.loc[state, "CLUSTER"]
    else:
        date = model.df_trained.loc[state, "TIME"]
        target = model.df_trained.loc[state, model.target_colname]
        s_init = model.df_trained.loc[state, "CLUSTER"]

    actions = [a - model.action_thresh[1] for a in range(len(model.action_thresh[0])+1)]
    fig, ax = plt.subplots(figsize=(22, 11))

    # prediction 0
    for a in actions:
        s = s_init
        dates = [date]
        targets_pred = [target]
        r = 1.
        for i in range(h):
            dates.append(date + timedelta((i+1)*model.days_avg))
            try:
                if i == action_adj:
                    s_bf = s
                    s, r_next = model.mdp_transition(s, a)
                    print("with action {}".format(a), " STATE bef:", s_bf, " STATE aft:", s)
                else:
                    s, r_next = model.mdp_transition(s, 0)
            except TypeError:
                print("Transition not found:", (s, a))
                break
            r *= np.exp(r_next)
            targets_pred.append(target*r)

        target_pred_act = pd.Series(index=dates, data=targets_pred)
        target_pred_act.plot(label='Predicted '+model.target_colname+ ' with ACTION {} after {} days'.format(a, action_day), ax=ax)

    true_cases = pd.Series(index=df_true.loc[df_true[model.region_colname]==state][model.date_colname],
                           data=df_true.loc[df_true[model.region_colname]==state][model.target_colname].values)
    true_cases.plot(label='True '+ model.target_colname, ax=ax, color="k", linestyle="--")

    ax.set_title('%s True vs Predicted '%state + model.target_colname)
    ax.set_xlabel('Date')
    ax.set_ylabel(model.target_colname)
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.grid()
    plt.show(block=False)


def plot_pred_fact(model, state, df_w_act, starting_date, n_days=30):

    fig, ax = plt.subplots()

    try :
        starting_date = datetime.strptime(starting_date, "%Y%m%d")
    except:
        pass

    df_state = df_w_act[df_w_act[model.region_colname] == state]

    first_date = df_state["TIME"].min()

    starting_n_days = int((starting_date - first_date).days)
    try:
        assert starting_n_days >= 0
    except AssertionError:
        print(" the given starting_date {} occurs before the first date {} of the given data set".format(str(starting_date),
                                                                                                         str(first_date)))
        raise AssertionError

    h = int(np.floor(n_days/model.days_avg))
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
            a = df_state.iloc[(starting_n_days+i)]["ACTION"]
        except:
            print(state)
            print(df_state.iloc[(starting_n_days+i)])
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
#############################################################################`


#############################################################################
def cluster_size(df):
    return df.groupby('CLUSTER')['RISK'].agg(['count', 'mean', 'std', 'min', 'max'])


def show_state(df_new,df,state,pfeatures):
    model = predict_cluster(df_new,pfeatures)
    st = df[df['state'] == state]
    st['CLUSTER'] = model.predict(st.iloc[:,2:pfeatures+2])
    return st[['TIME', 'cases', 'RISK', 'CLUSTER', 'r_t']]


def mape(df_pred,df_true, target_colname):
    df_pred['real '+target_colname] = df_true[target_colname]
    df_pred['rel_error'] = abs(df_pred[target_colname]-df_true[target_colname])/df_true[target_colname]
    return df_pred


def mape_(y_pred, y_true):
    return abs(y_pred-y_true)/y_true
#############################################################################
