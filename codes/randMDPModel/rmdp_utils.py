# %% Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # progress bar
from copy import deepcopy
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch, DBSCAN
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from collections import Counter
from itertools import groupby
from operator import itemgetter
import os

import warnings

warnings.filterwarnings("ignore")

from randMDPModel.rmdp_testing import R2_value, training_value_error, \
    predict_cluster, testing_value_error, MDPTrainingError, prediction_score, get_MDP, \
    compute_state_target_risk, compute_state_target_alpha, predict_region_date, MDPPredictionError
from data_utils import save_model


#############################################################################
# Splitter algorithm with cross-validation

#############################################################################

# splitter() is the wrap-up function. Takes as parameters a dataframe df,
# a time-horizon T, a number of features pfeatures, an indexer k, and a max
# number of iterations and performs the algorithm until all contradictions are
# resolved or until the max number of iterations is reached
# Plots the trajectory of testing metrics during splitting process
# Returns the final resulting dataframe

class MDPSplitterError(Exception):
    pass


class MDPPredictorError(Exception):
    pass


class MDPPredictor:
    def __init__(self):
        self.P_df = None
        self.R_df = None
        self.current_state_date = None
        self.training_error = None
        self.testing_error = None
        self.R2_train = None
        self.R2_test = None
        self.columns = None
        self.features = None
        self.model = None
        self.accuracy_cluster = None
        self.classifier = None

    def complete_path(self, all_data):
        if self.classifier is None:
            for region_id in self.current_state_date.index:
                current_date = self.current_state_date.loc[region_id, "TIME"]
                current_cluster = self.current_state_date.loc[region_id, "CLUSTER"]

                state_data = all_data.loc[region_id]
                state_data = state_data[state_data["TIME"] >= current_date].reset_index()
                missing_size = state_data.shape[0] - 1
                if missing_size > 0 :
                    previous_cluster = current_cluster
                    for row_id, row in state_data.iterrows():
                        previous_cluster = current_cluster
                        current_date = row["TIME"]
                        action = row["ACTION"]
                        target = row["TARGET"]
                        current_cluster = self.P_df.loc[(current_cluster, action)][0]
                    self.current_state_date.loc[region_id, "TIME"] = current_date
                    self.current_state_date.loc[region_id, "CLUSTER"] = previous_cluster
                    self.current_state_date.loc[region_id, "TARGET"] = target
        else:
            self.current_state_date = all_data.reset_index().groupby('ID').last()
            try:
                self.current_state_date['CLUSTER'] = self.classifier.predict(self.current_state_date.loc[:, self.features])
            except KeyError:
                raise MDPPredictionError("Error Init Cluster Prediction")
            self.current_state_date = self.current_state_date.loc[:, ["TIME", "CLUSTER", "RISK", "TARGET"]].copy()


class MDP_Splitter:

    def __init__(self,
                 df,
                 all_data,
                 days_avg,
                 features,
                 reward_name="RISK",
                 error_computing="horizon",
                 horizon=5,
                 alpha=1e-5,
                 error_function_name="relative",
                 clustering='Agglomerative',
                 actions=[0],
                 init_n_clusters=None,
                 mode="train",
                 distance_threshold=0.1,
                 random_state=42,
                 verbose=0):

        self.df = df.sort_values(["ID", "TIME"]).copy()
        self.all_data = all_data.set_index(["ID", "TIME"]).copy()
        self.clustering = clustering
        self.reward_name = reward_name
        self.init_n_clusters = init_n_clusters
        self.distance_threshold = distance_threshold
        self.features = features
        self.days_avg = days_avg
        self.error_computing = error_computing
        self.horizon = horizon
        self.alpha = alpha
        self.random_state = random_state
        self.actions = actions
        self.mode = mode

        self.error_function_name = error_function_name
        self.error_eval = get_error_eval(self.error_function_name, self.mode)
        self.verbose = verbose

        self.model = None
        self.P_df = None
        self.R_df = None
        self.error_transition_df = None
        self.current_state_date = None
        self.accuracy_cluster = []
        self.n_clusters = None
        self.R2_train = None
        self.R2_test = None
        self.training_error = None
        self.testing_error = None

    # initializeClusters() takes as input a dataframe, a time horizon T,
    # a clustering algorithm, a number of clusters n_clusters,
    # and a random seed (optional) and returns a dataframe
    # with two new columns 'CLUSTER' and 'NEXT_CLUSTER'
    def initializeClusters(self, reward=["RISK_SCALE"]):  # random seed for the clustering
        if self.clustering == 'KMeans':
            output = KMeans(
                n_clusters=self.init_n_clusters, random_state=self.random_state).fit(
                np.array(self.df[reward]).reshape(-1, len(reward))).labels_
        elif self.clustering == 'Agglomerative':
            output = AgglomerativeClustering(
                n_clusters=self.init_n_clusters, distance_threshold=self.distance_threshold).fit(
                np.array(self.df[reward]).reshape(-1, len(reward))).labels_
        elif self.clustering == 'Birch':
            output = Birch(
                n_clusters=self.init_n_clusters).fit(
                np.array(self.df[reward]).reshape(-1, len(reward))).labels_
        elif self.clustering == 'DBSCAN':
            output = DBSCAN(
                n_clusters=self.init_n_clusters).fit(
                np.array(self.df[reward]).reshape(-1, 1)).labels_
        else:
            output = LabelEncoder().fit_transform(np.array(self.df[reward]).reshape(-1, 1))
        self.df['CLUSTER'] = output
        self.df['NEXT_CLUSTER'] = self.df.groupby('ID')['CLUSTER'].shift(-1)

    def update_mdp(self):
        # compute the estimated rewards and transition matrix
        self.P_df, self.R_df = get_MDP(self.df, self.actions, self.features, n_cluster=self.n_clusters,
                                       OutputFlag=self.verbose, reward=self.reward_name)

        self.current_state_date = self.df.groupby(["ID"]).last()[["TIME", "CLUSTER", "RISK", "TARGET"]]

        # Add estimated next cluster [MUST BE IMPROVE]

    def get_mdp_tree(self, other):
        # (put comment)
        self.P_df = other.P_df.copy()
        self.R_df = other.R_df.copy()
        self.model = deepcopy(other.model)
        self.current_state_date = other.current_state_date.copy()

    def compute_risk_error(self):
        # TRAINING MODE
        if self.mode == "train":
            if self.reward_name == "RISK":
                self.df = compute_risk_error_train_risk(self.df, self.P_df, self.R_df,
                                                        self.error_eval, self.horizon)
            elif self.reward_name == "AlphaRISK":
                self.df = compute_risk_error_train_alpha(self.df, self.P_df, self.R_df,
                                                        self.error_eval, self.horizon)
            else:
                raise MDPSplitterError("(Train) Reward name not found")

        # TESTING MODE
        elif self.mode == "test":

            if self.reward_name == "RISK":
                test_initial_state_date = self.df.groupby("ID").first()["TIME"].copy()
                test_initial_state_date = compute_state_target_risk(
                    test_init_state=test_initial_state_date,
                    current_state_target=self.current_state_date,
                    transitions=self.P_df,
                    rewards=self.R_df,
                    all_data=self.all_data,
                    days_avg=self.days_avg,)

                try :
                    self.df.drop(["INIT_TARGET", "EST_H_NEXT_CLUSTER"], axis=1, inplace=True)
                except KeyError:  # do while equivalent
                    pass
                # compute the initial predicted clusters
                self.df = self.df.set_index("ID").join(test_initial_state_date[["INIT_TARGET", "EST_H_NEXT_CLUSTER"]], how="left").reset_index().dropna(subset=["EST_H_NEXT_CLUSTER"])
                self.df["EST_H_NEXT_CLUSTER"] = self.model.predict(self.df.loc[:, self.features])
                self.df = compute_risk_error_test_risk(self.df, self.P_df, self.R_df,
                                                        self.error_eval, self.horizon)
            elif self.reward_name == "AlphaRISK":
                test_initial_state_date = self.df.groupby("ID").first()["TIME"].copy()
                test_initial_state_date = compute_state_target_alpha(
                    test_init_state=test_initial_state_date,
                    current_state_target=self.current_state_date,
                    transitions=self.P_df,
                    rewards=self.R_df,
                    all_data=self.all_data,
                    days_avg=self.days_avg)

                try :
                    self.df.drop(["INIT_TARGET", "INIT_RISK", "EST_H_NEXT_CLUSTER"], axis=1, inplace=True)
                except KeyError:
                    pass
                self.df = self.df.set_index("ID").join(test_initial_state_date[["INIT_TARGET", "INIT_RISK", "EST_H_NEXT_CLUSTER"]], how="left").reset_index().dropna(subset=["EST_H_NEXT_CLUSTER"])
                
                # compute the initial predicted clusters
                self.df["EST_H_NEXT_CLUSTER"] = self.model.predict(self.df.loc[:, self.features])
                self.df = compute_risk_error_test_alpha(self.df, self.P_df, self.R_df,
                                                        self.error_eval, self.horizon)

            else:
                raise MDPSplitterError("(test) Reward name not found")

        else:
            raise MDPSplitterError("mode is not found : must be either 'train' or 'test'")

    def update_cluster(self):
        # pass
        self.model = deepcopy(predict_cluster(self.df, self.features))

    def compute_transition_error(self, method="myopic"):
        n_df = self.df.groupby(["CLUSTER", "ACTION"])["NEXT_CLUSTER"].nunique()
        if method == "robust":
            e_df = self.df.groupby(["CLUSTER", "ACTION"])["EST_H_ERROR"].max() / self.horizon
        elif method == "mean":
            e_df = self.df.groupby(["CLUSTER", "ACTION"])["EST_H_ERROR"].mean() / self.horizon
        elif method == "myopic":
            e_df = 1.
        else:
            raise MDPSplitterError
        self.error_transition_df = (np.log(n_df) * e_df).copy()
        pass

    def to_mdp(self, complete=False):
        mdp_predictor = MDPPredictor()
        mdp_predictor.P_df = self.P_df
        mdp_predictor.R_df = self.R_df
        mdp_predictor.current_state_date = self.current_state_date
        mdp_predictor.training_error = self.training_error
        mdp_predictor.testing_error = self.testing_error
        mdp_predictor.R2_train = self.R2_train
        mdp_predictor.R2_test = self.R2_test
        mdp_predictor.columns = self.features
        mdp_predictor.classifier = self.model
        mdp_predictor.accuracy_cluster = self.accuracy_cluster
        mdp_predictor.verbose = self.verbose
        mdp_predictor.features = self.features

        # add additional
        if complete:
            try:
                mdp_predictor.complete_path(self.all_data.reset_index().set_index("ID"))
            except MDPPredictionError:
                raise MDPSplitterError
        return mdp_predictor

#############################################################################

#############################################################################
# Error functions

# get_error_eval (MISSING DESCRIPTION)
def get_error_eval(function_name, mode="train"):
    try:
        assert function_name in {"sym_abs_relative", "absolute",
                                 "relative", "exp_relative"}, "the error function name must be " \
                                                                              "either 'sym_abs_relative', 'absolute', " \
                                                                              " 'relative' or 'exp_relative'"
        assert mode in {"train", "test"}, "the error function mode must be either 'train' or 'test'"
    except AssertionError:
        raise MDPSplitterError

    if mode == "train":
        if function_name == "sym_abs_relative":
            # return |y_true - y_est| / |y_true + y_est|
            return np.vectorize(lambda y_est, y_true: np.abs(y_est - y_true) / np.abs(y_est + y_true))

        if function_name == "absolute":
            # return |y_true - y_est|
            return np.vectorize(lambda y_est, y_true: np.abs(y_est - y_true))

        if function_name == "relative":
            # return |y_true - y_est|/|y_true|
            return np.vectorize(lambda y_est, y_true: np.abs(y_est - y_true) / np.abs(y_true))

        if function_name == "exp_relative":
            # return |exp(y_true) - exp(y_est)|/exp(y_true)
            return np.vectorize(lambda y_est, y_true: np.abs(np.exp(y_est) - np.exp(y_true)) / np.exp(y_true))

    if mode == "test":
        if function_name == "sym_abs_relative":
            # return |y_true - y_est| / |y_true + y_est|
            return np.vectorize(lambda y_est, y_true, init_target_y_est, target_y_true:
                                np.abs(y_est - y_true) / np.abs(y_est + y_true))

        if function_name == "absolute":
            # return |y_true - y_est|
            return np.vectorize(lambda y_est, y_true, init_target_y_est, target_y_true: np.abs(y_est - y_true))

        if function_name == "relative":
            # return |y_true - y_est|/|y_true|
            return np.vectorize(lambda y_est, y_true, init_target_y_est, target_y_true:
                                np.abs(y_est - y_true) / y_true)

        if function_name == "exp_relative":
            # return |exp(y_true) - exp(y_est)|/exp(y_true)
            return np.vectorize(lambda y_est, y_true, init_target_y_est, target_y_true:
                                np.abs(init_target_y_est * np.exp(y_est) - target_y_true) / target_y_true)


# Compute alpha risk error (training set - alpharisk as reward)
def compute_risk_error_train_alpha(df, P_df, R_df, error_eval, horizon=5):
    # df["EST_H_NEXT_CLUSTER"] = self.model.predict(df.iloc[:, 2:2 + self.pfeatures])
    df["EST_H_NEXT_CLUSTER"] = df["CLUSTER"].values

    df["EST_CUR_RISK"] = df["RISK"].values
    df["EST_H_NEXT_RISK"] = 0.
    df["EST_H_NEXT_AlphaRISK"] = 0.
    df["EST_H_ERROR"] = 0.

    for h in range(1, horizon + 1):  # [TO ADAPT WITH EXPONENTIAL]
        # COMPUTE THE NEXT CLUSTER
        df = pd.merge(
            df,
            P_df.reset_index().rename(columns={"ACTION": "ACTION_", "CLUSTER": "CLUSTER_"}),
            left_on=["EST_H_NEXT_CLUSTER", "ACTION"],
            right_on=["CLUSTER_", "ACTION_"],
            how="left").copy().drop(["CLUSTER_", "ACTION_"],
                                    axis=1).drop(["EST_H_NEXT_CLUSTER"],
                                                 axis=1).rename(columns={"TRANSITION_CLUSTER": "EST_H_NEXT_CLUSTER"})

        # COMPUTE THE ONE STEP ESTIMATED RISK
        df["EST_H_NEXT_AlphaRISK"] = pd.merge(
            df.drop('EST_H_NEXT_AlphaRISK', axis=1),
            R_df.reset_index(),
            left_on="EST_H_NEXT_CLUSTER",
            right_on="CLUSTER",
            how="left")["EST_AlphaRISK"].values
        df["EST_CUR_RISK"] *= np.exp(df["EST_H_NEXT_AlphaRISK"]).values
        df["EST_H_NEXT_RISK"] += df["EST_CUR_RISK"].values

        df["TRUE_H_NEXT_RISK"] = df.groupby("ID", sort=False).rolling(window=h, min_periods=h)[
            "RISK"].sum().values
        df["TRUE_H_NEXT_RISK"] = df.groupby("ID", sort=False)["TRUE_H_NEXT_RISK"].shift(-h)

        # COMPUTE THE ONE STEP ERROR
        df["EST_H_ERROR"] += error_eval(df["EST_H_NEXT_RISK"].values,
                                             df["TRUE_H_NEXT_RISK"].values)
    return df


# Compute risk error (training set - risk as reward)
def compute_risk_error_train_risk(df, P_df, R_df, error_eval, horizon=5):
    # df["EST_H_NEXT_CLUSTER"] = self.model.predict(df.iloc[:, 2:2 + self.pfeatures])
    df["EST_H_NEXT_CLUSTER"] = df["CLUSTER"].values

    df["EST_CUR_RISK"] = df["RISK"].values
    df["EST_H_NEXT_RISK"] = 0.
    df["EST_H_ERROR"] = 0.

    for h in range(1, horizon + 1):  # [TO ADAPT WITH EXPONENTIAL]
        # COMPUTE THE NEXT CLUSTER
        df = pd.merge(
            df,
            P_df.reset_index().rename(columns={"ACTION": "ACTION_", "CLUSTER": "CLUSTER_"}),
            left_on=["EST_H_NEXT_CLUSTER", "ACTION"],
            right_on=["CLUSTER_", "ACTION_"],
            how="left").copy().drop(["CLUSTER_", "ACTION_"],
                                    axis=1).drop(["EST_H_NEXT_CLUSTER"], axis=1).rename(columns={"TRANSITION_CLUSTER": "EST_H_NEXT_CLUSTER"})

        # # COMPLETE MISSING TRANSITION WITH THE LAST AVAILABLE CLUSTER
        # df["EST_H_NEXT_CLUSTER"] = df.groupby(['ID'], sort=False).apply(lambda group: group.ffill())[
        #     'EST_H_NEXT_CLUSTER'].values

        # COMPUTE THE ONE STEP ESTIMATED RISK
        df["EST_CUR_RISK"] = pd.merge(
            df.drop('EST_CUR_RISK', axis=1),
            R_df.reset_index(),
            left_on="EST_H_NEXT_CLUSTER",
            right_on="CLUSTER",
            how="left")["EST_RISK"].values
        df["EST_H_NEXT_RISK"] += df["EST_CUR_RISK"].values

        df["TRUE_H_NEXT_RISK"] = df.groupby("ID").rolling(window=h, min_periods=h)[
            "RISK"].sum().values
        df["TRUE_H_NEXT_RISK"] = df.groupby("ID")["TRUE_H_NEXT_RISK"].shift(-h)

        # COMPUTE THE ONE STEP ERROR
        df["EST_H_ERROR"] += error_eval(df["EST_H_NEXT_RISK"].values,
                                             df["TRUE_H_NEXT_RISK"].values)
    return df


# Compute risk error (testing set - alpharisk as reward)
def compute_risk_error_test_alpha(df, P_df, R_df, error_eval, horizon=5):

        df["EST_H_NEXT_RISK"] = 0.
        df["EST_H_NEXT_AlphaRISK"] = 0.
        df["EST_H_ERROR"] = 0.
        df["EST_CUR_RISK"] = df["INIT_RISK"].values

        for h in range(1, horizon + 1):  # [TO ADAPT WITH EXPONENTIAL]
            # COMPUTE THE NEXT CLUSTER
            df = pd.merge(
                df,
                P_df.reset_index().rename(columns={"ACTION": "ACTION_", "CLUSTER": "CLUSTER_"}),
                left_on=["EST_H_NEXT_CLUSTER", "ACTION"],
                right_on=["CLUSTER_", "ACTION_"],
                how="left").copy().drop(["CLUSTER_", "ACTION_"],
                                        axis=1).drop(["EST_H_NEXT_CLUSTER"],
                                                     axis=1).rename(columns={"TRANSITION_CLUSTER": "EST_H_NEXT_CLUSTER"})

            # COMPLETE MISSING TRANSITION WITH THE LAST AVAILABLE CLUSTER
            df["EST_H_NEXT_CLUSTER"] = df.groupby(['ID'], sort=False).apply(lambda group: group.ffill())[
                'EST_H_NEXT_CLUSTER'].values

            # COMPUTE THE ONE STEP ESTIMATED RISK
            df["EST_H_NEXT_AlphaRISK"] = pd.merge(
                df.drop('EST_H_NEXT_AlphaRISK', axis=1),
                R_df.reset_index(),
                left_on="EST_H_NEXT_CLUSTER",
                right_on="CLUSTER",
                how="left")["EST_AlphaRISK"].values
            df["EST_CUR_RISK"] *= np.exp(df["EST_H_NEXT_AlphaRISK"]).values
            df["EST_H_NEXT_RISK"] += df["EST_CUR_RISK"].values

            df["TRUE_H_NEXT_RISK"] = df.groupby("ID", sort=False).rolling(window=h, min_periods=h)[
                "RISK"].sum().values
            df["TRUE_H_NEXT_RISK"] = df.groupby("ID", sort=False)["TRUE_H_NEXT_RISK"].shift(-h)

            # COMPUTE THE ONE STEP ERROR
            df["EST_H_ERROR"] += error_eval(df["EST_H_NEXT_RISK"].values,
                                            df["TRUE_H_NEXT_RISK"].values,
                                            df["INIT_TARGET"].values,
                                            df["TARGET"])
        # y_est, y_true, init_target_y_est, target_y_true
        df.drop(["INIT_TARGET", "EST_H_NEXT_CLUSTER", "EST_H_NEXT_AlphaRISK", "EST_CUR_RISK", "INIT_RISK"], axis=1, inplace=True)
        return df


# Compute risk error (training set - risk as reward)
def compute_risk_error_test_risk(df, P_df, R_df, error_eval, horizon=5):

    # df["EST_H_NEXT_CLUSTER"] = self.model.predict(df.iloc[:, 2:2 + self.pfeatures])
    df["EST_H_NEXT_RISK"] = 0.
    df["EST_H_ERROR"] = 0.
    df['EST_CUR_RISK'] = 0.

    for h in range(1, horizon + 1):  # [TO ADAPT WITH EXPONENTIAL]
        # COMPUTE THE NEXT CLUSTER
        df = pd.merge(
            df,
            P_df.reset_index().rename(columns={"ACTION": "ACTION_", "CLUSTER": "CLUSTER_"}),
            left_on=["EST_H_NEXT_CLUSTER", "ACTION"],
            right_on=["CLUSTER_", "ACTION_"],
            how="left").copy().drop(["CLUSTER_", "ACTION_"],
                                    axis=1).drop(["EST_H_NEXT_CLUSTER"],
                                                 axis=1).rename(columns={"TRANSITION_CLUSTER": "EST_H_NEXT_CLUSTER"})

        # COMPUTE THE ONE STEP ESTIMATED RISK
        df["EST_CUR_RISK"] = pd.merge(
            df.drop('EST_CUR_RISK', axis=1),
            R_df.reset_index(),
            left_on="EST_H_NEXT_CLUSTER",
            right_on="CLUSTER",
            how="left")["EST_RISK"].values
        df["EST_H_NEXT_RISK"] += df["EST_CUR_RISK"].values

        df["TRUE_H_NEXT_RISK"] = df.groupby("ID", sort=False).rolling(window=h, min_periods=h)[
            "RISK"].sum().values
        df["TRUE_H_NEXT_RISK"] = df.groupby("ID", sort=False)["TRUE_H_NEXT_RISK"].shift(-h)

        # COMPUTE THE ONE STEP ERROR
        df["EST_H_ERROR"] += error_eval(df["EST_H_NEXT_RISK"].values,
                                        df["TRUE_H_NEXT_RISK"].values,
                                        df["INIT_TARGET"].values,
                                        df["TARGET"])

    return df


#############################################################################
# Optimization  function



#############################################################################
# Function for the Iterations
def compute_error_aux(df, P_df, R_df, h):
    pass

# findConstradiction() takes as input a dataframe and returns the tuple with
# initial cluster and action that have the most number of contradictions or
# (-1, -1) if no such cluster existss
def findContradiction(error_df,  # pandas dataFrame
                      th=1e-2):  # integer: threshold split size
    try:
        error_df_ = error_df.dropna()
        # np.random.seed(1234)
        # selectedCont = np.random.choice(error_df_.index,
        #                                 p=error_df_.values/error_df_.values.sum())
        selectedCont = error_df_.index[error_df_.argmax()]

        if error_df[selectedCont] > th:
            return selectedCont
        return -1, -1
    except ValueError:
        return -1, -1


# contradiction() outputs one found contradiction given a dataframe,
# a cluster and a an action or (None, None) if none is found
def contradiction(df,  # pandas dataFrame
                  i,  # integer: initial clusters
                  a):  # integer: action taken
    nc = df.query('CLUSTER == @i').query(
        'ACTION == @a').query('NEXT_CLUSTER.notnull()', engine='python')[['NEXT_CLUSTER', 'EST_H_ERROR']]

    if len(nc) <= 1:
        return None, None
    else:
        # nc = nc.groupby("NEXT_CLUSTER")["EST_H_ERROR"].sum().argmax()  # first measure
        nc = nc.groupby("NEXT_CLUSTER")["EST_H_ERROR"].sum()
        nc = nc.index[nc.argmax()]
        return a, nc

# [DEVELOP]
# contradiction() outputs one found contradiction given a dataframe,
# a cluster and a an action or (None, None) if none is found
# def contradiction(df,  # pandas dataFrame
#                   i,  # integer: initial clusters
#                   a):  # integer: action taken
#     nc = df.query('CLUSTER == @i').query(
#         'ACTION == @a').query('EST_H_NEXT_CLUSTER.notnull()', engine='python')[['EST_H_NEXT_CLUSTER', 'EST_H_ERROR']]
#
#     if len(nc) <= 1:
#         return None, None
#     else:
#         # nc = nc.groupby("NEXT_CLUSTER")["EST_H_ERROR"].sum().argmax()  # first measure
#         nc = nc.groupby("EST_NEXT_CLUSTER")["EST_H_ERROR"].max()
#         nc = nc.index[nc.argmax()]
#         return a, nc


# multimode() returns a list of the most frequently occurring values.
# Will return more than one result if there are multiple modes
# or an empty list if *data* is empty.
def multimode(data):
    counts = Counter(iter(data)).most_common()
    maxcount, mode_items = next(groupby(counts, key=itemgetter(1)), (0, []))
    return list(map(itemgetter(0), mode_items))


# split() takes as input a dataframe, an initial cluster, an action, a target_colname
# cluster that is a contradiction c, a time horizon T, then number of features,
# and an iterator k (that is the indexer of the next cluster), as well as the
# predictive classification algorithm used
# and returns a new dataframe with the contradiction resolved
# MAKE SURE TO CHECK the number to ensure group creation has all features!! ***
def split(df,  # pandas dataFrame
          i,  # integer: initial cluster
          a,  # integer: action taken
          c,  # integer: target cluster
          features,  # s: number of features
          k,  # integer: intedexer for next cluster
          classification='LogisticRegression',
          random_state=0):  # string: classification aglo

    g1 = df[(df['CLUSTER'] == i) & (
            df['ACTION'] == a) & (df['NEXT_CLUSTER'] == c)]
    g2 = df[(df['CLUSTER'] == i) & (
            df['ACTION'] == a) & (df['NEXT_CLUSTER'] != c) & (
                df['NEXT_CLUSTER'].notnull())]
    g3 = df[(df['CLUSTER'] == i) & (
            ((df['ACTION'] == a) & (df['NEXT_CLUSTER'].isnull())) | (
            df['ACTION'] != a))]
    groups = [g1, g2, g3]
    data = {}

    for j in range(len(groups)):
        d = pd.DataFrame(groups[j].loc[:, features].values.tolist())

        data[j] = d

    data[0].insert(data[0].shape[1], "GROUP", np.zeros(data[0].shape[0]))
    data[1].insert(data[1].shape[1], "GROUP", np.ones(data[1].shape[0]))

    training = pd.concat([data[0], data[1]])

    tr_X = training.loc[:, ~(training.columns == "GROUP")]
    tr_y = training.loc[:, "GROUP"]

    if classification == 'LogisticRegression':
        m = LogisticRegression(solver='liblinear', random_state=random_state)
    elif classification == 'LogisticRegressionCV':
        m = LogisticRegressionCV(random_state=random_state)
    elif classification == 'DecisionTreeClassifier':
        # m = DecisionTreeClassifier(random_state=random_state )
        params = {
            'max_depth': [6, 7, 10, 20, None],
            # 'ccp_alpha': np.logspace(-4, 0, 30)
        }
        m = DecisionTreeClassifier()
        # m = RandomForestClassifier()

        m = RandomizedSearchCV(m, params, cv=2, iid=True, n_iter=40)
    elif classification == 'RandomForestClassifier':
        # m = DecisionTreeClassifier(random_state=random_state )
        params = {
            'max_depth': [6, 7, 10, 20, None],
            # 'ccp_alpha': np.logspace(-4, 0, 30)
        }
        m = RandomForestClassifier(random_state=random_state)
    elif classification == 'XGBClassifier':
        m = XGBClassifier(random_state=random_state)

    else:
        m = LogisticRegression(solver='liblinear')

    m.fit(tr_X, tr_y.values.ravel())

    ids = g2.index.values

    test_X = data[2]
    if len(test_X) != 0:
        Y = m.predict(test_X)
        g3.insert(g3.shape[1], "GROUP", Y)
        id2 = g3.loc[g3["GROUP"] == 1].index.values
        ids = np.concatenate((ids, id2))

    df.loc[df.index.isin(ids), 'CLUSTER'] = k
    newids = ids - 1
    df.loc[(df.index.isin(newids)) & (df['ID'] == df['ID'].shift(-1)), 'NEXT_CLUSTER'] = k

    return df


# (MDP GRID SEARCH FUNCTION)
# Splitting function from the MDP learning algorithm
def splitter(splitter_dataframe,  # pandas dataFrame
             th,  # integer: threshold for minimum split
             test_splitter_dataframe=None,
             testing=False,
             classification='LogisticRegression',  # string: classification alg
             it=100,  # integer: max number of clusters
             OutputFlag=1,
             n=-1,
             random_state=0,
             plot=False,
             save=True,
             savepath=None):  # If we plot error

    # initializing lists for error & accuracy data
    training_R2 = []
    testing_R2 = []
    testing_error = []
    training_error = []
    mdp_splitter_list = []
    nc = splitter_dataframe.df['CLUSTER'].nunique()  # initial number of clusters
    k = nc

    best_testing_error = np.inf
    best_mdp_predictor = None

    # Setting progress bar--------------
    if OutputFlag >= 1:
        split_bar = tqdm(range(max(1, it - nc)), position=0)
        # line1 = tqdm(range(it - nc), position=1)
        # line2 = tqdm(range(it - nc), position=2)
    else:
        split_bar = range(max(1, it - nc))
    split_bar.set_description("Splitting...")
    # Setting progress bar--------------

    for _ in split_bar:
        splitter_dataframe.n_clusters = nc

        # compute the MDP features
        splitter_dataframe.update_mdp()

        # train the decision tree
        splitter_dataframe.update_cluster()

        # compute horizon error & transition error
        splitter_dataframe.compute_risk_error()

        # compute transition error
        splitter_dataframe.compute_transition_error()

        # error and accuracy calculations
        R2_train = R2_value(splitter_dataframe)  # OutputFlag=OutputFlag

        train_error = training_value_error(splitter_dataframe)
        training_R2.append(R2_train)
        training_error.append(train_error)

        mdp_splitter_list.append((nc, deepcopy(splitter_dataframe).to_mdp()))

        # printing error and accuracy values
        if OutputFlag >= 1:
            split_bar.set_description('training Error: {}'.format(train_error))

        if testing:
            # compute updated predicted cluster
            test_splitter_dataframe.get_mdp_tree(splitter_dataframe)
            test_splitter_dataframe.compute_risk_error()

            R2_test = R2_value(test_splitter_dataframe)
            test_error = testing_value_error(test_splitter_dataframe)
            testing_R2.append(R2_test)
            testing_error.append(test_error)

            if test_error < best_testing_error:  # DEBUG
                best_mdp_predictor = splitter_dataframe.to_mdp()
                best_testing_error = test_error

            # printing error and accuracy values
            if OutputFlag >= 1:
                # split_bar.set_description('test value R2: {}'.format(R2_test))
                pass

            # printing error and accuracy values
            # if OutputFlag >= 2:
                # line1.set_description('testing value R2: {}'.format(R2_test))
                # line1.set_postfix('testing value error: {}'.format(test_error))

            # line1.set_postfix('training value error: {}'.format(train_error))

        # print('predictions:', get_predictions(df_new))
        # print(df_new.head())

        # split_bar.set_description("Splitting... |#Clusters:%s" %(nc))
        cont = False
        c, a = findContradiction(splitter_dataframe.error_transition_df, th)
        # print('Iteration',i+1, '| #Clusters=',nc+1, '------------------------')

        if c != -1:
            # if OutputFlag == 1:
            # print('Cluster Content')
            # print(df_new.groupby(
            # ['CLUSTER', 'OG_CLUSTER'])['ACTION'].count())

            # finding contradictions and splitting
            a, b = contradiction(splitter_dataframe.df, c, a)

            # if OutputFlag == 1:
            try:
                splitter_dataframe.df = split(splitter_dataframe.df, c, a, b, splitter_dataframe.features, nc, classification,
                                                  random_state=random_state)
            except ValueError:
                break

            cont = True
            nc += 1
        if not cont:
            break
        if nc >= it:
            break

    splitter_dataframe.n_clusters = nc

    # compute the MDP features
    splitter_dataframe.update_mdp()

    # train the decision tree
    splitter_dataframe.update_cluster()

    # compute horizon error & transition error
    splitter_dataframe.compute_risk_error()

    # compute transition error
    splitter_dataframe.compute_transition_error()

    # error and accuracy calculations
    R2_train = R2_value(splitter_dataframe)  # OutputFlag=OutputFlag

    train_error = training_value_error(splitter_dataframe)
    training_R2.append(R2_train)
    training_error.append(train_error)

    mdp_splitter_list.append((nc, deepcopy(splitter_dataframe).to_mdp()))

    if testing:
        # compute updated predicted cluster
        test_splitter_dataframe.get_mdp_tree(splitter_dataframe)
        test_splitter_dataframe.compute_risk_error()

        R2_test = R2_value(test_splitter_dataframe)
        test_error = testing_value_error(test_splitter_dataframe)
        testing_R2.append(R2_test)
        testing_error.append(test_error)

        if test_error < best_testing_error:  # DEBUG
            best_mdp_predictor = splitter_dataframe.to_mdp()

        # printing error and accuracy values
        # if OutputFlag >= 2:
            # line1.set_description('testing value R2: {}'.format(R2_test))
            # line1.set_postfix('testing value error: {}'.format(test_error))

    # printing error and accuracy values
    if OutputFlag >= 1:
        split_bar.set_description('training value R2: {}'.format(R2_train))

    # if OutputFlag == 1:
    #   print(df_new.groupby(['CLUSTER', 'OG_CLUSTER'])['ACTION'].count())

    # plotting functions
    # Plotting accuracy and value R2
    #    fig1, ax1 = plt.subplots()
    its = np.arange(k, nc + 1)
    #    ax1.plot(its, training_R2, label= "Training R2")
    #    if testing:
    #        ax1.plot(its, testing_R2, label = "Testing R2")
    #    #ax1.plot(its, training_acc, label = "Training Accuracy")
    #    #ax1.plot(its, testing_acc, label = "Testing Accuracy")
    #    if n>0:
    #        ax1.axvline(x=n,linestyle='--',color='r') #Plotting vertical line at #cluster =n
    #    ax1.set_ylim(0,1)
    #    ax1.set_xlabel('# of Clusters')
    #    ax1.set_ylabel('R2 or Accuracy %')
    #    ax1.set_title('R2 and Accuracy During Splitting')
    #    ax1.legend()

    # Plotting value error E((v_est - v_true)^2) FOR COVID: plotting MAPE
    # for training
    if plot:
        try:
            fig2, ax2 = plt.subplots()
            ax2.plot(its, training_error, label="Training Error")
            if testing:
                ax2.plot(its, testing_error, label="Testing Error")
            if n > 0:
                ax2.axvline(x=n, linestyle='--', color='r')  # Plotting vertical line at #cluster =n
            ax2.set_ylim(0)
            ax2.set_xlabel('# of Clusters')
            ax2.set_ylabel('Cases MAPE error')
            ax2.set_title('MAPE error by number of clusters')
            ax2.legend()
            if save:
                try:
                    plt.savefig(savepath)
                except:
                    pass
            if OutputFlag >= 2:
                plt.show(block=False)
            else:
                plt.close()
        except ValueError:
            pass

    df_train_error = pd.DataFrame(list(zip(its, training_error)), \
                                  columns=['Clusters', 'Error'])
    splitter_dataframe.training_error = df_train_error
    splitter_dataframe.R2_train = training_R2
    if testing:
        df_test_error = pd.DataFrame(list(zip(its, testing_error)), \
                                     columns=['Clusters', 'Error'])
        best_mdp_predictor.R2_test = testing_R2
        best_mdp_predictor.testing_error = df_test_error
        best_mdp_predictor.training_error = df_train_error
        best_mdp_predictor.R2_train = training_R2
        best_mdp_predictor.complete_path(splitter_dataframe.all_data.reset_index().set_index("ID"))
        return best_mdp_predictor, mdp_splitter_list

    return splitter_dataframe.to_mdp(), mdp_splitter_list



# (MDP FUNCTION)
# Fitting function for a single fold,
def fit_cv_fold(split_idx,
                df,
                clustering,
                n_clusters,
                clustering_distance_threshold,
                actions,
                pfeatures,
                reward_name,
                splitting_threshold,
                classification,
                n_iter,
                days_avg,
                horizon,
                test_horizon,
                error_computing,
                error_function_name,
                alpha,
                n,
                OutputFlag=0,
                random_state=1234,
                mode="ID",
                save=False,
                savepath="",
                plot=False):
    if mode == "ALL":

        idx, (train_idx, test_idx), features = split_idx  # train_idx, _ = split_idx  / _, train_idx = split_idx
        df_test = df.loc[train_idx].groupby("ID").tail(test_horizon+1).reset_index(drop=True).copy()
        df_train = df.loc[train_idx].groupby("ID").apply(lambda x: x.head(-(test_horizon+1))).reset_index(drop=True).copy()

    elif mode == "TIME_CV":

        idx, (train_idx, test_idx), features = split_idx
        df_train = pd.concat(
            [df.loc[train_idx],
             df.loc[test_idx].groupby("ID").apply(lambda x: x.head(-(test_horizon+1))).reset_index(drop=True)]
        ).copy().reset_index(drop=True)
        df_test = df.loc[test_idx].groupby("ID").tail(test_horizon+1).reset_index(drop=True).copy()

    elif mode == "ID":

        idx, (train_idx, test_idx), features = split_idx
        df_train = df.groupby("ID").apply(lambda x: x.head(-(test_horizon+1))).reset_index(drop=True).copy().reset_index(drop=True)
        df_test = df.loc[test_idx].groupby("ID").tail(test_horizon+1).reset_index(drop=True).copy()

    else:
        if OutputFlag >= 1:
            print("TrainingError : 'mode' must be a string : either 'TIME', 'ALL' or 'ID' ")
        raise MDPTrainingError

    #################################################################
    # Initialize Clusters

    splitter_df = MDP_Splitter(df_train,
                               all_data=df,
                               features=features,
                               reward_name=reward_name,
                               days_avg=days_avg,
                               clustering=clustering,
                               init_n_clusters=n_clusters,
                               distance_threshold=clustering_distance_threshold,
                               error_computing=error_computing,
                               error_function_name=error_function_name,
                               actions=actions,
                               horizon=horizon,
                               alpha=alpha,
                               random_state=random_state,
                               verbose=OutputFlag)

    test_splitter_df = MDP_Splitter(df_test,
                                    all_data=df,
                                    features=features,
                                    reward_name=reward_name,
                                    days_avg=days_avg,
                                    clustering=clustering,
                                    init_n_clusters=n_clusters,
                                    distance_threshold=clustering_distance_threshold,
                                    error_computing="horizon",
                                    error_function_name="exp_relative",
                                    mode="test",
                                    actions=actions,
                                    horizon=test_horizon,
                                    random_state=random_state,
                                    verbose=OutputFlag)

    # splitter_df.initializeClusters(["RISK", "AlphaRISK"])
    splitter_df.initializeClusters(reward=["{}_SCALE".format(reward_name)])

    # k = df_train['CLUSTER'].nunique()
    #################################################################

    #################################################################
    # Run Iterative Learning Algorithm

    try:
        trained_splitter_df, splitter_list = splitter(splitter_df,
                                                      splitting_threshold,
                                                      test_splitter_df,
                                                      testing=True,
                                                      classification=classification,
                                                      it=n_iter,
                                                      OutputFlag=OutputFlag,
                                                      n=n,
                                                      random_state=random_state,
                                                      plot=plot,
                                                      save=save,
                                                      savepath=os.path.join(savepath, "plot_{}.PNG".format(idx)))

        testing_error = trained_splitter_df.testing_error
        training_error = trained_splitter_df.training_error
        # df_train = trained_splitter_df.df_train

        return testing_error, training_error, trained_splitter_df, splitter_list
    except MDPPredictionError:
        return None, None, None, None


# (MDP GRID SEARCH FUNCTION)
# Fitting end evaluation function a single parameter set
def fit_eval_params(param_id_mdp,
                    data,
                    testing_data=None,
                    mode="TIME_CV",
                    ignore_errors=True):
    param_id, mdp = param_id_mdp
    if mdp.verbose >= 0:
        print("[MDP param set {}]: ".format(param_id), mdp.__repr__())
    try:
        mdp.fit(data, mode=mode)
        if mdp.save:
            savepath_model = os.path.join(mdp.savepath, mode, str(mdp), "mdp_model.pkl")
            save_model(mdp, savepath_model)

    except MDPTrainingError:
        if mdp.verbose >= 1:
            print("MDP Error fit: {}".format(mdp))
        if not ignore_errors:
            raise MDPTrainingError
        else:
            return param_id, mdp, np.inf

    # if no testing set
    if testing_data is None:
        training_error = mdp.CV_error
        return param_id, mdp, training_error

    else:
        validation_error = prediction_score(mdp, testing_data)
        return param_id, mdp, validation_error


def process_prediction_region(region, mdp, dates, region_set, model_key="mean", agg=False, from_first=False):

    if mdp.verbose >= 1:
        print("Current region:{}".format(region))
    try:
        assert region in region_set
    # the state doesn't appear not in the region set
    except AssertionError:
        if mdp.verbose >= 1:
            print("The region '{}' is not in the trained region set".format(region))

    # predict from each dates
    pred_df = pd.DataFrame(columns=[mdp.region_colname, 'TIME', mdp.target_colname])
    try:
        # start from the first date
        if from_first & mdp.keep_first:
            last_date = mdp.df_trained_first.loc[region, "TIME"]
        # start from the last date
        else:
            last_date = mdp.df_trained.loc[region, "TIME"]
    except KeyError:
        return pred_df

    for date in dates:
        try:
            pred = predict_region_date(mdp, (region, last_date), date, model_key=model_key,
                                       agg=agg, from_first=from_first, verbose=mdp.verbose)
            pred_df = pred_df.append({mdp.region_colname: region, "TIME": date, mdp.target_colname: pred}, ignore_index=True)
        except MDPPredictionError:
            pass
    return pred_df
