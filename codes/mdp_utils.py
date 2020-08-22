# %% Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # progress bar
from copy import deepcopy
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from collections import Counter
from itertools import groupby
from operator import itemgetter
import os

import warnings

warnings.filterwarnings("ignore")

from codes.mdp_testing import R2_value, training_value_error, \
    predict_cluster, testing_value_error, error_per_ID, MDPTrainingError, prediction_score, get_MDP
from codes.data_utils import save_model


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
        self.training_error = None
        self.testing_error = None
        self.R2_train = None
        self.R2_test = None
        self.columns = None
        self.pfeatures = None
        self.model = None


class MDP_Splitter:

    def __init__(self,
                 df,
                 pfeatures,
                 error_computing="horizon",
                 horizon=5,
                 alpha=1e-5,
                 error_function_name="relative",
                 clustering='Agglomerative',
                 actions=[0],
                 init_n_clusters=None,
                 distance_threshold=0.1,
                 random_state=42,
                 verbose=0):
        self.df = df.sort_values(["ID", "TIME"]).copy()
        self.clustering = clustering
        self.init_n_clusters = init_n_clusters
        self.distance_threshold = distance_threshold
        self.pfeatures = pfeatures
        self.error_computing = error_computing
        self.horizon = horizon
        self.alpha = alpha
        self.random_state = random_state
        self.actions = actions

        self.error_function_name = error_function_name
        self.error_eval = get_error_eval(self.error_function_name)
        self.verbose = verbose

        self.model = None
        self.P_df = None
        self.R_df = None
        self.error_transition_df = None
        # self.accuracy_cluster = {"train": [], "test": []}
        self.n_clusters = None
        self.R2_train = None
        self.R2_test = None
        self.training_error = None
        self.testing_error = None

    # initializeClusters() takes as input a dataframe, a time horizon T,
    # a clustering algorithm, a number of clusters n_clusters,
    # and a random seed (optional) and returns a dataframe
    # with two new columns 'CLUSTER' and 'NEXT_CLUSTER'
    def initializeClusters(self):  # random seed for the clustering
        if self.clustering == 'KMeans':
            output = KMeans(
                n_clusters=self.init_n_clusters, random_state=self.random_state).fit(
                np.array(self.df.RISK).reshape(-1, 1)).labels_
        elif self.clustering == 'Agglomerative':
            output = AgglomerativeClustering(
                n_clusters=self.init_n_clusters, distance_threshold=self.distance_threshold).fit(
                np.array(self.df.RISK).reshape(-1, 1)).labels_
        elif self.clustering == 'Birch':
            output = Birch(
                n_clusters=self.init_n_clusters).fit(
                np.array(self.df.RISK).reshape(-1, 1)).labels_
        else:
            output = LabelEncoder().fit_transform(np.array(self.df.RISK).reshape(-1, 1))
        self.df['CLUSTER'] = output
        self.df['NEXT_CLUSTER'] = self.df.groupby('ID')['CLUSTER'].shift(-1)

    def update_mdp(self):
        # compute the estimated rewards and transition matrix
        self.P_df, self.R_df = get_MDP(self.df, self.actions, self.pfeatures, n_cluster=self.n_clusters,
                                       OutputFlag=self.verbose)

        # Add estimated next cluster [MUST BE IMPROVE]

    def get_mdp_tree(self, other):
        # (put comment)
        self.P_df = other.P_df.copy()
        self.R_df = other.R_df.copy()
        self.model = other.model

    def compute_risk_error(self, mode="train"):

        df = self.df.copy()
        df["EST_H_NEXT_CLUSTER"] = self.model.predict(df.iloc[:, 2:2 + self.pfeatures])

        # # compute accuracy score of cluster prediction
        if mode == "train":
            df["TRUE_H_NEXT_CLUSTER"] = df.groupby("ID", sort=False)["CLUSTER"].shift(-self.horizon)
        #     pred = self.df[["EST_H_NEXT_CLUSTER", "CLUSTER"]].dropna().copy()
        #     self.accuracy_cluster[mode].append(accuracy_score(pred["EST_H_NEXT_CLUSTER"], pred["CLUSTER"]))
        #     del pred

        df["EST_H_NEXT_RISK"] = 0.
        df["EST_H_ERROR"] = 0.

        for h in range(1, self.horizon + 1):  # [TO ADAPT WITH EXPONENTIAL]
            # COMPUTE THE NEXT CLUSTER
            df = pd.merge(
                df,
                self.P_df.reset_index().rename(columns={"ACTION" : "ACTION_", "CLUSTER" : "CLUSTER_"}),
                left_on=["EST_H_NEXT_CLUSTER", "ACTION"],
                right_on=["CLUSTER_", "ACTION_"],
                how="left").copy().drop(["CLUSTER_", "ACTION_"],
                                        axis=1).drop(["EST_H_NEXT_CLUSTER"],
                                                     axis=1).rename(columns={"TRANSITION_CLUSTER": "EST_H_NEXT_CLUSTER"})

            # COMPLETE MISSING TRANSITION WITH THE LAST AVAILABLE CLUSTER
            df["EST_H_NEXT_CLUSTER"] = df.groupby(['ID'], sort=False).apply(lambda group: group.ffill())[
                'EST_H_NEXT_CLUSTER'].values

            # COMPUTE THE ONE STEP ESTIMATED RISK
            df["EST_H_NEXT_RISK"] += pd.merge(
                df.drop('EST_H_NEXT_RISK', axis=1),
                self.R_df.reset_index(),
                left_on="EST_H_NEXT_CLUSTER",
                right_on="CLUSTER",
                how="left")["EST_RISK"].values

            df["TRUE_H_NEXT_RISK"] = df.groupby("ID", sort=False).rolling(window=h, min_periods=h)[
                "RISK"].sum().values
            df["TRUE_H_NEXT_RISK"] = df.groupby("ID", sort=False)["TRUE_H_NEXT_RISK"].shift(-h)

            # COMPUTE THE ONE STEP ERROR
            df["EST_H_ERROR"] += self.error_eval(df["EST_H_NEXT_RISK"].values,
                                                 df["TRUE_H_NEXT_RISK"].values)

        self.df = df.copy()

    def update_cluster(self):
        self.model = predict_cluster(self.df, self.pfeatures)

    def compute_transition_error(self):
        n_df = self.df.groupby(["CLUSTER", "ACTION"])["NEXT_CLUSTER"].nunique()
        e_df = self.df.groupby(["CLUSTER", "ACTION"])["EST_H_ERROR"].max() / self.horizon
        self.error_transition_df = (np.log(n_df) * e_df).copy()
        pass

    def to_mdp(self):
        mdp_predictor = MDPPredictor()
        mdp_predictor.P_df = self.P_df
        mdp_predictor.R_df = self.R_df
        mdp_predictor.training_error = self.training_error
        mdp_predictor.testing_error = self.testing_error
        mdp_predictor.R2_train = self.R2_train
        mdp_predictor.R2_test = self.R2_test
        mdp_predictor.columns = self.df.iloc[:, 2: self.pfeatures + 2].columns.tolist()
        mdp_predictor.pfeatures = self.pfeatures
        mdp_predictor.classifier = self.model
        mdp_predictor.verbose = self.verbose

        return mdp_predictor



#############################################################################


#############################################################################
# Error functions

# get_error_eval (MISSING DESCRIPTION)
def get_error_eval(function_name):
    try:
        assert function_name in {"sym_abs_relative", "absolute",
                                 "relative", "exp_relative"}, "the error function name must be " \
                                                                              "either 'sym_abs_relative', 'absolute', " \
                                                                              " 'relative' or 'exp_relative'"
    except AssertionError:
        raise MDPSplitterError

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


#############################################################################
# Function for the Iterations

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
        nc = nc.groupby("NEXT_CLUSTER")["EST_H_ERROR"].max()
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
          pfeatures,  # integer: number of features
          k,  # integer: intedexer for next cluster
          classification='LogisticRegression',
          random_state=0):  # string: classification aglo

    g1 = df[(df['CLUSTER'] == i) & (
            df['ACTION'] == a) & (df['NEXT_CLUSTER'] == c)]
    g2 = df[(df['CLUSTER'] == i) & (
            df['ACTION'] == a) & (df['NEXT_CLUSTER'] != c) & (
                df['NEXT_CLUSTER'].notnull())]
    g3 = df[(df['CLUSTER'] == i) & (
            ((df['ACTION'] == a) & (df['NEXT_CLUSTER'].notnull())) | (
            df['ACTION'] != a))]
    groups = [g1, g2, g3]
    data = {}

    for j in range(len(groups)):
        d = pd.DataFrame(groups[j].iloc[:, 2:2 + pfeatures].values.tolist())

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
        m = DecisionTreeClassifier(random_state=random_state)
    elif classification == 'RandomForestClassifier':
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
             pfeatures,  # integer: number of features
             th,  # integer: threshold for minimum split
             test_splitter_dataframe=None,
             testing=False,
             classification='LogisticRegression',  # string: classification alg
             it=6,  # integer: max number of clusters
             OutputFlag=1,
             n=-1,
             random_state=0,
             plot=False,
             save=False,
             savepath=None):  # If we plot error

    # initializing lists for error & accuracy data
    training_R2 = []
    testing_R2 = []
    testing_error = []
    training_error = []
    nc = splitter_dataframe.df['CLUSTER'].nunique()  # initial number of clusters
    k = nc
    # df_new = deepcopy(df)

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
        splitter_dataframe.compute_risk_error("train")

        # compute transition error
        splitter_dataframe.compute_transition_error()

        # error and accuracy calculations
        R2_train = R2_value(splitter_dataframe)  # OutputFlag=OutputFlag

        train_error = training_value_error(splitter_dataframe)
        training_R2.append(R2_train)
        training_error.append(train_error)

        if testing:
            # compute updated predicted cluster
            test_splitter_dataframe.get_mdp_tree(splitter_dataframe)
            test_splitter_dataframe.compute_risk_error("test")

            R2_test = R2_value(test_splitter_dataframe)
            test_error = testing_value_error(test_splitter_dataframe)
            testing_R2.append(R2_test)
            testing_error.append(test_error)

            # printing error and accuracy values
            # if OutputFlag >= 2:
                # line1.set_description('testing value R2: {}'.format(R2_test))
                # line1.set_postfix('testing value error: {}'.format(test_error))

        # printing error and accuracy values
        if OutputFlag >= 2:
            split_bar.set_description('training value R2: {}'.format(R2_train))
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
            splitter_dataframe.df = split(splitter_dataframe.df.copy(), c, a, b, pfeatures, nc, classification,
                                              random_state=random_state)

            cont = True
            nc += 1
        if not cont:
            break
        if nc >= it:
            break
    # if OutputFlag == 1:
    #   print(df_new.groupby(['CLUSTER', 'OG_CLUSTER'])['ACTION'].count())

    # plotting functions
    # Plotting accuracy and value R2
    #    fig1, ax1 = plt.subplots()
    its = np.arange(k + 1, nc + 1)
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

    df_train_error = pd.DataFrame(list(zip(its, training_error)), \
                                  columns=['Clusters', 'Error'])
    splitter_dataframe.training_error = df_train_error
    splitter_dataframe.R2_train = training_R2
    if testing:
        df_test_error = pd.DataFrame(list(zip(its, testing_error)), \
                                     columns=['Clusters', 'Error'])
        splitter_dataframe.testing_error = df_test_error
        splitter_dataframe.R2_test = testing_R2

    return splitter_dataframe.to_mdp()


# (MDP FUNCTION)
# Fitting function for a single fold,
def fit_cv_fold(split_idx,
                df,
                clustering,
                n_clusters,
                clustering_distance_threshold,
                actions,
                pfeatures,
                splitting_threshold,
                classification,
                n_iter,
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

        idx, (train_idx, test_idx) = split_idx  # train_idx, _ = split_idx  / _, train_idx = split_idx
        df_test = df.loc[train_idx].groupby("ID").tail(test_horizon+1).reset_index(drop=True).copy()
        df_train = df.loc[train_idx].groupby("ID").apply(lambda x: x.head(-(test_horizon+1))).reset_index(drop=True).copy()

    elif mode == "TIME_CV":

        idx, (train_idx, test_idx) = split_idx
        df_train = pd.concat(
            [df.loc[train_idx],
             df.loc[test_idx].groupby("ID").apply(lambda x: x.head(-(test_horizon+1))).reset_index(drop=True)]
        ).copy().reset_index(drop=True)
        df_test = df.loc[test_idx].groupby("ID").tail(test_horizon+1).reset_index(drop=True).copy()

    elif mode == "ID":

        idx, (train_idx, test_idx) = split_idx
        df_train = df.loc[train_idx].copy()
        df_test = df.loc[test_idx].copy()

    else:
        if OutputFlag >= 1:
            print("TrainingError : 'mode' must be a string : either 'TIME', 'ALL' or 'ID' ")
        raise MDPTrainingError

    #################################################################
    # Initialize Clusters

    splitter_df = MDP_Splitter(df_train,
                               pfeatures=pfeatures,
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
                                    pfeatures=pfeatures,
                                    clustering=clustering,
                                    init_n_clusters=n_clusters,
                                    distance_threshold=clustering_distance_threshold,
                                    error_computing="horizon",
                                    error_function_name="exp_relative",
                                    actions=actions,
                                    horizon=test_horizon,
                                    random_state=random_state,
                                    verbose=OutputFlag)

    splitter_df.initializeClusters()

    # k = df_train['CLUSTER'].nunique()
    #################################################################

    #################################################################
    # Run Iterative Learning Algorithm

    trained_splitter_df = splitter(splitter_df,
                                   pfeatures,
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


    # try:
    #    m = predict_cluster(df_train, pfeatures)
    #    P_df, R_df = splitter_df.P_df, splitter_df.R_df
    #    df_err, E_v = error_per_ID(df_test, m, pfeatures, P_df, R_df, relative=True, h=test_horizon, OutputFlag=OutputFlag)
    #    return testing_error, training_error, df_err, E_v, trained_splitter_df
    # except KeyError:
    return testing_error, training_error, trained_splitter_df


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
