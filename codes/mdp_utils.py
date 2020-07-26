#%% Libraries
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
# from xgboost import XGBClassifier
from collections import Counter
from itertools import groupby
from operator import itemgetter
import os

import warnings
warnings.filterwarnings("ignore")

from codes.mdp_testing import R2_value_training, training_value_error,  \
    predict_cluster, R2_value_testing, testing_value_error, error_per_ID, MDPTrainingError, prediction_score

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


# initializeClusters() takes as input a dataframe, a time horizon T,
# a clustering algorithm, a number of clusters n_clusters,
# and a random seed (optional) and returns a dataframe
# with two new columns 'CLUSTER' and 'NEXT_CLUSTER'
def initializeClusters(df,  # pandas dataFrame: MUST contain a "RISK" column
                       clustering='Agglomerative',  # string: clustering algorithm
                       n_clusters= None,
                       distance_threshold= 0.1,  # number of clusters
                       random_state=0):  # random seed for the clustering
    df = df.copy()
    if clustering == 'KMeans':
        output = KMeans(
                n_clusters=n_clusters, random_state=random_state).fit(
                        np.array(df.RISK).reshape(-1, 1)).labels_
    elif clustering == 'Agglomerative':
        output = AgglomerativeClustering(
            n_clusters=n_clusters, distance_threshold=distance_threshold).fit(
                    np.array(df.RISK).reshape(-1, 1)).labels_
    elif clustering == 'Birch':
        output = Birch(
            n_clusters=n_clusters).fit(
                    np.array(df.RISK).reshape(-1, 1)).labels_
    else:
        output = LabelEncoder().fit_transform(np.array(df.RISK).reshape(-1, 1))
    df['CLUSTER'] = output
    df['NEXT_CLUSTER'] = df.groupby('ID')['CLUSTER'].shift(-1)
    return df
#############################################################################


#############################################################################
# Function for the Iterations

# findConstradiction() takes as input a dataframe and returns the tuple with
# initial cluster and action that have the most number of contradictions or
# (-1, -1) if no such cluster existss
def findContradiction(df, # pandas dataFrame
                      th): # integer: threshold split size
    X = df.loc[:, ['CLUSTER', 'NEXT_CLUSTER', 'ACTION']]
    X = X[X.NEXT_CLUSTER != 'None']
    count = X.groupby(['CLUSTER', 'ACTION'])['NEXT_CLUSTER'].nunique()
    contradictions = list(count[list(count > 1)].index)
    if len(contradictions) > 0:
        ncontradictions = [sum(list(X.query('CLUSTER == @i[0]').query(
            'ACTION == @i[1]').groupby('NEXT_CLUSTER')['ACTION'].count().
                                    sort_values(ascending=False))[1:]) for i in contradictions]
        if max(ncontradictions) > th:
            selectedCont = contradictions[ncontradictions.index(
                max(ncontradictions))]
            return(selectedCont)
    return((-1, -1))


# contradiction() outputs one found contradiction given a dataframe,
# a cluster and a an action or (None, None) if none is found
def contradiction(df,  # pandas dataFrame
                  i,  # integer: initial clusters
                  a):  # integer: action taken
    nc = list(df.query('CLUSTER == @i').query(
            'ACTION == @a').query('NEXT_CLUSTER != "None"')['NEXT_CLUSTER'])
    if len(nc) == 1:
        return None, None
    else:
        return a, multimode(nc)[0]


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
                    df['NEXT_CLUSTER'] != 'None')]
    g3 = df[(df['CLUSTER'] == i) & (
            ((df['ACTION'] == a) & (df['NEXT_CLUSTER'] == 'None')) | (
                    df['ACTION'] != a))]
    groups = [g1, g2, g3]
    data = {}

    for j in range(len(groups)):

        d = pd.DataFrame(groups[j].iloc[:, 2:2+pfeatures].values.tolist())

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
    newids = ids-1
    df.loc[(df.index.isin(newids)) & (df['ID']== df['ID'].shift(-1)), 'NEXT_CLUSTER'] = k

    return df


# (MDP GRID SEARCH FUNCTION)
# Splitting function from the MDP learning algorithm
def splitter(df,  # pandas dataFrame
             pfeatures,  # integer: number of features
             th,  # integer: threshold for minimum split
             df_test=None,
             testing=False,
             classification='LogisticRegression',  # string: classification alg
             it=6,  # integer: max number of clusters
             h=5,
             error_computing="horizon",
             alpha=1e-5,
             OutputFlag=1,
             n=-1,
             random_state=0,
             plot=False,
             save=False,
             savepath=None):  #If we plot error
    # initializing lists for error & accuracy data
    training_R2 = []
    testing_R2 = []
    # training_acc = []
    # testing_acc = []
    testing_error = []
    training_error = []
    k = df['CLUSTER'].nunique() #initial number of clusters
    nc = k
    df_new = deepcopy(df)

    # Setting progress bar--------------
    if OutputFlag >= 1:
        split_bar = tqdm(range(it-k))
    else:
        split_bar = range(it-k)
    # split_bar.set_description("Splitting...")
    # Setting progress bar--------------
    for i in split_bar:
        # split_bar.set_description("Splitting... |#Clusters:%s" %(nc))
        cont = False
        c, a = findContradiction(df_new, th)
        # print('Iteration',i+1, '| #Clusters=',nc+1, '------------------------')
        if c != -1:
            #if OutputFlag == 1:
                #print('Cluster Content')
                #print(df_new.groupby(
                            #['CLUSTER', 'OG_CLUSTER'])['ACTION'].count())

            # finding contradictions and splitting
            a, b = contradiction(df_new, c, a)

            # if OutputFlag == 1:
                # print('Cluster splitted', c,'| Action causing contradiction:', a, '| Cluster most elements went to:', b)
            df_new = split(df_new, c, a, b, pfeatures, nc, classification, random_state=random_state)

            # error and accuracy calculations

            R2_train = R2_value_training(df_new, OutputFlag=OutputFlag)

            if testing:
                model = predict_cluster(df_new, pfeatures)
                R2_test = R2_value_testing(df_test, df_new, model, pfeatures, OutputFlag=OutputFlag)
                test_error = testing_value_error(df_test, df_new, model, pfeatures,
                                                 error_computing="exponential",
                                                 alpha=alpha,
                                                 h=h,
                                                 OutputFlag=OutputFlag)
                testing_R2.append(R2_test)
                testing_error.append(test_error)
            # train_acc = training_accuracy(df_new)[0]
            # test_acc = testing_accuracy(df_test, df_new, model, pfeatures)[0]
            train_error = training_value_error(df_new,
                                               error_computing=error_computing,
                                               alpha=alpha,
                                               h=h,
                                               OutputFlag=OutputFlag)
            training_R2.append(R2_train)
            training_error.append(train_error)
            # training_acc.append(train_acc)
            # testing_acc.append(test_acc)

            # printing error and accuracy values
            # if OutputFlag == 1:
            #   print('training value R2:', R2_train)
            #   print('training accuracy:', train_acc)
            #   print('testing accuracy:', test_acc)
            #   print('training value error:', train_error)
            #   if testing:
            #       print('testing value R2:', R2_test)
            #       print('testing value error:', test_error)
            # print('predictions:', get_predictions(df_new))
            # print(df_new.head())
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
    its = np.arange(k+1, nc+1)
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
            ax2.axvline(x=n, linestyle='--', color='r') #Plotting vertical line at #cluster =n
        ax2.set_ylim(0)
        ax2.set_xlabel('# of Clusters')
        ax2.set_ylabel('Cases MAPE error')
        ax2.set_title('MAPE error by number of clusters')
        ax2.legend()
        if save:
            plt.savefig(savepath)
        if OutputFlag >= 2:
            plt.show(block=False)
        else:
            plt.close()

    df_train_error = pd.DataFrame(list(zip(its, training_error)), \
                                  columns=['Clusters', 'Error'])
    if testing:
        df_test_error = pd.DataFrame(list(zip(its, testing_error)), \
                                  columns=['Clusters', 'Error'])
        return df_new, df_train_error, df_test_error

    return df_new, training_error, testing_error


# (MDP FUNCTION)
# Fitting function for a single fold,
def fit_cv_fold(split_idx,
                df,
                clustering,
                n_clusters,
                clustering_distance_threshold,
                pfeatures,
                splitting_threshold,
                classification,
                n_iter,
                horizon,
                error_computing,
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
        df_test = df.loc[train_idx].groupby("ID").tail(horizon).reset_index(drop=True).copy()
        df_train = df.loc[train_idx].groupby("ID").apply(lambda x: x.head(-horizon)).reset_index(drop=True).copy()

    elif mode == "TIME_CV":

        idx, (train_idx, test_idx) = split_idx
        df_train = pd.concat(
            [df.loc[train_idx],
             df.loc[test_idx].groupby("ID").apply(lambda x: x.head(-horizon)).reset_index(drop=True)]
        ).copy().reset_index(drop=True)
        df_test = df.loc[test_idx].groupby("ID").tail(horizon).reset_index(drop=True).copy()

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
    df_train = initializeClusters(df_train,
                                  clustering=clustering,
                                  n_clusters=n_clusters,
                                  distance_threshold=clustering_distance_threshold,
                                  random_state=random_state)
    # k = df_train['CLUSTER'].nunique()
    #################################################################

    #################################################################
    # Run Iterative Learning Algorithm

    df_train, training_error, testing_error = splitter(df_train,
                                                       pfeatures,
                                                       splitting_threshold,
                                                       df_test,
                                                       testing=True,
                                                       classification=classification,
                                                       it=n_iter,
                                                       h=horizon,
                                                       error_computing=error_computing,
                                                       alpha=alpha,
                                                       OutputFlag=OutputFlag,
                                                       n=n,
                                                       random_state=random_state,
                                                       plot=plot,
                                                       save=save,
                                                       savepath=os.path.join(savepath, "plot_{}.PNG".format(idx)))

    m = predict_cluster(df_train, pfeatures)
    df_err, E_v = error_per_ID(df_test, df_train, m, pfeatures, relative=True, h=horizon, OutputFlag=OutputFlag)

    return testing_error, training_error, df_err, E_v


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
            savepath_model = os.path.join(mdp.savepath, mode, str(mdp), "mdp_model.pickle")
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

