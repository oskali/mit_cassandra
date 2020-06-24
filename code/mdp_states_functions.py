# -*- coding: utf-8 -*-
"""
This file is the main file to run the MDP clustering algorithm

Specific to State COVID-19 Research.

Specific verion includes new updates on clustering functions.

Created on Sun April 7 18:51:20 2020

@author: omars
"""
#############################################################################
# Load Libraries
import pandas as pd
import numpy as np
import matplotlib as plt
import random
# from tqdm import tqdm #progress bar
import binascii
from copy import deepcopy
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, GroupKFold
#from xgboost import XGBClassifier
from collections import Counter
from itertools import groupby
from operator import itemgetter
from datetime import datetime

from mdp_testing import *
#############################################################################


#############################################################################
# Funtions for Initialization


# createSamples() takes the original dataframe from combined data, 
# the names of columns of features to keep, the treshold values to determine
# what type of action is made based on the FIRST feature of features_cols, 
# d_avg the number of days used to compress datapoints, and returns a data frame with 
# desired features and history, ratio values and history, and 'RISK' and 'ACTION'
# returns new dataframe with only the desired columns, number of features considered
def createSamples(df,#, # dataframe: original full dataframe
                  #new_cols, # str list: names of columns to be considered
                  target_col, # str: col name of target (i.e. 'deaths')
                  region_col, # str, col name of region (i.e. 'state')
                  date_col, # str, col name of time (i.e. 'date')
                  features_cols, # list of str: i.e. (['mobility', 'testing'])
                  action_thresh, # int list: defining size of jumps in stringency
                  #d_delay, # int: day lag before calculating death impact
                  d_avg): # int: # of days to average when reporting death
    
    df.rename(columns={date_col: 'TIME'}, inplace = True)
    df = df[df[region_col]!='Guam']
    df = df[df[region_col]!='Northern Mariana Islands']
    df = df[df[region_col]!='Puerto Rico']
    df = df[df[region_col]!='Diamond Princess']
    df = df[df[region_col]!='Grand Princess']
    df = df[df[region_col]!='American Samoa']
    df = df[df[region_col]!='Virgin Islands']
    

    #new_cols = ['state', 'date', 'cases', 'mobility_score']
    if target_col not in features_cols:
        new_cols = [region_col] + ['TIME'] + [target_col] + features_cols
    else:
        new_cols = [region_col] + ['TIME'] + features_cols
    df_new = df[new_cols]

    #df_new.rename(columns = {df_new.columns[1]: 'TIME'}, inplace = True)
    ids = df_new.groupby([region_col]).ngroup()
    df_new.insert(0, 'ID', ids, True)

    #print(df.columns)
    df_new.loc[:, ['TIME']]= pd.to_datetime(df_new['TIME'])
    df_new = df_new.sort_values(by=['ID', 'TIME'])
    df_new = df_new.set_index(['TIME'])
    # print(df_new)

    # calculating stringency based on sum of actions
    #df['StringencyIndex'] = df.iloc[:, 3:].sum(axis=1)

    # add a column for action, categorizing by change in stringency index
    #df['StringencyChange'] = df['StringencyIndex'].shift(-1) - df['StringencyIndex']
    #df.loc[df['ID'] != df['ID'].shift(-1), 'StringencyChange'] = 0
    #df.loc[df['StringencyIndex'] == '', 'StringencyChange'] = 0

    #print(df.loc[df['ID']=='California'])

    # resample data according to # of days
    g = df_new.groupby(['ID'])
    cols = df_new.columns
    #print('cols', cols)
    dictio = {i:'last' for i in cols}
    for key in set([target_col]+features_cols):
        dictio[key] = 'mean'
    #dictio['StringencyChange'] = 'sum'
    #del dictio['TIME']
    df_new = g.resample('%sD' %d_avg).agg(dictio)
    #df_new = g.resample('3D').mean()
    # print('new', df_new)
    df_new = df_new.drop(columns=['ID'])
    df_new = df_new.reset_index()

    # shifting cases by a d_delay value
    #df['Cases_Delay'] = df['cases'].shift(-d_delay)

    # averaging mobility score
    #df_new['mobility_score'] = df_new['mobility_score'].rolling(4).sum()/4

    
    # creating target-1, target-2, etc.
    #df_new[target_col+'-1'] = df_new[target_col].shift(1)
    #df_new[target_col+'-2'] = df_new[target_col].shift(2)

    # creating mobility-1, mobility-2 etc.
    for f in features_cols:
        df_new[f+'-1'] = df_new[f].shift(1)
        df_new[f+'-2'] = df_new[f].shift(2)

    # creating r_t, r_t-1, etc ratio values from cases
    df_new = df_new[df_new[target_col] != 0]
    df_new['r_t'] = df_new[target_col]/df_new[target_col].shift(1)
    df_new['r_t-1'] = df_new['r_t'].shift(1)
    df_new['r_t-2'] = df_new['r_t'].shift(2)

    df_new.loc[df_new['ID'] != df_new['ID'].shift(1), \
               ['r_t', 'r_t-1']] = 0
    df_new.loc[df_new['ID'] != df_new['ID'].shift(2), \
               ['r_t-1', 'r_t-2']] = 0
    df_new.loc[df_new['ID'] != df_new['ID'].shift(3), \
               ['r_t-2']] = 0
        
    for f in features_cols:
        df_new.loc[df_new['ID'] != df_new['ID'].shift(1), \
               [f+'-1', f+'-2']] = df_new[f]
        
        df_new.loc[df_new['ID'] != df_new['ID'].shift(2), \
           [f+'-2']] = df_new[f+'-1']
        
    

    # Here we assign initial clustering by r_t
    df_new['RISK'] = np.log(df_new['r_t'])


    # create action
    if len(action_thresh) == 0:
        df_new['ACTION'] = 0
        pfeatures = len(df_new.columns)-5
    else:
        actions = list(range(0, len(action_thresh)-1)) #[0, 1] #[0, 5000, 100000]
        df_new[features_cols[0]+'_change'] = df_new[features_cols[0]+'-1']-\
            df_new[features_cols[0]+'-2']
        df_new['ACTION'] = pd.cut(df_new[features_cols[0]+'_change'], bins = action_thresh, right = False, labels = actions)
        pfeatures = len(df_new.columns)-6

    df_new = df_new[df_new['r_t'] != 0]
    df_new = df_new.reset_index()
    df_new = df_new.drop(columns=['index'])
    # moving region col to the end, since not a feature
    if target_col not in features_cols:
        df_new = df_new[[c for c in df_new if c not in [region_col, target_col]]
           + [region_col] + [target_col]]
        pfeatures -= 1
    else:
        df_new = df_new[[c for c in df_new if c not in [region_col]]
           + [region_col]]
    
    # Drop all rows with empty cells
    #df_new.dropna(inplace=True)

    return df_new, pfeatures



# split_train_test_by_id() takes in a dataframe of all the data,
# returns Testing and Training dataset dataframes with the ratio of testing
# data defined by float test_ratio
def split_train_test_by_id(data, # dataframe: all the data
                           test_ratio, # float: portion of data for testing
                           id_column): # str: name of identifying ID column

    def test_set_check(identifier, test_ratio):
        return binascii.crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]


# initializeClusters() takes as input a dataframe, a time horizon T,
# a clustering algorithm, a number of clusters n_clusters,
# and a random seed (optional) and returns a dataframe
# with two new columns 'CLUSTER' and 'NEXT_CLUSTER'
def initializeClusters(df,  # pandas dataFrame: MUST contain a "RISK" column
                       clustering='Agglomerative',  # string: clustering algorithm
                       n_clusters= None,
                       distance_threshold= 0.1,# number of clusters
                       random_state=0):  # random seed for the clustering
    df = df.copy()
    if clustering == 'KMeans':
        output = KMeans(
                n_clusters=n_clusters, random_state=random_state).fit(
                        np.array(df.RISK).reshape(-1, 1)).labels_
    elif clustering == 'Agglomerative':
        output = AgglomerativeClustering(
            n_clusters=n_clusters, distance_threshold = distance_threshold).fit(
                    np.array(df.RISK).reshape(-1, 1)).labels_
    elif clustering == 'Birch':
        output = Birch(
            n_clusters=n_clusters).fit(
                    np.array(df.RISK).reshape(-1, 1)).labels_
    else:
        output = LabelEncoder().fit_transform(np.array(df.RISK).reshape(-1, 1))
    df['CLUSTER'] = output
    df['NEXT_CLUSTER'] = df['CLUSTER'].shift(-1)
    df.loc[df['ID'] != df['ID'].shift(-1), 'NEXT_CLUSTER'] = 'None'
    return(df)
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
        return (None, None)
    else:
        return a, multimode(nc)[0]

# multimode() returns a list of the most frequently occurring values.
# Will return more than one result if there are multiple modes
# or an empty list if *data* is empty.
def multimode(data):
    counts = Counter(iter(data)).most_common()
    maxcount, mode_items = next(groupby(counts, key=itemgetter(1)), (0, []))
    return list(map(itemgetter(0), mode_items))

# split() takes as input a dataframe, an initial cluster, an action, a target
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
          classification='LogisticRegression'):  # string: classification aglo

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

    tr_X = training.iloc[:, :-1]
    tr_y = training.iloc[:, -1:]

    if classification == 'LogisticRegression':
        m = LogisticRegression(solver='liblinear')
    elif classification == 'LogisticRegressionCV':
        m = LogisticRegressionCV()
    elif classification == 'DecisionTreeClassifier':
        m = DecisionTreeClassifier()
    elif classification == 'RandomForestClassifier':
        m = RandomForestClassifier()
    elif classification == 'XGBClassifier':
        m = XGBClassifier()

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

    return(df)



#############################################################################

# splitter() is the wrap-up function. Takes as parameters a dataframe df,
# a time-horizon T, a number of features pfeatures, an indexer k, and a max
# number of iterations and performs the algorithm until all contradictions are
# resolved or until the max number of iterations is reached
# Plots the trajectory of testing metrics during splitting process
# Returns the final resulting dataframe
def splitter(df,  # pandas dataFrame
             pfeatures,  # integer: number of features
             th, # integer: threshold for minimum split
             df_test = None,
             testing = False,
             classification='LogisticRegression',  # string: classification alg
             it=6, # integer: max number of clusters
             h=5,
             OutputFlag = 1,
             n=-1,
             plot = False):  #If we plot error
    # initializing lists for error & accuracy data
    training_R2 = []
    testing_R2 = []
    #training_acc = []
    #testing_acc = []
    testing_error = []
    training_error = []
    k = df['CLUSTER'].nunique() #initial number of clusters
    nc = k
    df_new = deepcopy(df)

    # Setting progress bar--------------
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
            df_new = split(df_new, c, a, b, pfeatures, nc, classification)

            # error and accuracy calculations

            R2_train = R2_value_training(df_new)

            if testing:
                model = predict_cluster(df_new, pfeatures)
                R2_test = R2_value_testing(df_test, df_new, model, pfeatures)
                test_error = testing_value_error(df_test, df_new, model, pfeatures, relative=True, h=h)
                testing_R2.append(R2_test)
                testing_error.append(test_error)
            #train_acc = training_accuracy(df_new)[0]
            #test_acc = testing_accuracy(df_test, df_new, model, pfeatures)[0]
            train_error = training_value_error(df_new, relative=True, h=h)
            training_R2.append(R2_train)
            training_error.append(train_error)
            #training_acc.append(train_acc)
            #testing_acc.append(test_acc)

            # printing error and accuracy values
            # if OutputFlag == 1:
                # print('training value R2:', R2_train)
                #print('training accuracy:', train_acc)
                #print('testing accuracy:', test_acc)
                # print('training value error:', train_error)
                # if testing:
                    # print('testing value R2:', R2_test)
                    # print('testing value error:', test_error)
            #print('predictions:', get_predictions(df_new))
            #print(df_new.head())
            cont = True
            nc += 1
        if not cont:
            break
        if nc >= it:
            break
    #if OutputFlag == 1:
        #print(df_new.groupby(['CLUSTER', 'OG_CLUSTER'])['ACTION'].count())


    # plotting functions
    ## Plotting accuracy and value R2
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
    ## Plotting value error E((v_est - v_true)^2) FOR COVID: plotting MAPE
    if plot:
        fig2, ax2 = plt.subplots()
        ax2.plot(its, training_error, label = "Training Error")
        if testing:
            ax2.plot(its, testing_error, label = "Testing Error")
        if n>0:
            ax2.axvline(x=n,linestyle='--',color='r') #Plotting vertical line at #cluster =n
        ax2.set_ylim(0)
        ax2.set_xlabel('# of Clusters')
        ax2.set_ylabel('Cases MAPE error')
        ax2.set_title('MAPE error by number of clusters')
        ax2.legend()
        plt.show()
    
    
    df_train_error = pd.DataFrame(list(zip(its, training_error)), \
                                  columns = ['Clusters', 'Error'])
    if testing:
        df_test_error = pd.DataFrame(list(zip(its, testing_error)), \
                                  columns = ['Clusters', 'Error'])
        return (df_new,df_train_error,df_test_error)
    
    return(df_new,training_error,testing_error)

#############################################################################
# Splitter algorithm with cross-validation
def fit_CV(df,
          pfeatures,
          th,
          clustering,
          distance_threshold,
          classification,
          n_iter,
          n_clusters,
          random_state,
          h=5,
          OutputFlag = 0,
          cv=5,
          n=-1,
          plot = False):

    df_training_error = pd.DataFrame(columns=['Clusters'])
    df_testing_error = pd.DataFrame(columns=['Clusters'])
    testing_errors = []
    
    gkf = GroupKFold(n_splits=cv)

    # shuffle ID's and create a new column 'ID_shuffle'
    random.seed(datetime.now())
    g = [df for _, df in df.groupby('ID')]
    random.shuffle(g)
    df = pd.concat(g).reset_index(drop=True)
    ids = df.groupby(['ID'], sort=False).ngroup()
    df['ID_shuffle'] = ids
    # cv_bar.set_description("Cross-Validation...")
    for train_idx, test_idx in gkf.split(df, y=None, groups=df['ID_shuffle']):
        # cv_bar.set_description("Cross-Validation... | Test set # %i" %i)

        df_train = df[df.index.isin(train_idx)]
        df_test = df[df.index.isin(test_idx)]
        #################################################################
        # Initialize Clusters
        df_init = initializeClusters(df_train,
                                clustering=clustering,
                                n_clusters=n_clusters,
                                distance_threshold = distance_threshold,
                                random_state=random_state)
        k = df_init['CLUSTER'].nunique()
        #################################################################

        #################################################################
        # Run Iterative Learning Algorithm

        df_new,training_error,testing_error = splitter(df_init,
                                          pfeatures,
                                          th,
                                          df_test,
                                          testing = True,
                                          classification=classification,
                                          it=n_iter,
                                          h=h,
                                          OutputFlag = 0,
                                          n=n,
                                          plot = plot)
        df_training_error = df_training_error.merge(training_error, \
                                                    how='outer', on=['Clusters'])
        df_testing_error = df_testing_error.merge(testing_error, \
                                                  how='outer', on=['Clusters'])

        m = predict_cluster(df_new, pfeatures)
        df_err, E_v = error_per_ID(df_test, df_new, m, pfeatures, relative=True, h=h)
        testing_errors.append(df_err)

    df_training_error.set_index('Clusters', inplace=True)
    df_testing_error.set_index('Clusters', inplace=True)
    df_training_error.dropna(inplace=True)
    df_testing_error.dropna(inplace=True)
    #print(df_training_error)
    #print(df_testing_error)
    cv_training_error = np.mean(df_training_error, axis=1)
    cv_testing_error = np.mean(df_testing_error, axis=1)
    #print(cv_training_error)
    #print(cv_testing_error)
    
    
    if plot:
        fig1, ax1 = plt.subplots()
        #its = np.arange(k+1,k+1+len(cv_training_error))
        ax1.plot(cv_training_error.index.values, cv_training_error, label= "CV Training Error")
        #ax1.plot(its, cv_testing_error, label = "CV Testing Error")
        ax1.plot(cv_testing_error.index.values, cv_testing_error, label= "CV Testing Error")
        #ax1.plot(its, training_acc, label = "Training Accuracy")
        #ax1.plot(its, testing_acc, label = "Testing Accuracy")
        if n>0:
            ax1.axvline(x=n,linestyle='--',color='r') #Plotting vertical line at #cluster =n
        ax1.set_ylim(0)
        ax1.set_xlabel('# of Clusters')
        ax1.set_ylabel('Mean CV Error or Accuracy %')
        ax1.set_title('Mean CV Error and Accuracy During Splitting')
        ax1.legend()
    
    #for t in testing_errors:
    #    print(t)
    
    return (cv_training_error,cv_testing_error)

    #return (list_training_error,list_testing_error,df_new, df_test)
#############################################################################
