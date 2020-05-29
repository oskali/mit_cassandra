#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 19:47:03 2020

Model Class that runs the Iterative Clustering algorithm on COVID States

data.

@author: janiceyang
"""
#############################################################################
# Load Libraries
import pandas as pd
import numpy as np
from datetime import timedelta

from mdp_states_functions import *
from mdp_testing import *
#############################################################################

class MDP_model:
    def __init__(self):
        self.df = None # original dataframe from data
        self.pfeatures = None # number of features
        self.d_avg = None # number of days to average and compress datapoints
        self.CV_error = None # error at minimum point of CV
        self.df_trained = None # dataframe after optimal training
        self.m = None # model for predicting cluster number from features
        self.P_df = None #Transition function of the learnt MDP
        self.R_df = None #Reward function of the learnt MDP
        self.verbose = False

    # fit() takes in parameters for prediction, and trains the model on the
    # optimal clustering for a given horizon h
    def fit(self,
            file, # csv file with data OR data frame
            h=5, # time horizon = n_days/d_avg
            n_iter=40, # number of iterations
            d_avg=3, # int: number of days to average data over
            distance_threshold = 0.05, # clustering diameter for Agglomerative clustering
            action_thresh = [], # list of cutoffs for each action bin
            cv=5, # number for cross validation
            th=0, # splitting threshold
            classification = 'DecisionTreeClassifier', # classification method
            clustering='Agglomerative',# clustering method from Agglomerative, KMeans, and Birch
            n_clusters = None, # number of clusters for KMeans
            random_state = 0):

        # load data
        if type(file) == str:
            df = pd.read_csv(file)
        else:
            df = file
        # creates samples from dataframe
        df, pfeatures = createSamples(df, action_thresh, d_avg)
        self.df = df
        self.pfeatures = pfeatures
        self.d_avg = d_avg

        # run cross validation on the data to find best clusters
        list_training_error,list_testing_error,df_new,df_test =fit_CV(df,
                                              pfeatures,
                                              th,
                                              clustering,
                                              distance_threshold,
                                              classification,
                                              n_iter,
                                              n_clusters,
                                              random_state,
                                              h = h,
                                              OutputFlag = 0,
                                              cv=cv)

        # find the best cluster
        cv_testing_error = np.mean(np.array(list_testing_error),axis=0)
        it = max(1 ,np.argmin(cv_testing_error))
        if self.verbose:
            print('minimum iterations:', it) # this is iterations, but should be cluster

        # error corresponding to chosen model
        self.CV_error = cv_testing_error[it]

        # actual training on all the data
        df_init = initializeClusters(self.df,
                                clustering=clustering,
                                n_clusters=n_clusters,
                                distance_threshold = distance_threshold,
                                random_state=random_state)

        df_new,training_error,testing_error = splitter(df_init,
                                          pfeatures=self.pfeatures,
                                          th=th,
                                          df_test = None,
                                          testing = False,
                                          classification=classification,
                                          it = it,
                                          h=h,
                                          OutputFlag = 0)

        # storing trained dataset and predict_cluster function
        self.df_trained = df_new
        self.m = predict_cluster(self.df_trained, self.pfeatures)

        # store P_df and R_df values
        P_df,R_df = get_MDP(self.df_trained)
        self.P_df = P_df
        self.R_df = R_df


    # predict() takes a state name and a time horizon, and returns the predicted
    # number of cases after h steps from the most recent datapoint
    def predict(self,
                state, # str: US state for prediction to be made
                n_days): # int: time horizon (number of days) for prediction
                        # preferably a multiple of d_avg (default 3)

        h = int(np.round(n_days/self.d_avg))
        delta = n_days - self.d_avg*h
        # get initial cases for the state at the latest datapoint
        cases = self.df[self.df['state']== state].iloc[-1, 2]
        date = self.df[self.df['state']== state].iloc[-1, 1]

        if self.verbose:
            print('current date:', date,'| current cases:', cases)

        # cluster the this last point
        s = self.df_trained[self.df_trained['state']==state].iloc[-1, -2]
        #s = int(self.m.predict([self.df[self.df['state']== state].iloc[-1, 2:2+self.pfeatures]]))
        if self.verbose:
            print('predicted initial cluster', s)

        r = 1
        clusters_seq =[s]
        # run for horizon h, multiply out the ratios
        for i in range(h):
            r = r*np.exp(self.R_df.loc[s])
            s = self.P_df.loc[s,0].values[0]
            clusters_seq.append(s)

        if self.verbose:
            print('Sequence of clusters:', clusters_seq)
        pred = cases*r*(np.exp(self.R_df.loc[s])**(delta/3))

        if self.verbose:
            print('Prediction for date:', date + timedelta(n_days),'| cases:', pred)
        return pred


    # predict_all() takes a time horizon, and returns the predicted number of
    # cases after h steps from the most recent datapoint for all states
    def predict_all(self,
                    n_days): # time horizon for prediction, preferably a multiple of d_avg (default 3)
        h = int(np.round(n_days/self.d_avg))
        df = self.df
        df = df[['state','TIME','cases']]
        df = df.groupby('state').last()
        df.reset_index(inplace = True)
        df['TIME'] = df['TIME'] + timedelta(n_days)
        df['cases'] = df['state'].apply(lambda st: int(self.predict(st,n_days)))
        return df


# model_testing() takes in n_days we want to predict on, all the model training
# parameters, creates the appropriate training data, and runs the fit and predict
# functions. Returns an instance of the trained model, and the mape error df
def model_testing(file, # csv file with data OR data frame
            n_days, # int: n_days for prediction (cut the data here)
            h, # time horizon to get best prediction
            n_iter=40, # number of iterations
            d_avg=3, # int: number of days to average data over
            distance_threshold = 0.06, # clustering diameter for Agglomerative clustering
            action_thresh = [], # list of cutoffs for each action bin
            cv=5, # number for cross validation
            th=0, # splitting threshold
            classification = 'DecisionTreeClassifier', # classification method
            clustering='Agglomerative',# clustering method from Agglomerative, KMeans, and Birch
            n_clusters = None, # number of clusters for KMeans
            random_state = 0):

    # load data
    if type(file) == str:
        df = pd.read_csv(file)
    else:
        df = file

    # take out dates for prediction
    df.loc[:, ['date']]= pd.to_datetime(df['date'])
    split_date = df['date'].max() - timedelta(n_days)
    df_train = df.loc[df['date'] <= split_date]

    m = MDP_model()
    m.fit(df_train, # csv file with data OR data frame
            h, # time horizon
            n_iter, # number of iterations
            d_avg, # int: number of days to average data over
            distance_threshold, # clustering diameter for Agglomerative clustering
            action_thresh, # list of cutoffs for each action bin
            cv, # number for cross validation
            th, # splitting threshold
            classification, # classification method
            clustering,# clustering method from Agglomerative, KMeans, and Birch
            n_clusters, # number of clusters for KMeans
            random_state)

    # create df_pred and df_true with the appropriate dates and sorted by state
    df_pred = m.predict_all(n_days)
    df_pred.set_index('state', inplace=True)
    date = df_pred['TIME'].max()
    df_true = df.loc[df['date']==date].sort_values(by=['state'])
    df_true.set_index('state', inplace=True)
    error = mape(df_pred, df_true)
    return m, error
#############################################################################
