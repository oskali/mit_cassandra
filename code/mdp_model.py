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
        self.region_col = None # str: i.e. 'state'
        self.target_col = None # str: i.e. 'cases'
        self.date_col = None # str: i.e. 'date'

    # fit() takes in parameters for prediction, and trains the model on the
    # optimal clustering for a given horizon h
    def fit(self,
            file, # csv file with data OR data frame
            target_col, # str: col name of target (i.e. 'deaths')
            region_col, # str, col name of region (i.e. 'state')
            date_col, # str, col name of time (i.e. 'date')
            features_cols, # list of str: i.e. (['mobility', 'testing'])
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
        df, pfeatures = createSamples(df, 
                                      target_col,
                                      region_col,
                                      date_col,
                                      features_cols,
                                      action_thresh, 
                                      d_avg)
        self.df = df
        self.pfeatures = pfeatures
        self.d_avg = d_avg
        self.target_col = target_col
        self.region_col = region_col
        self.date_col = date_col

        # run cross validation on the data to find best clusters
        cv_training_error,cv_testing_error=fit_CV(df,
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
        try:
            k = cv_testing_error.idxmin()
            self.CV_error = cv_testing_error.loc[k]
        except:
            k = n_iter
        self.opt_k = k
        if self.verbose:
            print('minimum iterations:', k)

        # error corresponding to chosen model
        

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
                                          it = k,
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
                region, # str: i.e. US state for prediction to be made
                n_days): # int: time horizon (number of days) for prediction
                        # preferably a multiple of d_avg (default 3)

        h = int(np.round(n_days/self.d_avg))
        delta = n_days - self.d_avg*h
        # get initial cases for the state at the latest datapoint
        target = self.df[self.df[self.region_col]== region].iloc[-1, 2]
        date = self.df[self.df[self.region_col]== region].iloc[-1, 1]
        
        if self.verbose:
            print('current date:', date,'| current %s:'%self.target_col, target)

        # cluster the this last point
        s = self.df_trained[self.df_trained[self.region_col]==region].iloc[-1, -2]
        
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
        pred = target*r*(np.exp(self.R_df.loc[s])**(delta/3))

        if self.verbose:
            print('Prediction for date:', date + timedelta(n_days),'| target:', pred)
        return pred


    # predict_all() takes a time horizon, and returns the predicted number of
    # cases after h steps from the most recent datapoint for all states
    def predict_all(self,
                    n_days): # time horizon for prediction, preferably a multiple of d_avg (default 3)
        h = int(np.round(n_days/self.d_avg))
        df = self.df
        df = df[[self.region_col,'TIME',self.target_col]]
        df = df.groupby(self.region_col).last()
        df.reset_index(inplace = True)
        df['TIME'] = df['TIME'] + timedelta(n_days)
        df[self.target_col] = df[self.region_col].apply(lambda st: int(self.predict(st,n_days)))
        return df


# model_testing() takes in n_days we want to predict on, all the model training
# parameters, creates the appropriate training data, and runs the fit and predict
# functions. Returns an instance of the trained model, and the mape error df
def model_testing(file, # csv file with data OR data frame
                  target_col, # str: col name of target (i.e. 'deaths')
            region_col, # str, col name of region (i.e. 'state')
            date_col, # str, col name of time (i.e. 'date')
            features_cols, # list of str: i.e. (['mobility', 'testing'])
            n_days, # int: n_days for prediction (cut the data here)
            h, # time horizon to get best prediction
            n_iter=40, # number of iterations
            d_avg=3, # int: number of days to average data over
            distance_threshold = 0.1, # clustering diameter for Agglomerative clustering
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
    df.loc[:, [date_col]]= pd.to_datetime(df[date_col])
    split_date = df[date_col].max() - timedelta(n_days)
    df_train = df.loc[df[date_col] <= split_date]

    m = MDP_model()
    m.fit(df_train, # csv file with data OR data frame
            target_col, # str: col name of target (i.e. 'deaths')
            region_col, # str, col name of region (i.e. 'state')
            date_col, # str, col name of time (i.e. 'date')
            features_cols, # list of str: i.e. (['mobility', 'testing'])
            h, # time horizon
            n_iter, # max # of clusters
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
    df_pred.set_index(region_col, inplace=True)
    date = df_pred['TIME'].max()
    df_true = df.loc[df[date_col]==date].sort_values(by=[region_col])
    df_true.set_index(region_col, inplace=True)
    error = mape(df_pred, df_true, m.target_col)
    
    return m, error
#############################################################################
