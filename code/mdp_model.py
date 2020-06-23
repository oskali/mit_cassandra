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
from copy import deepcopy
import datetime

from mdp_states_functions import *
from mdp_testing import *
import os
#############################################################################

class PredictionError(Exception):
    pass
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
        df.reset_index(inplace=True)
        df['TIME'] = df['TIME'] + timedelta(n_days)
        df[self.target_col] = df[self.region_col].apply(lambda st: int(self.predict(st,n_days)))
        return df


    # predict_class_date() takes a given state and a date and returns the predicted target
    def predict_class_date(self,
                    region_first_last_dates, # tuple (region, first_date, last_date), e.g (Alabama, Timestamp('2020-03-24 00:00:00'), Timestamp('2020-06-22 00:00:00'))
                           date, # target date for prediciton, e.g. (Timestamp('2020-05-24 00:00:00'))
                           verbose=0):

        region, first_date, last_date = region_first_last_dates
        try:
            date = datetime.strptime(date,'%Y-%m-%d')
        except TypeError:
            pass

        # Case 1 : the input date occurs before the first available date for a given region
        try :
            assert date >= first_date
        except AssertionError:
            if verbose:
                print("Prediction Error type I ('{}', '{}'): the input occurs before the first available ('{}') date in the training set".format(region,
                                                                                                                                          str(date),
                                                                                                                                          str(first_date)
                                                                                                                                           ))
            raise PredictionError  # test

        # Case 2 : the input date occurs within the range of input dates for a given region
        if date <= last_date:

            # compute the closest training date
            n_days = (last_date - date).days
            lag = ((- n_days) % self.d_avg)
            pos = n_days // self.d_avg + (lag > 0)
            clst_past_date = last_date - timedelta(pos * self.d_avg)

            # get the observation :
            try:
                clst_past_pred = self.df_trained[(self.df_trained[self.region_col] == region)
                                                 & (self.df_trained.TIME == clst_past_date)]
                assert (not clst_past_pred.empty)  # verify that the closest date is actually in the training date

                s = clst_past_pred["CLUSTER"]
                target = clst_past_pred[self.target_col].values[0] * (np.exp(self.R_df.loc[s].values[0])**(float(lag)/3))
                return target

            except AssertionError:
                if verbose:
                    print("Prediction Error type II ('{}', '{}'): The computed in-sample closest date '{}' is not in the training set".format(region,
                                                                                                                                          str(date),
                                                                                                                                          str(clst_past_date)
                                                                                                                                           ))
                raise PredictionError

        # Case 3 : the date has not been observed yet :
        n_days = (date-last_date).days
        return self.predict(region, n_days)

    # predict_class() takes a dictionary of states and time horizon and returns their predicted number of cases
    def predict_class(self,
                      state_dict, # dictionary containing states and corresponding dates to predict the target
                      verbose=0):
        assert isinstance(state_dict, dict), " the 'state_dict' must be a dictionary"

        # instantiate the prediction dataframe
        pred_df = pd.DataFrame(columns=[self.region_col,'TIME',self.target_col])

        # get the last dates for each states
        df = self.df.copy()
        df_last = df[[self.region_col, 'TIME', self.target_col]].groupby(self.region_col).last().reset_index().set_index(self.region_col)
        df_first = df[[self.region_col, 'TIME', self.target_col]].groupby(self.region_col).first().reset_index().set_index(self.region_col)
        region_set = set(df[self.region_col].values)

        for region, dates in state_dict.items():
            try:
                assert region in region_set

            # the state doesn't appear not in the region set
            except AssertionError:
                if verbose:
                    print("The state '{}' is not in the trained region set".format(region))
                continue  # skip skip to the next region

            first_date = df_first.loc[region, "TIME"]
            last_date = df_last.loc[region, "TIME"]
            for date in dates:
                try:
                    pred = self.predict_class_date((region, first_date, last_date), date, verbose=verbose)
                    pred_df = pred_df.append({self.region_col: region, "TIME": date, self.target_col: pred}, ignore_index=True)
                except PredictionError:
                    pass
        return pred_df


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


if __name__ == "__main__":
    from utils import models, metrics
    import warnings
    warnings.filterwarnings("ignore")  # to avoid Python deprecated version warnings

    # path = 'C:/Users/omars/Desktop/covid19_georgia/large_data/input/'
    path = os.path.join(os.path.dirname(os.getcwd()), 'data', 'input')
    file = '06_15_2020_states_combined.csv'
    training_cutoff = '2020-05-25'
    nmin = 20
    deterministic = True
    if deterministic:
        deterministic_label = ''
    else:
        deterministic_label = 'markov_'
    target = 'deaths'
    mdp_region_col = 'state' # str, col name of region (e.g. 'state')
    mdp_date_col = 'date' # str, col name of time (e.g. 'date')
    mdp_features_cols = [] # list of strs: feature columns

    sgm = .1
    n_iter_mdp = 50
    n_iter_ci = 10
    ci_range = 0.75

    cols_to_keep = ['state',
                    'date',
                    target,
                    'prevalence'] + [model + '_' + metric for model in models for metric in metrics]

    df_orig = pd.read_csv(os.path.join(path, file))
    print('Data Wrangling in Progress...')
    df = deepcopy(df_orig)
    df.columns = map(str.lower, df.columns)
    #df = df.query('cases >= @nmin')
    df= df[df[target] >= nmin]
    df['date'] = df['date'].apply(lambda x: datetime.strptime(x,'%Y-%m-%d'))
    df_orig['date'] = df_orig['date'].apply(lambda x: datetime.strptime(x,'%Y-%m-%d'))
    df = df.sort_values(by = ['state','date'])
    states = sorted(list(set(df['state'])))
    pop_df = df.loc[:, ['state', 'population']]
    pop_dic = {pop_df .iloc[i, 0] : pop_df .iloc[i, 1] for i in range(pop_df .shape[0])}
    features = list(df.columns[5:35])

    df = df.loc[:, df.columns[df.isnull().sum() * 100 / len(df) < 20]]
    features = list(set(features).intersection(set(df.columns)))

    df_train = df[df['date'] <= training_cutoff]
    df_test = df[df['date'] > training_cutoff]
    pred_out = len(set(df_test.date))
    day_0 = str(df_test.date.min())[:10]
    print('Data Wrangling Complete.')

    # ####### test fitting methods ##########
    print('MDP Model Training in Progress...')
    # df_train = df_orig[df_orig['date'] <= training_cutoff].drop(columns='people_tested').dropna(axis=0)
    df_train = df_orig[df_orig['date'] <= training_cutoff]
    mdp = MDP_model()
    mdp_abort=False
    try:
        mdp.fit(df_train,
                target_col = target, # str: col name of target (i.e. 'deaths')
                region_col = mdp_region_col, # str, col name of region (i.e. 'state')
                date_col = mdp_date_col, # str, col name of time (i.e. 'date')
                features_cols = mdp_features_cols, # list of strs: feature columns
                h=5,
                n_iter=n_iter_mdp,
                d_avg=3,
                distance_threshold = 0.1)
    except ValueError:
        print('ERROR: Feature columns have missing values! Please drop' \
              ' rows or fill in missing data.')
        print('MDP Model Aborted.')
        mdp_abort=True
        run_mdp = False

    # ####### test prediction methods ##########
    run_predict_all = False
    run_predict_class = True

    # test predict all :
    if run_predict_all:
        if not mdp_abort:
            mdp_output = pd.DataFrame()
            for i in range(pred_out):
                mdp_output = mdp_output.append(mdp.predict_all(n_days=i))

    # test predict class :
    if run_predict_class:
        example_dict = {"Alabama": ["2019-06-14", "2020-05-14", "2020-07-01"]}
        mdp_output = mdp.predict_class(example_dict, verbose=1)

        print(mdp_output)
    print('MDP Model (test) Complete.')
