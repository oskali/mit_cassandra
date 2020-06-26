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

from mdp_states_functions import createSamples, fit_CV, initializeClusters, \
        splitter
from mdp_testing import predict_cluster, get_MDP, predict_region_date, mape, \
        PredictionError
import os
#############################################################################

class MDP_model:
    def __init__(self,
                 days_avg=None,
                 horizon=5,
                 n_iter=40,
                 n_folds_cv=5,
                 clustering_distance_threshold=0.05,
                 splitting_threshold=0.,
                 classification_algorithm='DecisionTreeClassifier',
                 clustering_algorithm='Agglomerative',
                 n_clusters=None,
                 action_thresh=[],
                 date_colname='date',
                 region_colname='state',
                 features_list=[],
                 target_colname='cases',
                 random_state=42,
                 verbose=False):

        self.days_avg = days_avg  # number of days to average and compress datapoints
        self.horizon = horizon  # done
        self.n_iter = n_iter  # done
        self.n_folds_cv = n_folds_cv  # done
        self.clustering_distance_threshold = clustering_distance_threshold  # clustering diameter for Agglomerative clustering
        self.splitting_threshold = splitting_threshold  # done
        self.classification_algorithm = classification_algorithm  # done
        self.clustering_algorithm = clustering_algorithm  # clustering method from Agglomerative, KMeans, and Birch
        self.n_clusters = n_clusters  # number of clusters for KMeans
        self.action_thresh = action_thresh  # done
        self.date_colname = date_colname  # done
        self.region_colname = region_colname  # column name of the region, i.e. 'state'
        self.features_list = features_list  # list of the features that are considered to be trained
        self.target_colname = target_colname  # done
        self.random_state = random_state  # done
        self.verbose = verbose  # print out the intermediate steps

        # training attributes
        self.CV_error = None  # error at minimum point of CV
        self.classifier = None  # model for predicting cluster number from features # done
        self.P_df = None  # Transition function of the learnt MDP
        self.R_df = None  # Reward function of the learnt MDP
        self.optimal_cluster_size = None

        # data attributes
        self.df = None  # original dataframe from data
        self.df_trained = None  # dataframe after optimal training
        self.pfeatures = None  # number of features

    def fit(self,
            data, # csv file with data OR data frame
            ):

        # load data
        if type(data) == str:
            data = pd.read_csv(data)
        # creates samples from dataframe
        df, pfeatures = createSamples(data,
                                      self.target_colname,
                                      self.region_colname,
                                      self.date_colname,
                                      self.features_list,
                                      self.action_thresh,
                                      self.days_avg)

        self.df = df
        self.pfeatures = pfeatures

        # run cross validation on the data to find best clusters
        #
        # def fit_CV(df,
        #        pfeatures,
        #        th,
        #        clustering,
        #        clustering_distance_threshold,
        #        classification,
        #        n_iter,
        #        n_clusters,
        #        h=5,
        #        OutputFlag = 0,
        #        cv=5,
        #        n=-1,
        #        random_state=1234,
        #        plot=False):
        cv_training_error, cv_testing_error = fit_CV(df,
                                                     pfeatures=self.pfeatures,
                                                     th=self.splitting_threshold,
                                                     clustering=self.clustering_algorithm,
                                                     clustering_distance_threshold=self.clustering_distance_threshold,
                                                     classification=self.classification_algorithm,
                                                     n_iter=self.n_iter,
                                                     n_clusters=self.n_clusters,
                                                     h=self.horizon,
                                                     OutputFlag=self.verbose,
                                                     cv=self.n_folds_cv,
                                                     random_state=self.random_state,
                                                     )

        # find the best cluster
        try:
            k = cv_testing_error.idxmin()
            self.CV_error = cv_testing_error.loc[k]
        except:
            k = self.n_iter

        # update the optimal number of clusters
        self.optimal_cluster_size = k
        if self.verbose:
            print('minimum iterations:', k)

        # error corresponding to chosen model

        # actual training on all the data
        df_init = initializeClusters(self.df,
                                     clustering=self.clustering_algorithm,
                                     n_clusters=self.n_clusters,
                                     distance_threshold=self.clustering_distance_threshold,
                                     random_state=self.random_state)

        df_new,training_error,testing_error = splitter(df_init,
                                                       pfeatures=self.pfeatures,
                                                       th=self.splitting_threshold,
                                                       df_test=None,
                                                       testing=False,
                                                       classification=self.classification_algorithm,
                                                       it=k,
                                                       h=self.horizon,
                                                       OutputFlag=0)

        # storing trained dataset and predict_cluster function
        self.df_trained = df_new
        self.classifier = predict_cluster(self.df_trained, self.pfeatures)

        # store P_df and R_df values
        P_df,R_df = get_MDP(self.df_trained)
        self.P_df = P_df
        self.R_df = R_df

    # predict() takes a state name and a time horizon, and returns the predicted
    # number of cases after h steps from the most recent datapoint
    def predict_region_ndays(self,
                region,  # str: i.e. US state for prediction to be made
                n_days):  # int: time horizon (number of days) for prediction
        # preferably a multiple of days_avg (default 3)
        h = int(np.round(n_days/self.days_avg))
        delta = n_days - self.days_avg*h

        # get initial cases for the state at the latest datapoint
        target = self.df[self.df[self.region_colname] == region].iloc[-1, :][self.target_colname]
        date = self.df[self.df[self.region_colname] == region].iloc[-1, 1]

        if self.verbose:
            print('current date:', date,'| current %s:'%self.target_colname, target)

        # cluster the this last point
        s = self.df_trained[self.df_trained[self.region_colname]==region].iloc[-1, -2]
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
    def predict_allregions_ndays(self,
                    n_days): # time horizon for prediction, preferably a multiple of days_avg (default 3)
        df = self.df
        df = df[[self.region_colname,'TIME',self.target_colname]]
        df = df.groupby(self.region_colname).last()
        df.reset_index(inplace=True)
        df['TIME'] = df['TIME'] + timedelta(n_days)
        df[self.target_colname] = df[self.region_colname].apply(
            lambda region: int(self.predict_region_ndays(region, n_days)))
        return df

    # predict_class() takes a dictionary of states and time horizon and returns their predicted number of cases
    def predict(self,
                regions,  # list of states to predict the target
                dates,  # list of dates to predict the target
                ):

        # instantiate the prediction dataframe
        pred_df = pd.DataFrame(columns=[self.region_colname, 'TIME', self.target_colname])

        # get the last dates for each states
        df = self.df.copy()
        df_last = df[[self.region_colname, 'TIME', self.target_colname]].groupby(self.region_colname).last().reset_index().set_index(self.region_colname)
        df_first = df[[self.region_colname, 'TIME', self.target_colname]].groupby(self.region_colname).first().reset_index().set_index(self.region_colname)
        region_set = set(df[self.region_colname].values)

        for region in regions:
            try:
                assert region in region_set

            # the state doesn't appear not in the region set
            except AssertionError:
                if self.verbose:
                    print("The region '{}' is not in the trained region set".format(region))
                continue  # skip skip to the next region

            first_date = df_first.loc[region, "TIME"]
            last_date = df_last.loc[region, "TIME"]
            for date in dates:
                try:
                    pred = predict_region_date(self, (region, first_date, last_date), date, verbose=self.verbose)
                    pred_df = pred_df.append({self.region_colname: region, "TIME": date, self.target_colname: pred}, ignore_index=True)
                except PredictionError:
                    pass
        return pred_df


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
    target_colname = 'deaths'
    mdp_region_colname = 'state' # str, col name of region (e.g. 'state')
    mdp_date_colname = 'date' # str, col name of time (e.g. 'date')
    mdp_features_list = [] # list of strs: feature columns

    sgm = .1
    n_iter_mdp = 50
    n_iter_ci = 10
    ci_range = 0.75

    cols_to_keep = ['state',
                    'date',
                    target_colname,
                    'prevalence'] + [model + '_' + metric for model in models for metric in metrics]

    df_orig = pd.read_csv(os.path.join(path, file))
    print('Data Wrangling in Progress...')
    df = deepcopy(df_orig)
    df.columns = map(str.lower, df.columns)
    #df = df.query('cases >= @nmin')
    df= df[df[target_colname] >= nmin]
    df['date'] = df['date'].apply(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d'))
    df_orig['date'] = df_orig['date'].apply(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d'))
    df = df.sort_values(by=['state', 'date'])
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
    mdp = MDP_model(
        target_colname=target_colname,  # str: col name of target_colname (i.e. 'deaths')
        region_colname=mdp_region_colname,  # str, col name of region (i.e. 'state')
        date_colname=mdp_date_colname,  # str, col name of time (i.e. 'date')
        features_list=mdp_features_list,  # list of strs: feature columns
        horizon=5,
        n_iter=n_iter_mdp,
        days_avg=3,
        n_folds_cv=3,
        clustering_distance_threshold=0.1,
        verbose=False,
        random_state=1234)

    mdp_abort=False
    try:
        mdp.fit(df_train)
    except ValueError:
        print('ERROR: Feature columns have missing values! Please drop'
              'rows or fill in missing data.')
        print('MDP Model Aborted.')
        mdp_abort = True
        run_mdp = False

    # ####### test prediction methods ##########
    run_predict_all = True
    run_predict_class = True

    # test predict all :
    if run_predict_all:
        if not mdp_abort:
            mdp_output = pd.DataFrame()
            for i in range(pred_out):
                mdp_output = mdp_output.append(mdp.predict_allregions_ndays(n_days=i))
            print(mdp_output)

    # test predict class :
    if run_predict_class:
        if not mdp_abort:
            example_dict = (["Alabama", "Gabon", "Iowa", "Massachusetts"], ["2019-06-14", "2020-05-14", "2020-07-01"])
            mdp_output = mdp.predict(*example_dict)

            print(mdp_output)
    print('MDP Model (test) Complete.')
