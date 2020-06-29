# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 16:21:17 2020

@author: janiceyang, omars
"""

#%% Libraries

import pandas as pd
import numpy as np
from datetime import timedelta
from mdp_states_functions import createSamples, fit_CV, initializeClusters, \
        splitter
from mdp_testing import predict_cluster, get_MDP, predict_region_date, \
        PredictionError

#%% Model

class MDPModel:
    def __init__(self,
                 days_avg=3,
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

        # self.df = df
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
        df_init = initializeClusters(df,
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

        # store only the last states for prediction
        self.df_trained = self.df_trained.groupby(self.region_colname).last()

    # predict() takes a state name and a time horizon, and returns the predicted
    # number of cases after h steps from the most recent datapoint
    def predict_region_ndays(self,
                region,  # str: i.e. US state for prediction to be made
                n_days):  # int: time horizon (number of days) for prediction
        # preferably a multiple of days_avg (default 3)
        h = int(np.round(n_days/self.days_avg))
        delta = n_days - self.days_avg*h

        # get initial cases for the state at the latest datapoint
        target = self.df_trained.loc[region, self.target_colname]
        date = self.df_trained.loc[region, "TIME"]

        if self.verbose:
            print('current date:', date,'| current %s:'%self.target_colname, target)

        # cluster the this last point
        s = self.df_trained.loc[region, "CLUSTER"]
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
        df = self.df_trained.copy()
        df = df[['TIME', self.target_colname]]
        df['TIME'] = df['TIME'] + timedelta(n_days)
        df[self.target_colname] = df.index.map(
            lambda region: int(self.predict_region_ndays(region, n_days)))
        return df

    # predict_class() takes a dictionary of states and time horizon and returns their predicted number of cases
    def predict(self,
                regions,  # list of states to predict the target
                dates,  # list of dates to predict the target
                ):

        # instantiate the prediction dataframe
        pred_df = pd.DataFrame(columns=[self.region_colname, 'TIME', self.target_colname])

        region_set = set(self.df_trained.index)

        for region in regions:
            try:
                assert region in region_set

            # the state doesn't appear not in the region set
            except AssertionError:
                if self.verbose:
                    print("The region '{}' is not in the trained region set".format(region))
                continue  # skip skip to the next region

            last_date = self.df_trained.loc[region, "TIME"]
            for date in dates:
                try:
                    pred = predict_region_date(self, (region, last_date), date, verbose=self.verbose)
                    pred_df = pred_df.append({self.region_colname: region, "TIME": date, self.target_colname: pred}, ignore_index=True)
                except PredictionError:
                    pass

        pred_df.rename(columns={'TIME': self.date_colname}, inplace = True)
        pred_dic = {state: pred_df[pred_df[self.region_colname] == state].set_index([self.date_colname])[self.target_colname] for state in pred_df[self.region_colname].unique()}

        return pred_dic
