# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 16:21:17 2020

@author: janiceyang, omars
"""

#%% Libraries

import os
import pandas as pd
import numpy as np
from datetime import timedelta
from itertools import product
import operator


from mdp_states_functions import createSamples, fit_cv, fit_eval
from mdp_utils import initializeClusters, splitter
from mdp_testing import predict_cluster, get_MDP, predict_region_date, \
        MDPPredictionError, MDPTrainingError

#%% Model


class MDPModel:
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
                 action_thresh=([], 0),
                 date_colname='date',
                 region_colname='state',
                 features_list=[],
                 target_colname='cases',
                 random_state=42,
                 n_jobs=2,
                 verbose=0,
                 plot=True,
                 save=False,
                 savepath="",
                 keep_first=False,
                 region_exceptions=None):

        # Model dependent input attributes
        self.days_avg = days_avg  # number of days to average and compress datapoints
        self.horizon = horizon  # size of the validation set by region in terms of the number of forward observations
        self.n_iter = n_iter  # number of iterations in the training set
        self.n_folds_cv = n_folds_cv  # number of folds for the cross validation
        self.clustering_distance_threshold = clustering_distance_threshold  # clustering diameter for Agglomerative clustering
        self.splitting_threshold = splitting_threshold  # threshold to which one cluster is selected for a split
        self.classification_algorithm = classification_algorithm  # classification algorithm used for learning the state
        self.clustering_algorithm = clustering_algorithm  # clustering method from Agglomerative, KMeans, and Birch
        self.n_clusters = n_clusters  # number of clusters for KMeans
        self.action_thresh = action_thresh  # ([list of action threshold], default action)  # ex : ([-0.5, 0.5], 1) --> ACTION=[-1, 0, 1]
        self.date_colname = date_colname  # column name of the date, i.e. 'date'
        self.region_colname = region_colname  # column name of the region, i.e. 'state'
        self.features_list = features_list  # list of the features that are considered to be trained
        #                                     PS: feature[0] --> Action feature
        self.target_colname = target_colname  # column name of the target, i.e. 'cases'
        self.region_exceptions = region_exceptions  # exception region to be removed

        # Experiment attributes
        self.random_state = random_state  # random seed using the random generators
        self.verbose = verbose  # define the precision OutputFlag
        # 0 : no print, 1: print global steps of the algorithm, 2: print up to the warnings and secondary information
        self.plot = plot  # plot out the training error curves (without interrupting the run)
        self.save = save  # save the results (plots)
        self.savepath = savepath  # path to the save folder
        self.keep_first = keep_first  # bool for keeping the initial clusters (experiment purpose)
        self.n_jobs = n_jobs  # number of jobs for multiprocessing

        # Training output attributes
        self.CV_error = None  # error at minimum point of CV
        self.classifier = None  # model for predicting cluster number from features # done
        self.P_df = None  # Transition function of the learnt MDP
        self.R_df = None  # Reward function of the learnt MDP
        self.optimal_cluster_size = None  # optimal number of clusters of the MDP

        # Training data attributes
        self.df_trained = None  # dataframe after optimal training
        self.df_trained_first = None  # dataframe containing the initial clusters (updated if keep_first = True)
        self.pfeatures = None  # number of features

    # create an independent copy of the MDP object
    def __copy__(self):

        other = MDPModel(
            days_avg=self.days_avg,
            horizon=self.horizon,
            n_iter=self.n_iter,
            n_folds_cv=self.n_folds_cv,
            clustering_distance_threshold=self.clustering_distance_threshold,
            splitting_threshold=self.splitting_threshold,
            classification_algorithm=self.classification_algorithm,
            clustering_algorithm=self.clustering_algorithm,
            n_clusters=self.n_clusters,
            action_thresh=self.action_thresh,
            date_colname=self.date_colname,
            region_colname=self.region_colname,
            features_list=self.features_list,
            target_colname=self.target_colname,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            plot=self.plot,
            save=self.save,
            savepath=self.savepath,
            keep_first=self.keep_first)

        # trained attributes
        other.CV_error = self.CV_error
        other.classifier = self.classifier
        other.P_df = self.P_df
        other.R_df = self.R_df
        other.optimal_cluster_size = self.optimal_cluster_size

        try:
            other.df_trained = self.df_trained.copy()
        except AttributeError:
            other.df_trained = None

        try:
            other.df_trained_first = self.df_trained_first.copy()
        except AttributeError:
            other.df_trained_first = None

        other.pfeatures = self.pfeatures

        return other

    # provide a representation of the (cleaned) MDP for printing
    def __repr__(self):
        return "MDPModel(target={}, " \
               "horizon={}, " \
               "days_avg={}," \
               "distance_threshold={}pct, " \
               "n_iter={}, " \
               "classification_algorithm={}, " \
               "features_list={}," \
               "action_thresh={})".format(self.target_colname,
                                          self.horizon,
                                          self.days_avg,
                                          int(self.clustering_distance_threshold * 100),
                                          self.n_iter,
                                          self.classification_algorithm,
                                          self.features_list,
                                          self.action_thresh)

    # provide a condensed representation of the MDP as a string
    def __str__(self):
        return "mdp__target_{}__h{}__davg{}__cdt_{}pct__n_iter{}__ClAlg_{}".format(self.target_colname,
                                                                                   self.horizon,
                                                                                   self.days_avg,
                                                                                   int(self.clustering_distance_threshold * 100),
                                                                                   self.n_iter,
                                                                                   self.classification_algorithm)

    # Fitting method to learn from the data
    # data should be a pandas.DataFrame object containing the provided columns
    # mode : either 'ID', 'TIME_CV', or 'ALL' defines how the cross validation is proceed
    # - 'ID' : the data are split onto folds  according to the ID columns,
    #          The training set compounds every folds except one
    #          The validation set compounds the excluded fold onto which the error in calculated over the horizon of last observations
    # - 'TIME_CV' : the data are split onto folds  according to the ID columns,
    #          The training set compounds every folds except one + data of the excluded fold before the horizon of last observations
    #          The validation set compounds the excluded fold onto which the error in calculated during the horizon of last observations
    # - 'ALL' : there is no splitting
    #          The training set compounds every ID before the horizon of last observations
    #          The validation set compounds every ID during the horizon of last observations
    def fit(self,
            data,  # csv file with data OR data frame
            mode="TIME_CV"):

        # create paths for training results if not existing
        if self.save:
            try:
                assert os.path.exists(os.path.join(self.savepath, mode, str(self)))
            except AssertionError:
                os.makedirs(os.path.join(self.savepath, mode, str(self)))

        # assert if the training mode is available
        try:
            assert mode in {"ALL", "ID", "TIME_CV"}
            if self.verbose >= 2:
                print("The split mode is by '{}'".format(mode))
        except AssertionError:
            if self.verbose >= 1:
                print("TrainingError: the mode must be a string; either 'ALL', 'TIME_CV' or 'ID'.")
            raise MDPTrainingError

        # load data
        if type(data) == str:
            data = pd.read_csv(data)

        # creates samples from DataFrame
        df, pfeatures = createSamples(data.copy(),
                                      target_colname=self.target_colname,
                                      region_colname=self.region_colname,
                                      date_colname=self.date_colname,
                                      features_list=self.features_list,
                                      action_thresh_base=self.action_thresh,
                                      days_avg=self.days_avg,
                                      region_exceptions=self.region_exceptions)

        self.pfeatures = pfeatures

        # run cross validation on the data to find best clusters
        cv_training_error, cv_testing_error = fit_cv(df.copy(),
                                                     pfeatures=self.pfeatures,
                                                     splitting_threshold=self.splitting_threshold,
                                                     clustering=self.clustering_algorithm,
                                                     clustering_distance_threshold=self.clustering_distance_threshold,
                                                     classification=self.classification_algorithm,
                                                     n_iter=self.n_iter,
                                                     n_clusters=self.n_clusters,
                                                     horizon=self.horizon,
                                                     OutputFlag=self.verbose,
                                                     cv=self.n_folds_cv,
                                                     random_state=self.random_state,
                                                     n_jobs=self.n_jobs,
                                                     mode=mode,
                                                     plot=self.plot,
                                                     save=self.save,
                                                     savepath=os.path.join(self.savepath, mode, str(self))
                                                     )

        # find the best cluster
        try:
            k = cv_testing_error.idxmin()
            self.CV_error = cv_testing_error.loc[k]
        except:
            k = self.n_iter

        # update the optimal number of clusters
        self.optimal_cluster_size = k
        if self.verbose >= 1:
            print('minimum iterations:', k)

        # error corresponding to chosen model

        # actual training on all the data
        df_trained = initializeClusters(df.copy(),
                                        clustering=self.clustering_algorithm,
                                        n_clusters=self.n_clusters,
                                        distance_threshold=self.clustering_distance_threshold,
                                        random_state=self.random_state)

        df_trained, training_error, testing_error = splitter(df_trained.copy(),
                                                             pfeatures=self.pfeatures,
                                                             th=self.splitting_threshold,
                                                             df_test=None,
                                                             testing=False,
                                                             classification=self.classification_algorithm,
                                                             it=k,
                                                             h=self.horizon,
                                                             OutputFlag=self.verbose,
                                                             plot=self.plot,
                                                             save=self.save,
                                                             savepath=os.path.join(self.savepath, mode, str(self),  "plot_final.PNG")
                                                             )

        # storing trained dataset and predict_cluster function
        self.df_trained = df_trained
        self.classifier = predict_cluster(self.df_trained, self.pfeatures)

        # store P_df and R_df values
        P_df,R_df = get_MDP(self.df_trained)
        self.P_df = P_df
        self.R_df = R_df

        # store the initial clusters
        if self.keep_first:
            if self.verbose >= 2:
                print("Saving the initial clusters per region...")
            self.df_trained_first = self.df_trained.groupby(self.region_colname).first().copy()
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

        if self.verbose >= 2:
            print('current date:', date, '| current %s:'%self.target_colname, target)

        # cluster the this last point
        s = self.df_trained.loc[region, "CLUSTER"]
        if self.verbose >= 2:
            print('predicted initial cluster', s)

        r = 1
        clusters_seq = [s]
        # run for horizon h, multiply out the ratios
        for i in range(h):
            r = r*np.exp(self.R_df.loc[s])
            s = self.P_df.loc[s, 0].values[0]
            clusters_seq.append(s)

        if self.verbose >= 2:
            print('Sequence of clusters:', clusters_seq)
        pred = target*r*(np.exp(self.R_df.loc[s])**(delta/3.))

        if self.verbose >= 2:
            print('Prediction for date:', date + timedelta(n_days), '| target:', pred)
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

        # get the last dates for each states
        # df_last = df[[self.region_colname, 'TIME', self.target_colname]].groupby(self.region_colname).last().reset_index().set_index(self.region_colname)
        # df_first = df[[self.region_colname, 'TIME', self.target_colname]].groupby(self.region_colname).first().reset_index().set_index(self.region_colname)
        region_set = set(self.df_trained.index)

        for region in regions:
            try:
                assert region in region_set

            # the state doesn't appear not in the region set
            except AssertionError:
                if self.verbose >=1:
                    print("The region '{}' is not in the trained region set".format(region))
                continue  # skip skip to the next region

            last_date = self.df_trained.loc[region, "TIME"]
            for date in dates:
                try:
                    pred = predict_region_date(self, (region, last_date), date, verbose=self.verbose)
                    pred_df = pred_df.append({self.region_colname: region, "TIME": date, self.target_colname: pred}, ignore_index=True)
                except MDPPredictionError:
                    pass
        pred_df.rename(columns={'TIME': self.date_colname}, inplace=True)
        pred_dic = {state: pred_df[pred_df[self.region_colname] == state].set_index([self.date_colname])[self.target_colname] for state in pred_df[self.region_colname].unique()}

        return pred_dic
