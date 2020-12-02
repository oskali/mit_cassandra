# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 16:21:17 2020

@author: janiceyang, omars, davidnzendong
"""

#%% Libraries

import os
import pandas as pd
import numpy as np
from datetime import timedelta
from itertools import product
import operator

import multiprocessing as mp
from functools import partial

from codes.mdp_states_functions import createSamples, createSamplesRelative, fit_cv, fit_eval
from codes.mdp_utils import MDP_Splitter, splitter, MDPPredictorError, process_prediction_region
from codes.mdp_testing import predict_region_date, \
        MDPPredictionError, MDPTrainingError, compute_exp_mean_alpha, compute_exp_sigmoid_alpha

#%% Model

class MDPModel:
    def __init__(self,
                 days_avg=None,
                 horizon=5,
                 test_horizon=5,
                 n_iter=40,
                 n_folds_cv=5,
                 clustering_distance_threshold=0.05,
                 splitting_threshold=0.,
                 classification_algorithm='DecisionTreeClassifier',
                 clustering_algorithm='Agglomerative',
                 error_computing="horizon",
                 error_function_name="relative",
                 reward_name="RISK",
                 alpha=1e-5,
                 n_clusters=None,
                 action_thresh=([], 0),
                 date_colname='date',
                 region_colname='state',
                 features_list=[],
                 target_colname='cases',
                 random_state=42,
                 n_jobs=2,
                 verbose=0,
                 randomized=False,
                 randomized_split_pct=0.6,
                 nfeatures=20,
                 plot=False,
                 save=False,
                 savepath="",
                 keep_first=False,
                 region_exceptions=None):

        # Model dependent input attributes
        self.days_avg = days_avg  # number of days to average and compress datapoints
        self.horizon = horizon  # size of the train set by region in terms of the number of forward observations
        self.test_horizon = test_horizon  # size of the test set by region in terms of the number of forward observations
        self.n_iter = n_iter  # number of iterations in the training set
        self.n_folds_cv = n_folds_cv  # number of folds for the cross validation
        self.randomized = randomized  # boolean, if True, ID and features are randomly selected
        self.randomized_split_pct = randomized_split_pct  #
        self.clustering_distance_threshold = clustering_distance_threshold  # clustering diameter for Agglomerative clustering
        self.splitting_threshold = splitting_threshold  # threshold to which one cluster is selected for a split
        self.classification_algorithm = classification_algorithm  # classification algorithm used for learning the state
        self.clustering_algorithm = clustering_algorithm  # clustering method from Agglomerative, KMeans, and Birch
        self.error_computing = error_computing  # mode of computing the error, in {"horizon", "exponential", "uniform"}
        #                                       # horizon : compute error on the h last dates,
        #                                       # exponential : compute an exponential waited average other the whole training set
        #                                       # uniform : compute the error uniformly over the training period
        self.error_function_name = error_function_name  # [DEV : TO COMMENT]
        self.alpha = alpha  # if computing error is exponential, it define the rate of decay
        self.n_clusters = n_clusters  # number of clusters for KMeans
        self.action_thresh = action_thresh  # ([list of action threshold], default action)  # ex : ([-0.5, 0.5], 1) --> ACTION=[-1, 0, 1]
        self.date_colname = date_colname  # column name of the date, i.e. 'date'
        self.region_colname = region_colname  # column name of the region, i.e. 'state'
        self.features_list = features_list  # list of the features that are considered to be trained
        #                                     PS: feature[0] --> Action feature
        self.target_colname = target_colname  # column name of the target, i.e. 'cases'
        self.region_exceptions = region_exceptions  # exception region to be removed
        self.reward_name = reward_name  # the reward we estimate through the MDP

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
        self.features = None
        self.pfeatures = None  # number of features
        self.nfeatures = nfeatures
        self.actions = None
        self.splitter_dict = {}

        # Calibration after the training
        self.calibrated = False
        self.calibration_dict = {}

    # create an independent copy of the MDP object
    def __copy__(self):

        other = MDPModel(
            days_avg=self.days_avg,
            horizon=self.horizon,
            test_horizon=self.test_horizon,
            error_computing=self.error_computing,
            error_function_name=self.error_function_name,  # [DEV : TO COMMENT]
            alpha=self.alpha,
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
            region_exceptions=self.region_exceptions,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            randomized=self.randomized,
            randomized_split_pct=self.randomized_split_pct,
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
        other.nfeatures = self.nfeatures
        other.actions = self.actions

        return other

    # provide a representation of the (cleaned) MDP for printing
    def __repr__(self):
        return "MDPModel(risk={},target={}, " \
               "horizon={}, " \
               "days_avg={}," \
               "distance_threshold={}pct, " \
               "n_iter={}, " \
               "error_c={}, " \
               "classification_algorithm={}, " \
               "features_list={}," \
               "action_thresh={})".format(self.reward_name,
                                          self.target_colname,
                                          self.horizon,
                                          self.days_avg,
                                          int(self.clustering_distance_threshold * 100),
                                          self.n_iter,
                                          self.error_computing,
                                          self.classification_algorithm,
                                          self.features_list,
                                          self.action_thresh)

    # provide a condensed representation of the MDP as a string
    def __str__(self):
        return "mdp{}__target_{}__h{}__davg{}__cdt_{}pct__n_iter{}__ClAlg_{}__err{}_cv{}_nbfs{}".format(self.reward_name[0],
                                                                                                      self.target_colname,
                                                                                                      self.horizon,
                                                                                                      self.days_avg,
                                                                                                      int(self.clustering_distance_threshold * 100),
                                                                                                      self.n_iter,
                                                                                          self.classification_algorithm[:5],
                                                                                          self.error_computing[:5],
                                                                                                self.n_folds_cv, len(self.features_list))

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
            except:
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
        df, pfeatures, actions = createSamplesRelative(data.copy(),
                                                       target_colname=self.target_colname,
                                                       region_colname=self.region_colname,
                                                       date_colname=self.date_colname,
                                                       features_list=self.features_list,
                                                       action_thresh_base=self.action_thresh,
                                                       days_avg=self.days_avg,
                                                       region_exceptions=self.region_exceptions)

        self.pfeatures = pfeatures
        self.features = df.columns[2: 2+pfeatures].tolist()
        self.actions = actions

        # run cross validation on the data to find best clusters
        cv_training_error, cv_testing_error, trained_splitter_list = fit_cv(df.copy(),
                                                                            pfeatures=self.pfeatures,
                                                                            nfeatures=self.nfeatures,
                                                                            reward_name=self.reward_name,
                                                                            splitting_threshold=self.splitting_threshold,
                                                                            clustering=self.clustering_algorithm,
                                                                            clustering_distance_threshold=self.clustering_distance_threshold,
                                                                            classification=self.classification_algorithm,
                                                                            n_iter=self.n_iter,
                                                                            days_avg=self.days_avg,
                                                                            n_clusters=self.n_clusters,
                                                                            actions=self.actions,
                                                                            horizon=self.horizon,
                                                                            test_horizon=self.test_horizon,
                                                                            error_computing=self.error_computing,
                                                                            error_function_name=self.error_function_name,
                                                                            alpha=self.alpha,
                                                                            OutputFlag=self.verbose,
                                                                            cv=self.n_folds_cv,
                                                                            randomized=self.randomized,
                                                                            randomized_split_pct=self.randomized_split_pct,
                                                                            random_state=self.random_state,
                                                                            n_jobs=self.n_jobs,
                                                                            mode=mode,
                                                                            plot=self.plot,
                                                                            save=self.save,
                                                                            savepath=os.path.join(self.savepath, mode, str(self))
                                                                            )

        # save the predictor models
        for idx, trained_splitter in enumerate(trained_splitter_list):
            self.splitter_dict[idx] = trained_splitter

        # find the best cluster
        try:
            k = cv_testing_error.idxmin()  # DEBUG skip first errors
            self.CV_error = cv_testing_error.loc[k]
        except:
            k = self.n_iter

        # update the optimal number of clusters
        self.optimal_cluster_size = k
        if self.verbose >= 1:
            print('minimum iterations:', k)

        # error corresponding to chosen model

        # actual training on all the data

        splitter_df = MDP_Splitter(df.copy(),
                                   all_data=df.copy(),
                                   features=self.features,
                                   reward_name=self.reward_name,
                                   days_avg=self.days_avg,
                                   clustering=self.clustering_algorithm,
                                   init_n_clusters=self.n_clusters,
                                   distance_threshold=self.clustering_distance_threshold,
                                   error_computing=self.error_computing,
                                   horizon=self.horizon,
                                   alpha=self.alpha,
                                   actions=self.actions,
                                   random_state=self.random_state,
                                   verbose=self.verbose)

        splitter_df.initializeClusters(reward=["{}_SCALE".format(self.reward_name)])

        # k = df_train['CLUSTER'].nunique()
        #################################################################

        #################################################################
        # Run Iterative Learning Algorithm

        trained_splitter_df = splitter(splitter_df,
                                       th=self.splitting_threshold,
                                       test_splitter_dataframe=None,
                                       testing=False,
                                       classification=self.classification_algorithm,
                                       it=k,
                                       OutputFlag=self.verbose,
                                       random_state=self.random_state,
                                       plot=self.plot,
                                       save=self.save,
                                       savepath=os.path.join(self.savepath, mode, str(self),  "plot_final.PNG")
                                       )

        self.splitter_dict["k_opt"] = trained_splitter_df

        # storing trained dataset and predict_cluster function
        self.df_trained = df
        # self.classifier = splitter_df.model

        # store P_df and R_df values
        # self.P_df = splitter_df.P_df
        # self.R_df = splitter_df.R_df

        # store the initial clusters
        if self.keep_first:
            if self.verbose >= 2:
                print("Saving the initial clusters per region...")
            self.df_trained_first = self.df_trained.groupby(self.region_colname).first().copy()
        # store only the last states for prediction
        self.df_trained = self.df_trained.groupby(self.region_colname).last()
        # self.df_trained_calibrate = self.df_trained.groupby(self.region_colname).tail( np.ceil(20 / self.horizon)).copy()

    # calibrate MDP
    def calibrate(self, df_calibrate):

        self.calibration_dict = {}

        regions = list(set(df_calibrate[self.region_colname]))
        dates = list(set(df_calibrate[self.date_colname]))
        dates = sorted(dates)

        output = self.predict(regions, dates, model_key="all", n_jobs=-1)
        alphas = np.linspace(-4., 4., 200)
        for region in regions:
            output_region = output[region]
            res_agg = output_region.apply(lambda v: compute_exp_mean_alpha(v, alphas))
            #
            output_alpha = pd.DataFrame(index=res_agg.index, columns=alphas)
            for date in res_agg.index:
                output_alpha.loc[date] = res_agg[date].reshape(1, -1)
            for date in res_agg.index:
                output_alpha.at[date, "True"] = df_calibrate[(df_calibrate[self.region_colname] == region)
                                                                          & (df_calibrate[self.date_colname] == date)][self.target_colname].values[0]

            for alpha in alphas:
                output_alpha[alpha] = np.abs(output_alpha[alpha] - output_alpha["True"]) / output_alpha["True"]

            self.calibration_dict[region] = output_alpha.tail(7).mean()[:-1].idxmin()
        self.calibrated = True

    # calibrate MDP
    def calibrate_(self, df_calibrate):

        self.calibration_dict = {}

        regions = list(set(df_calibrate[self.region_colname]))
        dates = list(set(df_calibrate[self.date_colname]))
        dates = sorted(dates)

        output = self.predict(regions, dates, model_key="all", n_jobs=-1)
        alphas = np.linspace(-4., 4., 200)
        for region in regions:
            output_region = output[region]
            output_region_diff = output[region].copy()
            for date_idx in range(len(dates) - 1):
                output_region_diff.iloc[date_idx+1] = [tuple([_,__]) for _, __ in enumerate(list([_[1] for _ in output_region.iloc[date_idx+1]]) - np.array([_[1] for _ in output_region.iloc[date_idx]]))]
            output_region_diff = output_region_diff.iloc[1:]
            res_agg = output_region_diff.apply(lambda v: compute_exp_mean_alpha(v, alphas))
            #
            output_alpha = pd.DataFrame(index=res_agg.index, columns=alphas)
            for date in res_agg.index:
                output_alpha.loc[date] = res_agg[date].reshape(1, -1)
            for date in res_agg.index:
                output_alpha.at[date, "True"] = df_calibrate[(df_calibrate[self.region_colname] == region)
                                                                          & (df_calibrate[self.date_colname] == date)][self.target_colname].values[0]

            output_alpha["True_diff"] = output_alpha["True"].diff().values
            for alpha in alphas:
                output_alpha[alpha] = (output_alpha[alpha] - output_alpha["True_diff"])**2

            self.calibration_dict[region] = output_alpha.tail(7).mean()[:-2].idxmin()
        self.calibrated = True

    # predict() takes a state name and a time horizon, and returns the predicted
    # number of cases after h steps from the most recent datapoint
    def predict_region_ndays_alpha(self,
                             region,  # str: i.e. US state for prediction to be made
                             n_days,
                             model_key="k_opt",
                             from_first=False,
                             actions_df=None,
                             ):  # int: time horizon (number of days) for prediction
        # preferably a multiple of days_avg (default 3)
        try:
            if not (model_key in {"median", "best_r2", "best_err"}):
                h = int(np.floor(n_days/self.days_avg))
                delta = n_days - self.days_avg*h
                try:
                    mdp_predictor = self.splitter_dict[model_key]
                except KeyError:
                    print("MDPPredictionError: the model key doesn't exist")
                    raise MDPPredictionError

                # start from the initial dates
                if from_first:
                    df_clusters = self.df_trained_first
                # start from the last available dates
                else:
                    df_clusters = self.df_trained

                # get initial cases for the state at the latest datapoint
                region_id = df_clusters.loc[region, "ID"]
                target = mdp_predictor.current_state_date.loc[region_id, "TARGET"]
                risk = mdp_predictor.current_state_date.loc[region_id, "RISK"]
                date = mdp_predictor.current_state_date.loc[region_id, "TIME"]

                if not (actions_df is None):
                    actions_list = actions_df.loc[actions_df[self.region_colname] == region]
                    actions_list = list(actions_list[actions_list["TIME"] >= date]["ACTION"].values)
                    actions_list = actions_list + [0] * (max(h+1 -len(actions_list), 0))

                else:
                    actions_list = [0] * (h+1)
                if self.verbose > 0:
                    print(region, actions_list)

                if self.verbose >= 2:
                    print('current date:', date, '| current %s:'%self.target_colname, target)

                # cluster the this last point
                s = mdp_predictor.current_state_date.loc[region_id, "CLUSTER"]
                if self.verbose >= 2:
                    print('predicted initial cluster', s)

                r = 1.
                clusters_seq = [s]
                # run for horizon h, multiply out the ratios
                for i in range(h):
                    s = mdp_predictor.P_df.loc[s, actions_list[i]].values[0]
                    risk *= np.exp(mdp_predictor.R_df.loc[s][0])
                    r *= np.exp(risk)
                    clusters_seq.append(s)

                # last next cluster
                s = mdp_predictor.P_df.loc[s, actions_list[h]].values[0]
                clusters_seq.append(s)
                if self.verbose >= 2:
                    print('Sequence of clusters:', clusters_seq)
                pred = target * r * np.exp(risk * np.exp(mdp_predictor.R_df.loc[s][0]))**(float(delta/self.days_avg))

                if self.verbose >= 2:
                    print('Prediction for date:', date + timedelta(n_days), '| target:', pred)
                return pred

            # averaged prediction
            elif model_key == "median":
                preds = []
                for model_key_ in self.splitter_dict.keys():
                    if model_key_ != "k_opt":
                        preds.append(self.predict_region_ndays_alpha(
                            region,
                            n_days,
                            model_key=model_key_,
                            from_first=from_first
                        ))
                return np.median(preds)

            # best r2 score
            elif model_key == "best_r2":
                r2_model_dict = {key: (mdp_pred.R2_test[-1]+mdp_pred.R2_train[-1])/2 for key, mdp_pred in self.splitter_dict.items() if key != "k_opt"}
                best_key = max(r2_model_dict.items(), key=operator.itemgetter(1))[0]

                return self.predict_region_ndays_alpha(region, n_days, model_key=best_key, from_first=from_first)

            # best r2 score
            elif model_key == "best_err":
                r2_model_dict = {key: mdp_pred.testing_error.last() for key, mdp_pred in self.splitter_dict.items()}
                best_key = min(r2_model_dict.items(), key=operator.itemgetter(1))[0]

                return self.predict_region_ndays_alpha(region, n_days, model_key=best_key, from_first=from_first)
        except:
            return np.nan

# predict() takes a state name and a time horizon, and returns the predicted
# number of cases after h steps from the most recent datapoint
    def predict_region_ndays(self,
                             region,  # str: i.e. US state for prediction to be made
                             n_days,
                             model_key="k_opt",
                             from_first=False,
                             actions_df=None
                             ):  # int: time horizon (number of days) for prediction
        # preferably a multiple of days_avg (default 3)
        try:
            if not (model_key in {"median", "best_r2", "best_err", "mean", "std", "std_diff", 'all',
                                  'q75', 'exp_mean', "log_mean", 'exp_mean_diff', "sigmoid"}):
                h = int(np.floor(n_days/self.days_avg))
                delta = n_days - self.days_avg*h
                try:
                    mdp_predictor = self.splitter_dict[model_key]
                except KeyError:
                    print("MDPPredictionError: the model key doesn't exist")
                    raise MDPPredictionError

                # start from the initial dates
                if from_first:
                    df_clusters = self.df_trained_first
                # start from the last available dates
                else:
                    df_clusters = self.df_trained

                # get initial cases for the state at the latest datapoint
                region_id = df_clusters.loc[region, "ID"]
                target = mdp_predictor.current_state_date.loc[region_id, "TARGET"]
                date = mdp_predictor.current_state_date.loc[region_id, "TIME"]

                if not (actions_df is None):
                    actions_list = actions_df.loc[actions_df[self.region_colname] == region]
                    actions_list = list(actions_list[actions_list["TIME"] >= date]["ACTION"].values)
                    actions_list = actions_list + [0] * (max(h+1 - len(actions_list), 0))

                else:
                    actions_list = [0] * (h+1)
                if self.verbose >= 2:
                    print(region, actions_list)

                if self.verbose >= 2:
                    print('current date:', date, '| current %s:'%self.target_colname, target)

                # cluster the this last point
                s = mdp_predictor.current_state_date.loc[region_id, "CLUSTER"]
                if self.verbose >= 2:
                    print('predicted initial cluster', s)

                r = 1.
                clusters_seq = [s]
                # run for horizon h, multiply out the ratios
                for i in range(h):
                    s = mdp_predictor.P_df.loc[s, actions_list[i]].values[0]
                    risk = mdp_predictor.R_df.loc[s][0]
                    r *= np.exp(risk)
                    clusters_seq.append(s)

                # last next cluster
                s = mdp_predictor.P_df.loc[s, actions_list[h]].values[0]
                clusters_seq.append(s)
                if self.verbose >= 2:
                    print('Sequence of clusters:', clusters_seq)
                pred = target * r * np.exp(mdp_predictor.R_df.loc[s][0])**(float(delta/self.days_avg))

                if self.verbose >= 2:
                    print('Prediction for date:', date + timedelta(n_days), '| target:', pred)
                return pred

            # averaged prediction
            elif model_key == "all":
                preds = []
                for model_key_ in self.splitter_dict.keys():
                    if model_key_ != "k_opt":
                        preds.append((model_key_, self.predict_region_ndays(
                            region,
                            n_days,
                            model_key=model_key_,
                            from_first=from_first
                        )))

                return preds

            # averaged prediction
            elif model_key == "mean":
                preds = []
                for model_key_ in self.splitter_dict.keys():
                    if model_key_ != "k_opt":
                        preds.append(self.predict_region_ndays(
                            region,
                            n_days,
                            model_key=model_key_,
                            from_first=from_first
                        ))

                preds = np.array(preds)
                preds = preds[~np.isnan(preds)]
                return np.mean(preds)

            # averaged prediction
            elif model_key == "log_mean":
                preds = []
                for model_key_ in self.splitter_dict.keys():
                    if model_key_ != "k_opt":
                        preds.append(self.predict_region_ndays(
                            region,
                            n_days,
                            model_key=model_key_,
                            from_first=from_first
                        ))

                preds = np.array(preds)
                preds = preds[~np.isnan(preds)]
                return np.exp(np.mean(np.log(preds)))

            # exponentially weighted prediction
            elif model_key == "exp_mean":
                preds = []
                for model_key_ in self.splitter_dict.keys():
                    preds.append(self.predict_region_ndays(
                        region,
                        n_days,
                        model_key=model_key_,
                        from_first=from_first
                    ))

                preds = np.array(preds)
                preds = preds[~np.isnan(preds)]
                if self.calibrated:
                    return np.average(preds, weights=np.exp(- self.calibration_dict[region] * (preds - np.mean(preds)) / np.std(preds)))
                else:
                    return np.average(preds, weights=np.exp(- 1. * (preds - np.mean(preds)) / np.std(preds)))

            # exponentially weighted prediction
            elif model_key == "sigmoid":
                preds = []
                for model_key_ in self.splitter_dict.keys():
                    preds.append(self.predict_region_ndays(
                        region,
                        n_days,
                        model_key=model_key_,
                        from_first=from_first
                    ))

                preds = np.array(preds)
                preds = preds[~np.isnan(preds)]
                if self.calibrated:
                    weights = np.exp(- self.calibration_dict[region] * (preds - np.mean(preds)) / np.std(preds)) / (1. + np.exp(- self.calibration_dict[region] * (preds - np.mean(preds)) / np.std(preds)))
                    return np.average(preds, weights=weights)
                else:
                    return np.average(preds, weights=np.exp(- (preds - np.mean(preds)) / np.std(preds)) / (1 + np.exp(- (preds - np.mean(preds)) / np.std(preds))))

            # exponentially weighted prediction
            elif model_key == "exp_mean_diff":
                preds = []
                for model_key_ in self.splitter_dict.keys():
                    preds.append(self.predict_region_ndays(
                        region,
                        n_days,
                        model_key=model_key_,
                        from_first=from_first
                    )-self.predict_region_ndays(
                        region,
                        n_days-1,
                        model_key=model_key_,
                        from_first=from_first
                    ))

                preds = np.array(preds)
                preds = preds[~np.isnan(preds)]
                if self.calibrated:
                    return np.average(preds, weights=np.exp(- self.calibration_dict[region] * preds / np.std(preds)))
                else:
                    return np.average(preds, weights=np.exp(- 0.5 * preds / np.std(preds)))

            # quantile 75 mean
            elif model_key == "q75":
                preds = []
                for model_key_ in self.splitter_dict.keys():
                    if model_key_ != "k_opt":
                        preds.append(self.predict_region_ndays(
                            region,
                            n_days,
                            model_key=model_key_,
                            from_first=from_first
                        ))

                preds = np.array(preds)
                preds = preds[~np.isnan(preds)]
                return preds[np.where(preds < np.quantile(preds, 0.9))].mean()

            # averaged prediction
            elif model_key == "median":
                preds = []
                for model_key_ in self.splitter_dict.keys():
                    if model_key_ != "k_opt":
                        preds.append(self.predict_region_ndays(
                            region,
                            n_days,
                            model_key=model_key_,
                            from_first=from_first
                        ))

                preds = np.array(preds)
                preds = preds[~np.isnan(preds)]
                return np.median(preds)

            # averaged prediction
            elif model_key == "std_diff":
                preds = []
                for model_key_ in self.splitter_dict.keys():
                    if model_key_ != "k_opt":
                        preds.append(self.predict_region_ndays(
                            region,
                            n_days,
                            model_key=model_key_,
                            from_first=from_first
                        )-self.predict_region_ndays(
                            region,
                            n_days-1,
                            model_key=model_key_,
                            from_first=from_first
                        ))

                preds = np.array(preds)
                preds = preds[~np.isnan(preds)]
                return np.std(preds)

            # best r2 score
            elif model_key == "best_r2":
                r2_model_dict = {key: (mdp_pred.R2_test[-1]+mdp_pred.R2_train[-1])/2 for key, mdp_pred in self.splitter_dict.items() if key != "k_opt"}
                best_key = max(r2_model_dict.items(), key=operator.itemgetter(1))[0]

                return self.predict_region_ndays(region, n_days, model_key=best_key, from_first=from_first)

            # averaged prediction
            elif model_key == "std":
                preds = []
                for model_key_ in self.splitter_dict.keys():
                    if model_key_ != "k_opt":
                        preds.append(self.predict_region_ndays(
                            region,
                            n_days,
                            model_key=model_key_,
                            from_first=from_first
                        ))

                preds = np.array(preds)
                preds = preds[~np.isnan(preds)]
                return np.std(preds)

            # best r2 score
            elif model_key == "best_r2":
                r2_model_dict = {key: (mdp_pred.R2_test[-1]+mdp_pred.R2_train[-1])/2 for key, mdp_pred in self.splitter_dict.items() if key != "k_opt"}
                best_key = max(r2_model_dict.items(), key=operator.itemgetter(1))[0]

                return self.predict_region_ndays(region, n_days, model_key=best_key, from_first=from_first)

            # best err score
            elif model_key == "best_err":
                r2_model_dict = {key: mdp_pred.testing_error.last() for key, mdp_pred in self.splitter_dict.items()}
                best_key = min(r2_model_dict.items(), key=operator.itemgetter(1))[0]

                return self.predict_region_ndays(region, n_days, model_key=best_key, from_first=from_first)
        except:
            return np.nan

    # predict_all() takes a time horizon, and returns the predicted number of
    # cases after h steps from the most recent datapoint for all states
    def predict_allregions_ndays(self,
                                 n_days,
                                 from_first=False,
                                 model_key="median"): # time horizon for prediction, preferably a multiple of days_avg (default 3)
        df = self.df_trained.copy()
        df = df[['TIME', "TARGET"]]
        df['TIME'] = df['TIME'] + timedelta(n_days)
        if self.reward_name == "RISK":
            df["TARGET"] = df.index.map(
                lambda region: np.ceil(self.predict_region_ndays(region, n_days, from_first=from_first, model_key=model_key)))
        elif self.reward_name == "AlphaRISK":
            df["TARGET"] = df.index.map(
                lambda region: np.ceil(self.predict_region_ndays_alpha(region, n_days, from_first=from_first, model_key=model_key)))
        return df

    # predict_class() takes a dictionary of states and time horizon and returns their predicted number of cases
    def predict(self,
                regions,  # list of states to predict the target
                dates,  # list of dates to predict the target
                model_key="k_opt",
                from_first=False,
                n_jobs=1):

        # instantiate the prediction dataframe
        pred_df = pd.DataFrame(columns=[self.region_colname, 'TIME', self.target_colname])

        # get the last dates for each states
        region_set = set(self.df_trained.index)

        if n_jobs in {0, 1}:
            for region in regions:

                pred_region_df = process_prediction_region(region, self, dates, region_set, model_key=model_key, from_first=from_first)
                pred_df = pred_df.append(pred_region_df, ignore_index=True)

        # maximum cpu allowed
        if n_jobs == -1:
            pool = mp.Pool(processes=mp.cpu_count())
        elif isinstance(n_jobs, int) & (n_jobs <= mp.cpu_count()) :
            pool = mp.Pool(processes=n_jobs)
        else:
            raise MDPPredictionError("n_jobs should be an integer between -1 and {}".format(mp.cpu_count()))

        prediction_regions_pool = partial(process_prediction_region,
                                          mdp=self,
                                          dates=dates,
                                          region_set=region_set,
                                          model_key=model_key,
                                          from_first=from_first)

        # apply in parallel fitting function
        prediction_per_region = pool.map(prediction_regions_pool, regions)
        for _ in prediction_per_region:
            pred_df = pred_df.append(_, ignore_index=True)

        pred_df.rename(columns={'TIME': self.date_colname}, inplace=True)
        pred_dic = {state: pred_df[pred_df[self.region_colname] == state].set_index([self.date_colname])[self.target_colname] for state in pred_df[self.region_colname].unique()}

        return pred_dic


class MDPGridSearch:

    def __init__(self,
                 target_colname,
                 region_colname,
                 date_colname,
                 features_list,
                 action_thresh,
                 hyperparams,
                 verbose=0,
                 random_state=1234,
                 n_folds_cv=5,
                 n_jobs=1,
                 mdp_n_jobs=1,
                 mode="TIME_CV",
                 keep_first=True,
                 ignore_errors=True,
                 plot=False,
                 save=False,
                 savepath=None
                 ):

        # Multiprocessing restriction of jobs attribution
        try:
            assert ((n_jobs == 1) | (mdp_n_jobs == 1))
        except AssertionError:
            print("MultiProcessingWarning: cannot process Sub-Serialize tasks on MDP and MDP GradSearch "
                  "(n_jobs (GS) ={}, mdp_n_jobs (MDP) ={}. Default Change : mdp_n_jobs=1)".format(n_jobs, mdp_n_jobs))
            mdp_n_jobs = 1

        self.target_colname = target_colname  # column name of the target, i.e. 'cases'
        self.region_colname = region_colname  # column name of the region, i.e. 'state'
        self.date_colname = date_colname  # column name of the date, i.e. 'date'
        self.features_list = features_list  # list of the features that are considered to be trained
        #                                     PS: feature[0] --> Action feature
        self.action_thresh = action_thresh  # ([list of action threshold], default action)  # ex : ([-0.5, 0.5], 1) --> ACTION=[-1, 0, 1]
        self.hyperparams = hyperparams  # dictionary of MDP hyperparameters
        self.verbose = verbose  # define the precision OutputFlag
        self.random_state = random_state  # random seed using the random generator(s)
        self.n_jobs = n_jobs  # number of jobs for multiprocessing for the gridsearch
        self.mdp_n_jobs = mdp_n_jobs  # number of jobs for multiprocessing for the MDPs
        self.mode = mode  # mode : either 'ID', 'TIME_CV', or 'ALL' defines how the cross validation is proceed
        self.n_folds_cv = n_folds_cv  # number of folds for the cross validation
        self.keep_first = keep_first  # bool for keeping the initial clusters (experiment purpose)

        self.ignore_errors = ignore_errors  # boolean for interrupting the learning process if there is an error occurring
        self.plot = plot  # plot out the training error curves (without interrupting the run)
        self.save = save  # save the results (plots)
        self.savepath = savepath  # path to the save folder

        self.training_error = dict()  # dictionary of the training error by parameter set id
        self.best_estimator_ = None  # trained MDP model with the model
        self.best_params_ = None  # best set of parameters
        self.testing_error = dict()  # dictionary of the testing error by parameter set id
        self.params_dict = dict()  # dictionary of the parameters by parameter set id
        self.all_estimators_dict = dict()  # dictionary of the trained MDP models by parameter set id

    # Fitting method to learn from the data
    # data (and testing_data if not None) should be a pandas.DataFrame object containing the provided columns
    # testing_data : data to compute validation error, otherwise (None) use fitting validation error
    def fit(self,
            data,
            testing_data=None,
            ):

        hparams_items = self.hyperparams.items()

        stack_features = ["n_folds_cv", "target_colname", "region_colname", "date_colname", "features_list", "action_thresh",
                          "verbose", "random_state", "n_jobs", "save", "savepath", "plot"]
        hparams_keys = [_[0] for _ in hparams_items if not (_[0] in stack_features)] + stack_features

        hparams_values = [_[1] for _ in hparams_items] \
                         + [[self.n_folds_cv], [self.target_colname], [self.region_colname], [self.date_colname],
                            [self.features_list], [self.action_thresh], [self.verbose], [self.random_state],
                            [self.mdp_n_jobs], [self.save], [self.savepath], [self.plot]]
        hparams_values = tuple(hparams_values)

        # construct the cartesian product of hyper-parameters
        params_set = product(*hparams_values)
        self.params_dict = {i: dict(zip(hparams_keys, hparam)) for i, hparam in enumerate(params_set)}
        self.all_estimators_dict = {key: MDPModel(**params) for key, params in self.params_dict.items()}

        del params_set, hparams_keys, hparams_values

        grid_results_ = fit_eval(mdp_dict=self.all_estimators_dict,
                                 data=data,
                                 testing_data=testing_data,
                                 mode=self.mode,
                                 ignore_errors=self.ignore_errors,
                                 n_jobs=self.n_jobs,
                                 verbose=self.verbose)

        # save results from grid_search
        for param_id, mdp, error in grid_results_:
            self.all_estimators_dict[param_id] = mdp
            self.testing_error[param_id] = error

        # get the best model
        best_params_id = min(self.testing_error.items(), key=operator.itemgetter(1))[0]
        self.best_params_ = self.params_dict[best_params_id]
        self.best_estimator_ = MDPModel(**self.best_params_)


if __name__ == "__main__":
    import os
    import warnings
    import datetime
    from copy import deepcopy
    warnings.filterwarnings("ignore")  # to avoid Python deprecated version warnings

    # path = 'C:/Users/omars/Desktop/covid19_georgia/large_data/input/'
    path = os.path.join(os.path.dirname(os.getcwd()), 'data', 'input')
    file = '05_27_states_combined_v2_w_trend.csv'
    training_cutoff = '2020-05-25'
    nmin = 20
    deterministic = True
    if deterministic:
        deterministic_label = ''
    else:
        deterministic_label = 'markov_'
    target_colname = 'deaths'
    mdp_region_colname = 'state'  # str, col name of region (e.g. 'state')
    mdp_date_colname = 'date'  # str, col name of time (e.g. 'date')
    mdp_features_list = ["cases_pct3", "cases_pct5"]  # list of strs: feature columns

    sgm = .1
    n_iter_mdp = 150
    n_iter_ci = 10
    ci_range = 0.75

    df_orig = pd.read_csv(os.path.join(path, file))
    print('Data Wrangling in Progress...')
    df = deepcopy(df_orig)
    df.columns = map(str.lower, df.columns)

    df = df[df[target_colname] >= nmin]
    df['date'] = df['date'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))
    df_orig['date'] = df_orig['date'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))
    df = df.sort_values(by=['state', 'date'])
    states = sorted(list(set(df['state'])))
    pop_df = df.loc[:, ['state', 'population']]
    pop_dic = {pop_df .iloc[i, 0]: pop_df .iloc[i, 1] for i in range(pop_df .shape[0])}
    features = list(df.columns[5:35])

    df = df.loc[:, df.columns[df.isnull().sum() * 100 / len(df) < 20]]
    features = list(set(features).intersection(set(df.columns)))

    df_train = df[df['date'] <= training_cutoff]
    df_test = df[df['date'] > training_cutoff]
    pred_out = len(set(df_test.date))
    day_0 = str(df_test.date.min())[:10]
    df_train = df_orig[df_orig['date'] <= training_cutoff]
    print('Data Wrangling Complete.')

    # ####### test fitting methods ##########
    mdp_test_fitting = True

    if mdp_test_fitting:
        print('MDP Model Training in Progress...')
        mdp = MDPModel(
            target_colname=target_colname,  # str: col name of target_colname (i.e. 'deaths')
            region_colname=mdp_region_colname,  # str, col name of region (i.e. 'state')
            date_colname=mdp_date_colname,  # str, col name of time (i.e. 'date')
            features_list=mdp_features_list,  # list of strs: feature columns
            horizon=6,
            error_function_name="relative",
            error_computing="exponential",
            alpha=1e-5,
            n_iter=n_iter_mdp,
            days_avg=3,
            n_folds_cv=6,
            clustering_distance_threshold=0.1,
            classification_algorithm="RandomForestClassifier",
            verbose=1,
            action_thresh=([], 0),  # (-1e10, -1000) --> no action
            random_state=1234,
            n_jobs=1,
            keep_first=True,
            plot=True,
            save=True,
            savepath=r"C:\Users\david\Desktop\MIT\Courses\Research internship\results\11 - test algo ID")

        mdp_abort = False
        # try:
        mdp.fit(df_train, mode="TIME_CV")
        # except ValueError:
        #     print('ERROR: Feature columns have missing values! Please drop'
        #           'rows or fill in missing data.')
        #     print('MDP Model Aborted.')
        #     mdp_abort = True
        #     run_mdp = False

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
                example_dict = (["Alabama", "Gabon", "Iowa", "Massachusetts"], ["2019-06-14", "2020-05-14",
                                                                                "2020-06-01", "2020-07-01"])
                mdp_output = mdp.predict(*example_dict)

                print(mdp_output)
        print('MDP Model (test) Complete.')

    # ####### MDP Grid Search test fitting methods ##########
    mdpgs_test_fitting = False

    if mdpgs_test_fitting:
        print('MDP Grid Search Model Training in Progress...')

        hparams = {
            "horizon": [3, 5, 7],
            "n_iter": [50],
            "days_avg": [3, 4, 5],
            "clustering_distance_threshold": [0.08, 0.1, .12],
            "classification_algorithm": ["DecisionTreeClassifier"]
        }
        mdp_gs = MDPGridSearch(
            target_colname=target_colname,  # str: col name of target_colname (i.e. 'deaths')
            region_colname=mdp_region_colname,  # str, col name of region (i.e. 'state')
            date_colname=mdp_date_colname,  # str, col name of time (i.e. 'date')
            features_list=mdp_features_list,  # list of strs: feature columns
            action_thresh=([], 0),
            hyperparams=hparams,
            n_folds_cv=4,
            verbose=0,
            random_state=1234,
            n_jobs=2,
            mdp_n_jobs=2,
            mode="TIME_CV",
            ignore_errors=True,
            save=True,
            savepath=r"C:\Users\david\Desktop\MIT\Courses\Research internship\results\00 - test GridSearch algo")

        mdp_gs.fit(df_train)
