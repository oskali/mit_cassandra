# -*- coding: utf-8 -*-
"""
Created on Sun April 7 18:51:20 2020

@author: omars
"""

#%% Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import binascii
from sklearn.model_selection import GroupKFold
from tqdm import tqdm
import os

import multiprocessing as mp
from functools import partial
import warnings
warnings.filterwarnings("ignore")
from mdp_utils import fit_cv_fold


#%% Funtions for Initialization

# createSamples() takes the original dataframe from combined data,
# the names of columns of features to keep, the treshold values to determine
# what type of action is made based on the FIRST feature of features_list,
# days_avg the number of days used to compress datapoints, and returns a data frame with
# desired features and history, ratio values and history, and 'RISK' and 'ACTION'
# returns new dataframe with only the desired columns, number of features considered
def createSamples(df,#, # dataframe: original full dataframe
                  # new_cols, # str list: names of columns to be considered
                  target_colname,  # str: col name of target_colname (i.e. 'deaths')
                  region_colname,  # str, col name of region (i.e. 'state')
                  date_colname,  # str, col name of time (i.e. 'date')
                  features_list,  # list of str: i.e. (['mobility', 'testing'])
                  action_thresh_base,  # int list: defining size of jumps in stringency
                  # d_delay, # int: day lag before calculating death impact
                  days_avg,  # int: # of days to average when reporting death
                  region_exceptions=None):

    df.sort_values(by=[region_colname, date_colname], inplace=True)
    df.rename(columns={date_colname: 'TIME'}, inplace=True)

    # remove exceptions
    if not (region_exceptions is None):
        df = df[~(df[region_colname].isin(region_exceptions))]

    action_thresh, no_action_id = action_thresh_base

    # new_cols = ['state', 'date', 'cases', 'mobility_score']
    if target_colname not in features_list:
        new_cols = [region_colname] + ['TIME'] + [target_colname] + features_list
    else:
        new_cols = [region_colname] + ['TIME'] + features_list
    df_new = df[new_cols]

    # df_new.rename(columns = {df_new.columns[1]: 'TIME'}, inplace = True)
    ids = df_new.groupby([region_colname]).ngroup()
    df_new.insert(0, 'ID', ids, True)

    # print(df.columns)
    df_new.loc[:, ['TIME']]= pd.to_datetime(df_new['TIME'])
    dfs = []
    for region_name, group_region in df_new.groupby(region_colname):
        first_date = group_region.TIME.min()
        last_date = group_region.TIME.max()
        date_index = pd.date_range(first_date, last_date, freq="1D")
        date_index.name = 'TIME'
        group_ = pd.DataFrame(index=date_index)
        group_ = group_.join(group_region.set_index("TIME"))
        if group_.shape[0] != group_region.shape[0]:
            print("Missing dates: {} {} - {} missing rows".format(region_colname,
                                                                  region_name,
                                                                  group_.shape[0] - group_region.shape[0]))

            last_missing_date = group_[group_["ID"].isnull()].tail(1).index[0]
            print("last missing date: {}".format(str(last_missing_date)))
            group_ = group_[group_.index > last_missing_date].copy()
        dfs.append(group_)

    df_new = pd.concat(dfs)
    # print(df_new)

    # calculating stringency based on sum of actions
    # df['StringencyIndex'] = df.iloc[:, 3:].sum(axis=1)

    # add a column for action, categorizing by change in stringency index
    # df['StringencyChange'] = df['StringencyIndex'].shift(-1) - df['StringencyIndex']
    # df.loc[df['ID'] != df['ID'].shift(-1), 'StringencyChange'] = 0
    # df.loc[df['StringencyIndex'] == '', 'StringencyChange'] = 0

    # print(df.loc[df['ID']=='California'])

    # resample data according to # of days
    g = df_new.groupby(['ID'])
    cols = df_new.columns
    # print('cols', cols)
    dictio = {i:'last' for i in cols}
    for key in set([target_colname]+features_list):
        dictio[key] = 'mean'
    # dictio['StringencyChange'] = 'sum'
    # del dictio['TIME']
    df_new = g.resample('%sD' %days_avg).agg(dictio)
    # df_new = g.resample('3D').mean()
    # print('new', df_new)
    df_new = df_new.drop(columns=['ID'])
    df_new = df_new.reset_index()

    # creating feature lag 1, feature lag 2 etc.
    df_new.sort_values(by=["ID", "TIME"], inplace=True)
    for f in features_list:
        df_new[f+'-1'] = df_new.groupby("ID")[f].shift(1)
        df_new[f+'-2'] = df_new.groupby("ID")[f].shift(2)

    # deleting target == 0
    df_new = df_new.loc[df_new[target_colname] != 0, :]

    # creating r_t, r_t-1, etc ratio values from cases
    df_new['r_t'] = df_new.groupby("ID")[target_colname].pct_change(1) + 1
    df_new['r_t-1'] = df_new.groupby("ID")['r_t'].shift(1)
    df_new['r_t-2'] = df_new.groupby("ID")['r_t'].shift(2)

    new_features = [f+'-1' for f in features_list] + [f+'-2' for f in features_list] + ['r_t', 'r_t-1', 'r_t-2']
    df_new.dropna(subset=new_features,
                  inplace=True)

    # Here we assign initial clustering by r_t
    df_new['RISK'] = np.log(df_new['r_t'])

    # create action
    if len(action_thresh) == 0:
        df_new['ACTION'] = 0
        pfeatures = len(df_new.columns)-5
    else:
        action_thresh = [-1e20] + action_thresh + [1e20]
        actions = list(range(-no_action_id, len(action_thresh)-1-no_action_id)) #[0, 1] #[0, 5000, 100000]
        df_new[features_list[0]+'_change'] = df_new[features_list[0]+'-1']-\
            df_new[features_list[0]+'-2']
        df_new['ACTION'] = pd.cut(df_new[features_list[0]+'_change'], bins=action_thresh, right=False, labels=actions)

        # set the no action to 0
        pfeatures = len(df_new.columns)-6

    # df_new = df_new[df_new['r_t'] != 0]
    df_new = df_new.reset_index()
    df_new = df_new.drop(columns=['index'])
    # moving region col to the end, since not a feature
    if target_colname not in features_list:
        df_new = df_new.loc[:, [c for c in df_new if c not in [region_colname, target_colname]]
           + [region_colname] + [target_colname]]
        pfeatures -= 1
    else:
        df_new = df_new[[c for c in df_new if c not in [region_colname]]
           + [region_colname]]

    # Drop all rows with empty cells
    # df_new.dropna(inplace=True)

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

# (MDP functions)
# Fitting function used for the MDP fitting
# this function realize the split according to the mode,
# compute the training and testing errors for each fold and the averaged cross-validation error (training and testing)
def fit_cv(df,
           pfeatures,
           splitting_threshold,
           clustering,
           clustering_distance_threshold,
           classification,
           n_iter,
           n_clusters,
           horizon=5,
           OutputFlag=0,
           cv=3,
           n=-1,
           random_state=1234,
           n_jobs=1,
           mode="ID",
           plot=False,
           save=False,
           savepath=""):

    df_training_error = pd.DataFrame(columns=['Clusters'])
    df_testing_error = pd.DataFrame(columns=['Clusters'])
    testing_errors = []

    # shuffle ID's and create a new column 'ID_shuffle'
    random.seed(random_state)
    g = [df for _, df in df.groupby('ID')]
    random.shuffle(g)
    df = pd.concat(g).reset_index(drop=True)
    ids = df.groupby(['ID'], sort=False).ngroup()
    df['ID_shuffle'] = ids

    if cv in {0, 1} or mode == "ALL":
        split_ = [(0, (df.index, df.index))]

    # cross validation
    else:
        split_ = enumerate(GroupKFold(n_splits=cv).split(df, y=None, groups=df['ID_shuffle']))

    ###############################
    # --- BEGIN PARALLELED TASK ---

    # training is done in queue
    if n_jobs in {0, 1}:
        for train_test_idx in split_:
            # cv_bar.set_description("Cross-Validation... | Test set # %i" %i)
            testing_error, training_error, df_err, _ = fit_cv_fold(train_test_idx,
                                                                   df,
                                                                   clustering=clustering,
                                                                   n_clusters=n_clusters,
                                                                   clustering_distance_threshold=clustering_distance_threshold,
                                                                   pfeatures=pfeatures,
                                                                   splitting_threshold=splitting_threshold,
                                                                   classification=classification,
                                                                   n_iter=n_iter,
                                                                   horizon=horizon,
                                                                   n=n,
                                                                   OutputFlag=OutputFlag,
                                                                   random_state=random_state,
                                                                   plot=plot,
                                                                   save=save,
                                                                   savepath=savepath,
                                                                   mode=mode)

            # save the training and testing error
            df_training_error = df_training_error.merge(training_error, how='outer', on=['Clusters'])
            df_testing_error = df_testing_error.merge(testing_error, how='outer', on=['Clusters'])
            testing_errors.append(df_err)
            # DEBUG : not used E_v ?

    # training using multiprocessing
    else:

        # maximum cpu allowed
        if n_jobs == -1:
            pool = mp.Pool(processes=mp.cpu_count())
        else:
            pool = mp.Pool(processes=n_jobs)
        fit_cv_fold_pool = partial(fit_cv_fold,
                                   df=df,
                                   clustering=clustering,
                                   n_clusters=n_clusters,
                                   clustering_distance_threshold=clustering_distance_threshold,
                                   pfeatures=pfeatures,
                                   splitting_threshold=splitting_threshold,
                                   classification=classification,
                                   n_iter=n_iter,
                                   horizon=horizon,
                                   n=n,
                                   OutputFlag=OutputFlag,
                                   random_state=random_state,
                                   plot=plot,
                                   save=save,
                                   savepath=savepath,
                                   mode=mode)

        # apply in parallel fitting function
        results = pool.map(fit_cv_fold_pool, split_)

        # save the training and testing error
        for testing_error, training_error, df_err, _ in results:
            df_training_error = df_training_error.merge(training_error, how='outer', on=['Clusters'])
            df_testing_error = df_testing_error.merge(testing_error, how='outer', on=['Clusters'])
            testing_errors.append(df_err)

    # --- END PARALLELED TASK ---
    #############################

    df_training_error.set_index('Clusters', inplace=True)
    df_testing_error.set_index('Clusters', inplace=True)
    df_training_error.dropna(inplace=True)
    df_testing_error.dropna(inplace=True)
    # print(df_training_error)
    # print(df_testing_error)
    cv_training_error = np.mean(df_training_error, axis=1)
    cv_testing_error = np.mean(df_testing_error, axis=1)
    # print(cv_training_error)
    # print(cv_testing_error)

    if plot:
        fig1, ax1 = plt.subplots()
        # its = np.arange(k+1,k+1+len(cv_training_error))
        ax1.plot(cv_training_error.index.values, cv_training_error, label="CV Training Error")
        # ax1.plot(its, cv_testing_error, label = "CV Testing Error")
        ax1.plot(cv_testing_error.index.values, cv_testing_error, label="CV Testing Error")
        # ax1.plot(its, training_acc, label = "Training Accuracy")
        # ax1.plot(its, testing_acc, label = "Testing Accuracy")
        if n > 0:
            ax1.axvline(x=n, linestyle='--', color='r')  # Plotting vertical line at #cluster =n
        ax1.set_ylim(0)
        ax1.set_xlabel('# of Clusters')
        ax1.set_ylabel('Mean CV Error or Accuracy %')
        ax1.set_title('Mean CV Error and Accuracy During Splitting')
        ax1.legend()
        if save:
            plt.savefig(os.path.join(savepath, "plot_mean.PNG"))
        if OutputFlag >= 2:
            plt.show(block=False)
        else:
            plt.close()

    # for t in testing_errors:
    #    print(t)

    return cv_training_error, cv_testing_error


# (MDP GridSearch Function)
# Fitting function used for the GrdiSearch applied to the MDP fitting
# this function realize the split according to the mode,
# compute the training and testing errors for each fold and the averaged cross-validation error (training and testing)
def fit_eval(mdp_dict,
             data,
             testing_data=None,
             mode="TIME_CV",
             ignore_errors=True,
             n_jobs=2,
             verbose=0,
             ):

    results_ = []

    if verbose >= 0:
        print("""
        
------------------------------------
Starting process...
MDP Grid Search: #parameter set = {}, mode = {}, n_jobs = {}

        """.format(len(mdp_dict.items()), mode, n_jobs))

    if n_jobs in {0, 1}:
        for param_id_mdp in tqdm(mdp_dict.items()):
            param_id, mdp, error = fit_eval_params(param_id_mdp,
                                                   data,
                                                   testing_data=testing_data,
                                                   mode=mode,
                                                   ignore_errors=ignore_errors)

            results_.append((param_id, mdp, error))

    else:
        if n_jobs == -1:
            pool = mp.Pool(processes=mp.cpu_count())
        else:
            pool = mp.Pool(processes=n_jobs)

        fit_eval_params_pool = partial(fit_eval_params,
                                       data=data,
                                       testing_data=testing_data,
                                       mode=mode,
                                       ignore_errors=ignore_errors)

        # apply the multiprocessing over each split
        results_ = pool.map(fit_eval_params_pool, tqdm(mdp_dict.items()))

    if verbose >= 0:
        print("""
        
MDP Grid Search process done!
------------------------------------
        
        """)
    return results_
