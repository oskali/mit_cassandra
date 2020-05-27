# -*- coding: utf-8 -*-
"""
This file is the main file to run the MDP clustering algorithm

 Specific to COVID-19 Research

Created on Sun April 7 18:51:20 2020

@author: omars
"""

#################################################################
# Set Working Directory
#import os
#os.chdir("C:/Users/omars/Desktop/Georgia/opioids/iterativeMDP/") # To change
from MDPtools import Generate_random_MDP, sample_MDP_with_features_list
import numpy as np
import pandas as pd
#################################################################


#################################################################
# Load Libraries
from covid_states_functions import createSamples, splitter, \
                        initializeClusters, split_train_test_by_id, \
                        fit_CV
                        
from covid_testing import *
#################################################################


#################################################################
# Set Parameters
new_cols = ['state', 'date', 'cases', 'mobility_score']
# currently ACTION all set at 0
action_thresh = [-100, 0, 1, 3, 10] # corresponding to ACTION [-1, 0, 1, 2] respectively
d_delay = 7
d_avg = 0
clustering = 'Agglomerative'
distance_threshold = 0.3 #Max diameter of initial clusters
n_clusters = None
random_state = 0
k = n_clusters
classification = 'DecisionTreeClassifier'
n_iter = 5
th = 0 #int(0.1*N*(T-1)/n) #Threshold to stop splitting
ratio = 0.3 # portion of data to be used for testing
cv = 5
h = 8
#################################################################


#################################################################
# Import data
df = pd.read_csv('state_mobility_combined.csv')
#################################################################


#################################################################
# Create samples & split data
df, pfeatures = createSamples(df, action_thresh, d_avg)
#print('Features:', pfeatures)
print(df)
#df.to_csv('clusters.csv')



#################################################################
# Cross Validation Function
list_training_R2,list_testing_R2,df_new, df_test =fit_CV(df,
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

'''

df_train, df_test = split_train_test_by_id(df, ratio, 'ID')
#################################################################
# Initialize Clusters

df = initializeClusters(df_train,
                        clustering=clustering,
                        n_clusters=n_clusters,
                        random_state=random_state)

print('initialized clusters', df)
#################################################################


#################################################################
# Run Iterative Learning Algorithm
df_new,training_R2,testing_R2 = splitter(df,
                                  pfeatures,
                                  k,
                                  th,
                                  df_test,
                                  classification,
                                  n_iter,
                                  OutputFlag = 1)
#################################################################
#print(purity(df_new))
#plot_features(df)
model = predict_cluster(df_new, pfeatures)

#print('training accuracy:',training_accuracy(df_new)[0])
print('training error:', training_value_error(df_new, relative = True, h = 5))
print('testing error:', testing_value_error(df_test, df_new, model, pfeatures, relative = True, h = 5))
#print('training R2:', R2_value_training(df_new))
#print('testing R2:', R2_value_testing(df_test, df_new, model, pfeatures))



# Export final results

#print(df_new.groupby(['CLUSTER', 'ACTION', 'NEXT_CLUSTER'])['ACTION'].count())

print('Overview of clusters by RISK')
print(df_new.groupby(['CLUSTER'])['RISK'].describe())

print('Overview of clusters by Next_Cluster')
print(df_new.groupby(['CLUSTER', 'NEXT_CLUSTER'])['cases'].count())

#df.to_csv('clusters.csv')
df_new.to_csv('clusters.csv')
'''