# -*- coding: utf-8 -*-
"""
Created on Wed May 27 10:57:04 2020

@author: Amine
"""

from model import *

df = pd.read_csv('state_mobility_combined.csv')

m = MDP_model()
m.fit(df, # csv file name with data OR data frame
        h=5, # time horizon for cross val = n_days/d_avg
        n_iter=70, # number of iterations, gives an upper bound on the #of clusters wanted.
        d_avg=3, # int: number of days to average data over
        distance_threshold = 0.05) #Tolerated error in each cluster. param to be tuned (range advised: 0.05-0.3)

#prediction fora single state
m.predict('Maryland',n_days = 15)

#prediction for all state, otuput as a df
m.predict_all(n_days =15)
