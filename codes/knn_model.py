# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 01:55:57 2020
Last update including clusters & tunning per region 25/10/2020
@authors: Yiannis, Bibha, Margaret
"""

#%% Libraries
from datetime import datetime
import numpy as np
import pandas as pd
from knn_utils import cassandra_fit, cassandra_predict , mod_date , days_between ,add_one_fake_day_forward, time_clustering
from copy import deepcopy
from sklearn.preprocessing import normalize
from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans
from kneed import KneeLocator
#%% Model

class KNNModel:

    def __init__(self,
                 date='date',
                 region='state',
                 target='cases',
                 memory = 10,
                 real_GR = True,
                 r = 2,
                 deterministic=True,
                 use_mob = False,
                 mob_feature = 'retail_and_recreation',
                 starting_point = 'low',
                 hfilter = True ,
                 clusters_map = {},
                 active_cases = False,
                 extra_features = [],
                 region_best_parameters = {}):

        self.date = date
        self.region = region
        self.target = target
        self.memory = memory
        self.real_GR = real_GR
        self.r = r
        self.deterministic = deterministic
        self.use_mob = use_mob
        self.mob_feature = mob_feature
        self.starting_point = starting_point
        self.hfilter = hfilter
        self.state_df = None
        self.split_date = None
        self.day1 = None
        self.startdates = None
        self.region_best_parameters = None
        self.clusters_map = clusters_map
        self.active_cases = active_cases
        self.extra_features = extra_features
        self.region_best_parameters = region_best_parameters

    def give_parameters(self):
        return self.region_best_parameters




    def fit(self, df , regions):



        self.state_df = df
        self.split_date = max( self.state_df[self.date] )    #we should add 1 here to make split date first of testdates
        print(self.state_df)
        print(list(self.state_df.columns) )

        # asterios
        # ['state', 'date', 'cases', 'deaths', 'temperature', 'prcp_median_scaled']
        self.state_df = self.state_df[['state', 'date', 'cases', 'deaths','temperature']]
        self.state_df = self.state_df.loc[self.state_df.state.isin(regions)]
        self.state_df.temperature = self.state_df.temperature.shift(5).fillna(method='backfill')
        # self.state_df.growthcasespred = self.state_df.growthcasespred.shift(20).fillna(method='backfill')
        # self.state_df.prcp_median_scaled = self.state_df.prcp_median_scaled.shift(5).fillna(method='backfill')
        self.state_df = self.state_df.loc[(self.state_df.date <= self.split_date)]

        # clusters_ = time_clustering(self.state_df, self.split_date, days_before=30,
        #             date_col=self.date, region_col=self.region, target_col=self.target)

        # clusters = [x for x in clusters_ if x != []]        
        # self.clusters_map = clusters
        clusters = [list(self.state_df[self.region].unique())]


        d = {}
        for c in clusters:
            for state in c:
                d[state] = c
        self.clusters_map = d

        if self.region == 'fips':
            
            self.day1 = '2020-05-10' #training overall starting date
            min_case_requirement = 1
            
        else:
            self.day1 = '2020-04-10' #training overall starting date

            min_case_requirement = 1

        if self.target == 'deaths':
            min_case_requirement = 1

        mask = self.state_df[self.target] >= min_case_requirement #need to remove regions with only few cases at the end
        self.state_df = self.state_df[mask]
        self.state_df = self.state_df.sort_values(by = [self.region, self.date])


        min_dates = self.state_df[[self.region, self.date]].groupby(self.region).aggregate(min)
        self.startdates = {region : mod_date(min_dates.loc[region][self.date],30) for region in self.state_df[self.region].unique()}

        if self.use_mob:
            for state in self.state_df[self.region].unique():
                arr = self.state_df.loc[self.state_df[self.region] == state, self.mob_feature]
                self.state_df.loc[self.state_df[self.region] == state, self.mob_feature] = (arr-np.nanmin(arr))/(np.nanmax(arr) - np.nanmin(arr))
            #shift by 7 days and average
            self.state_df['mob'] = self.state_df[self.mob_feature].rolling(7).mean().shift(7)

        #fiting is done here and best parameters are saved into the model

        self.region_best_parameters = cassandra_fit(
            df = self.state_df,
            list_states = regions,
            start_date = self.startdates,
            memory = self.memory ,
            split_date = self.split_date,
            day_0 = self.split_date,
            day1= self.day1,
            real_GR = self.real_GR,
            deterministic = self.deterministic,
            r = self.r, date_col=self.date,
            region_col=self.region,
            target_col=self.target,
            use_mob = self.use_mob,
            starting_point = self.starting_point,
            hfilter = self.hfilter,
            clusters_map= self.clusters_map,
            active_cases = self.active_cases,
            extra_features= self.extra_features)

    
    
    def predict(self, regions, dates):

        forward_days = days_between(max(dates), self.split_date)   #split date is the first test day (we need a +1)

        # one line prediction since every parameter has been incorporated into the fitting
        df_simple, df_with_growth_rates = cassandra_predict(
            df = add_one_fake_day_forward(self.state_df , date = self.date, region = self.region),
            region_best_parameters=self.region_best_parameters,  
            list_states = regions, 
            start_date = self.startdates, 
            memory = self.memory,
            forward_days = forward_days, 
            split_date = mod_date(self.split_date,1),
            day_0 = mod_date(self.split_date,1),
            day1= self.day1, 
            real_GR = self.real_GR, 
            deterministic = self.deterministic, 
            r = self.r, 
            date_col=self.date,
            region_col=self.region, 
            target_col=self.target, 
            use_mob = self.use_mob, 
            starting_point = self.starting_point, 
            hfilter = self.hfilter,
            clusters_map= self.clusters_map,
            active_cases = self.active_cases,
            extra_features = self.extra_features)

        
        
        df_simple[self.date] = df_simple[self.date].apply(lambda x: datetime.strptime(x[:10], '%Y-%m-%d'))
        out = dict()
        for state1 in regions:
            out[state1] = dict()
            for date1 in dates:
                df = df_simple[[a and b for a, b in zip(df_simple[self.region]==state1, df_simple[self.date] == date1)]]
                pred= (df['pred_' + self.target]).to_string(index=False)
                out[state1][date1] = pred
        output = {region: pd.DataFrame([float(x) if str(x).find('S') < 0 else x for x in out[region].values()], index=out[region].keys())[0][pd.DataFrame(out[region].values(), index=out[region].keys())[0] != 'Series([], )'] for region in out.keys()}
        filter_regions = deepcopy(list(output.keys()))
        for region in filter_regions:
            if len(output[region]) == 0:
                del output[region]
        return output
        # return df_simple
