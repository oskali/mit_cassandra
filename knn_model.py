# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 01:55:57 2020

@authors: Yiannis, Bibha, omars
"""

# %% Libraries
from datetime import datetime
import numpy as np
import pandas as pd
from knn_utils import predict_covid
from copy import deepcopy

# %% Model


class KNNModel():

    def __init__(self,
                 date='date',
                 region='state',
                 target='cases',
                 deterministic=True):
        self.date = date
        self.region = region
        self.target = target
        self.deterministic = deterministic
        self.state_df = None

    def fit(self,
            df):
        self.state_df = df

    def predict(self, regions, dates):

        split_date = max(self.state_df[self.date])  # this needs to change to the max(self.state_df.date.unique()) +1
        # split_date = (datetime.strptime(ll, '%Y-%m-%d')).strftime('%Y-%m-%d')

        #split_date = split_date.strftime("%Y-%m-%d")
        day_0 = split_date
        forward_days = (max(dates)-split_date).days + 1

        state_df = self.state_df.copy()

        try:
            state_df = state_df.loc[~state_df['state'].isin(['West Virginia', 'District of Columbia', 'Puerto Rico',
                                                             'American Samoa', 'Diamond Princess', 'Grand Princess', 'Guam', 'Northern Mariana Islands', 'Virgin Islands'])]
        except:
            pass

        state_df = state_df[[self.region, self.date, self.target]].groupby([self.region, self.date]).sum().reset_index()

        state_df = state_df.sort_values(by=[self.region, self.date])
        is_100 = state_df[self.target] >= 100
        state_df1 = state_df[is_100]
        array = np.arange(len(state_df[self.date]))
        state_df.reindex(array)

        final_df = state_df1
        m0 = 0
        m = [0]
        for state2 in final_df[self.region].unique():
            for state1 in final_df[self.region]:
                if(state1 == state2):
                    m0 = m0+1
            m.append(m0)
        dat = final_df[self.date]
        numdat = dat.values
        startdates = dict()
        for state2 in final_df[self.region].unique():
            for i in range(len(m)-1):
                startdates[state2] = numdat[m[i]+15]
        deterministic = self.deterministic

        # print("Split Date and Day_0 for the given code is", split_date, day_0)

        df_simple, df_with_growth_rates = predict_covid(df=state_df, start_date=startdates, memory=7, forward_days=forward_days, split_date=split_date,
                                                        day_0=day_0, real_GR=True, deterministic=deterministic, r=1, date_col=self.date, region_col=self.region, target_col=self.target)
        df_simple[self.date] = df_simple[self.date].apply(lambda x: datetime.strptime(x[:10], '%Y-%m-%d'))

        out = dict()
        for state1 in regions:
            print(state1)
            out[state1] = dict()
            for date1 in dates:
                df = df_simple[[a and b for a, b in zip(
                    df_simple[self.region] == state1, df_simple[self.date] == date1)]]
                pred = (df['pred_' + self.target]).to_string(index=False)
                out[state1][date1] = pred
        output = {region: pd.DataFrame([float(x) if str(x).find('S') < 0 else x for x in out[region].values()], index=out[region].keys())[
            0][pd.DataFrame(out[region].values(), index=out[region].keys())[0] != 'Series([], )'] for region in out.keys()}
        filter_regions = deepcopy(list(output.keys()))
        for region in filter_regions:
            if len(output[region]) == 0:
                del output[region]
        return output
