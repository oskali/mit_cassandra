from datetime import datetime
import numpy as np
import pandas as pd
from bilstm_utils import find_best_model_covid,  fit_covid, predict_covid
from copy import deepcopy
from tensorflow.keras.models import load_model
import os

#%% Model

class BILSTMModel():

    def __init__(self,
                 date='date',
                 region='state',
                 target='cases',
                 deterministic=True,
                 dates=None):
        self.date = date
        self.region = region
        self.target = target
        self.deterministic = deterministic
        self.dates = dates
        self.state_df = None

        self.startdates = None

    def fit(self,
            df):
        # print("FIT")
        self.state_df = df
        
        split_date = max(self.state_df[self.date])  #this needs to change to the max(self.state_df.date.unique()) +1
        # split_date = (datetime.strptime(ll, '%Y-%m-%d')).strftime('%Y-%m-%d')
        forward_days = (max(self.dates)-split_date).days + 1

        # print(self.dates)
        # print(split_date)
        #split_date = split_date.strftime("%Y-%m-%d")
        day_0 = split_date
        memory = 10

        state_df = self.state_df.copy()

        try:
            state_df = state_df.loc[~state_df["state"].isin(['District of Columbia', 'Puerto Rico','American Samoa', 'Diamond Princess','Grand Princess','Guam','Northern Mariana Islands','Virgin Islands'])]
        except:
            pass

        state_df = state_df[[self.region, self.date,self.target]].groupby([self.region, self.date]).sum().reset_index()

        state_df = state_df.sort_values(by = [self.region, self.date])
        is_100 = state_df[self.target] >= 100
        state_df1 = state_df[is_100]
        array = np.arange(len(state_df[self.date]))
        state_df.reindex(array)

        final_df = state_df1
        m0=0
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
                startdates[state2] = numdat[m[i]+8]
        deterministic = self.deterministic

        self.startdates = startdates

        state_df=final_df.copy(deep=True)
        state_df.reset_index(drop=True, inplace=True)

        # fix dataset
        for index, _ in state_df.iterrows():

            if (index < state_df.shape[0]-1):
                
                if (state_df.loc[index,self.region] == state_df.loc[index+1, self.region] ):

                    if (state_df.loc[index,self.target] > state_df.loc[index+1,self.target]):
                        
                        state_df.loc[index+1,self.target] = state_df.loc[index,self.target]

        state_df = state_df[~state_df.isin([np.nan, np.inf, -np.inf]).any(1)].copy(deep=True)

        df0 = state_df.copy(deep=True) # deep copy might not be needed, just for security
         
        df0 = df0.sort_values(by=[self.region, self.date])#has to be sorted by days to create growth rates
        #is_100 = df0['cases']>=100
        #df0 = df0[is_100]


        df0['GrowthRate'] = (df0.groupby(self.region)[self.target].shift(0) / df0[self.target].shift(1) - 1) #group by state so that consecutive rows are consecutive days in a single state

        #create the t-1 to t-memory growth rates
        for i in range(memory):
            df0['GrowthRate_t-' + str(i+1)] = df0.groupby(self.region)['GrowthRate'].shift(i+1)
            
        df0[self.target+'_t-1'] = df0[self.target].shift(1)

        
        for i in range(forward_days):
            df0['GrowthRate_t+' + str(i)] = df0.groupby(self.region)['GrowthRate'].shift(-i)
        
        self.state_df = df0.copy()

        (model_winner,mode_winner,epochs_winner) = find_best_model_covid(df = self.state_df, start_date = startdates, 
                                                        memory = 10, forward_days = forward_days, 
                                                        split_date = split_date, day_0 = day_0, 
                                                        real_GR = True, deterministic = deterministic,
                                                        r = min(30,forward_days),  date_col=self.date,
                                                        region_col=self.region,
                                                        target_col=self.target)

        _ = fit_covid(df = self.state_df, start_date = startdates, 
                            model_winner = model_winner, mode_winner = mode_winner,epochs_winner = epochs_winner,
                            memory = 10, forward_days = forward_days,
                            day_0 = day_0, split_date = split_date,
                            deterministic = deterministic,
                            region_col=self.region, date_col=self.date)

        return None

    def predict(self, regions, dates):

        #self.model = load_model("./models/bidir_lstm.")
        dir_path = os.path.dirname(os.path.abspath(__file__))
        self.model = load_model(os.path.join(dir_path, "models/bidir_lstm.h5"))
        # print("PREDICT")
        split_date = max(self.state_df[self.date])
        day_0 = split_date
        forward_days = (max(dates)-split_date).days + 1
        # print("Split Date and Day_0 for the given code is", split_date, day_0)
        df_simple, df_with_growth_rates = predict_covid(df = self.state_df, start_date = self.startdates, 
                                                        model = self.model,
                                                        memory = 10, forward_days = forward_days, 
                                                        split_date = split_date, day_0 = day_0, 
                                                        real_GR = True, deterministic = self.deterministic,
                                                        r = min(30,forward_days),  date_col=self.date,
                                                        region_col=self.region,
                                                        target_col=self.target)
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