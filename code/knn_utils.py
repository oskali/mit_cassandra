# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 20:15:47 2020
Last update including clusters & tunning per region 25/10/2020
@authors: Yiannis, Bibha, Margaret
"""

#%% Libraries
import time
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.neighbors import KNeighborsRegressor
from random import choices
import matplotlib.pyplot as plt
import pickle


#%% Helper Functions

def wmape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred).astype('float')
    return sum((np.abs(y_true - y_pred)) * 100) / sum(y_true)

def get_in_date_range(dataset, first_date = '2020-01-01', last_date = '2020-12-31', date_col='date'):
    return dataset.loc[(dataset[date_col].astype('datetime64') >= np.datetime64(first_date)) & (dataset[date_col].astype('datetime64') < np.datetime64(last_date))]

def mod_date(date, interval):
    return str(np.datetime64(date) + np.timedelta64(interval, 'D'))

def days_between(d1, d2):
    d1=np.array(d1,dtype='datetime64[D]') 
    d2=np.array(d2,dtype='datetime64[D]') 
    return abs((d2-d1).item().days)

def get_weights(arr, threshold = 100):
    def weight_func(row, threshold = 100):
        min_dist = min(row)
        if min_dist == 0:
            return [1/len(row) for elt in row]
        filt_arr = []
        count = 0
        for dist in row:
            if dist <= threshold*min_dist:
                filt_arr.append(1/dist)
            else:
                filt_arr.append(-1)
                count += 1
        denom = sum(filt_arr) + count
        final_arr = []
        for dist in filt_arr:
            if dist == -1:
                final_arr.append(0)
            else:
                final_arr.append(dist/denom)
        return final_arr
    return np.apply_along_axis(lambda x: weight_func(x, threshold), 1, arr)


def get_weights_sq(arr, benchmark = 100):
    def weight_func_sq(row, benchmark = 100):
        sq_row = np.apply_along_axis(lambda x: x**2, axis= 0, arr = row)
        min_dist = min(sq_row)
        if min_dist == 0:
            return [1/len(sq_row) for elt in sq_row]
        filt_arr = []
        count = 0
        for dist in sq_row:
            if dist <= benchmark*min_dist:
                filt_arr.append(1/dist)
            else:
                filt_arr.append(-1)
                count += 1
        denom = sum(filt_arr) + count
        final_arr = []
        for dist in filt_arr:
            if dist == -1:
                final_arr.append(0)
            else:
                final_arr.append(dist/denom)
        return final_arr
    return np.apply_along_axis(lambda x: weight_func_sq(x, benchmark), 1, arr)



def get_matching_indexes(start_index, val_indexes, mob_indexes):
    n = len(val_indexes)
    m = len(mob_indexes)
    mob_index_dict = {mob_indexes[i]:i for i in range(m)}
    keep_vals = []

    #put first match in keep_vals

    start_left = val_indexes[start_index]
    keep_vals.append(start_left)


    # borders are the points above/below points we can still use (new match cannot include the border)
    top_left_border = start_index # last index examined on the top left
    bottom_left_border = start_index # last index examined on the top right
    top_right_border = mob_index_dict[start_left] #last index used on the top right
    bottom_right_border = mob_index_dict[start_left] #last index used on the bottom right
    
    #put all top matches in keep_vals until no more values on the top left to add or no more value or the top right to be matched.
    while top_left_border > 0 and top_right_border > 0:
        next_left = val_indexes[top_left_border - 1]
        top_right_index = mob_index_dict[next_left]
        if top_right_index < top_right_border:
            keep_vals.append(next_left)
            top_right_border = top_right_index   
        top_left_border -= 1

    #put all bottom matches in keep_vals until no more values on the bottom left to add or no more value or the bottom right to be matched
    while bottom_left_border < n-1 and bottom_right_border < m-1:
        next_left = val_indexes[bottom_left_border + 1]
        bottom_right_index = mob_index_dict[next_left]
        if bottom_right_index > bottom_right_border:
            keep_vals.append(next_left)
            bottom_right_border = bottom_right_index   
        bottom_left_border += 1

    return keep_vals


def rename_features(i, features, memory):
    dictionary = {}
    for j in range(memory):
        dictionary[features[j]] = 'GrowthRate_t-' +str(memory-j)
    return dictionary

def transpose_case_df(simple_output, forward_days, day_0, date_col='date', region_col='county', target_col='cases'):

    dates = []
    cases = []
    low_cases = []
    high_cases = []
    states = []
    for i in range(forward_days):
        date = mod_date(day_0, i)

        dates.extend([date for i in range(len(simple_output))])
        cases.extend(simple_output[target_col+'_predicted_day_' + str(i)])
        low_cases.extend(simple_output[target_col+'_low_predicted_day_' + str(i)])
        high_cases.extend(simple_output[target_col+'_high_predicted_day_' + str(i)])
        states.extend(simple_output[region_col])

    df = pd.DataFrame({date_col:dates, region_col: states, 'pred_'+ target_col:cases})#, 'pred_cases_low':low_cases, 'pred_cases_high': high_cases})
    return df

def convert_active_to_total(df, active_df, forward_days, region_col = 'state',date_col='date',target_col='two_week_cases', old_target_col = 'cases'):
    total_df = pd.DataFrame()
    df_merge = df.merge(active_df, on = [region_col, date_col], how = 'outer')
    df_merge = df_merge.sort_values([region_col,date_col])
    df_merge['cases_back_2_weeks'] = df_merge[old_target_col].shift(14)
    df_merge = df_merge.loc[df_merge['pred_'+target_col].notna()]

    for region in active_df[region_col].unique():
        region_df = df_merge.loc[df_merge[region_col] == region].reset_index()
        for i in np.arange(forward_days, 0, -1):
            region_df.loc[len(region_df)-i, 'pred_'+old_target_col] = region_df.loc[len(region_df)-i, 'pred_'+target_col]+ region_df.loc[len(region_df)-i, 'cases_back_2_weeks']
            if i > 14:
                region_df.loc[len(region_df)-i + 14, 'cases_back_2_weeks'] = region_df.loc[len(region_df)-i, 'pred_'+old_target_col]
        total_df = pd.concat([total_df, region_df[[date_col,region_col,'pred_'+old_target_col]]])
    return total_df

def get_best_parameters(df, chosen_region , memory, split_date, forward_days, r, day1, region_col = 'state',
                        date_col='date', mob_col = 'mob', use_mob = False, starting_point = 'high',
                        hfilter = False , clusters_map = {}, extra_features = [] ):
        # if clusters_map is not empty, we isolate the states that belong to the same cluster to train only to those
        if clusters_map:
            df=df.loc[df[region_col].isin(clusters_map[chosen_region])]

        N = 0 #default is no external features
        I = 1 #deafault is one random generation, no external features
        if extra_features:
            N = len(extra_features)  # number of external features
            I = 25 # number of random sampling of weights

        np.random.seed(42) #random seed for reproducability
        random_weight = 5 * np.random.random((I, N))  # Empty if no external features ## the 5 is random

        output = []
        features = ['GrowthRate_t-'+ str(i+1) for i in range(memory)] + extra_features
        for sample_iteration in range(I):
            for threshold in [100]: ##1.1, 1.5, 2, 5, 10, 20, 50,
                for n in [1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 100]:
                    for p in [1,2]:
                    # once we determine first date for each state, we will modify the March 22 hard coding
                        #threshold= threshold+4
                        train = get_in_date_range(df, first_date = day1, last_date = mod_date(split_date, -r), date_col=date_col)
                        test = get_in_date_range(df, first_date = mod_date(split_date, -r), last_date = split_date, date_col=date_col)

                        if extra_features:
                            #multiply each external feature with the random weight
                            for ext in range(N):
                                train[extra_features[ext]] = train[extra_features[ext]]* random_weight[sample_iteration][ext]
                                test[extra_features[ext]] = test[extra_features[ext]] *  random_weight[sample_iteration][ext]

                        test = test.loc[test[region_col] == chosen_region] # running prediction only for one state

                        train = train[~train[features].isna().any(axis=1)] #added this line due to an error in counties

                        x_train = train[features]
                        #np.isnan(x_train.values.any())
                        y_train = train[~train[features].isna().any(axis=1)].GrowthRate
                        x_test = test[features]
                        y_test = test.GrowthRate
                        if test.shape[0] != 0 and train.shape[0] != 0:
                            if n <= train.shape[0]:
                                # we create a model using each weight method, one for non-squared and one for squared
                                model = KNeighborsRegressor(n_neighbors=n, weights = lambda x: get_weights(x, threshold),p = p)
                                modelsq = KNeighborsRegressor(n_neighbors=n, weights = lambda x: get_weights_sq(x, threshold),p = p)


                                model.fit(x_train, y_train) #fit is not actually training anything, just assigns each y value to corresponding x vector
                                modelsq.fit(x_train, y_train)

                                preds = {}
                                if use_mob:
                                    for m in [model, modelsq]:
                                        predictions = []
                                        for i in range(x_test.shape[0]):
                                            point = x_test.iloc[[i]]
                                            distances, indexes = m.kneighbors(point)
                                            weights = m.weights(distances)[0]

                                            ## start mobility stuff
                                            ## Used same as before where we assume each test set is 1 day -- may need to change for multiple test days

                                            weights_not_zero = np.ma.masked_not_equal(weights, 0.0, copy=False).mask
                                            mobs = train.iloc[indexes[0]][mob_col][weights_not_zero]
                                            values = train.iloc[indexes[0]]['GrowthRate'][weights_not_zero]


                                            mob_indexes = list(mobs.argsort())
                                            val_indexes = list(values.argsort())

                                            if hfilter:
                                                hmatch = mob_indexes[len(mob_indexes) - 1]
                                                val_indexes = val_indexes[:val_indexes.index(hmatch) + 1]

                                            #Change start_index to change where we start on the left side of matching

                                            if starting_point == 'high':
                                                start_index = len(val_indexes) - 1
                                            elif starting_point == 'low':
                                                start_index = 0
                                            elif starting_point =='mid':
                                                start_index = round( len(val_indexes)/2 )


                                            matched_indexes = get_matching_indexes(start_index, val_indexes, mob_indexes)


                                            selected_weights = np.array(weights)[matched_indexes]
                                            selected_values = np.array(values)[matched_indexes]
                                            # selected_mobs = np.array(mobs)[matched_indexes]


                                            ## end mobility stuff

                                            # values is the GrowthRate of the n nearest neighbors
                                            scaled_weights = selected_weights * 1/sum(selected_weights)
                                            predictions.append(np.dot(selected_values, scaled_weights))
                                        preds[m] = predictions

                                    pred = preds[model]
                                    pred_sq = preds[modelsq]
                                else:
                                    pred = model.predict(x_test)
                                    pred_sq = modelsq.predict(x_test)

                                output.append( [[threshold,n,p,get_weights, list(random_weight[sample_iteration]) ] , wmape(y_test, pred)])
                                output.append( [[threshold,n,p,get_weights_sq, list(random_weight[sample_iteration]) ] , wmape(y_test, pred_sq)])
        return output[np.argmin([x[1] for x in output])][0] #returning the lowest wmape parameters

def match_to_real_growth(df, region_best_parameters, list_states, start_date, memory, forward_days, day_0, day1,
                         split_date, deterministic, region_col='county', date_col='date', mob_col = 'mob',
                         use_mob = False, starting_point = 'high', hfilter = False, clusters_map= {},
                         extra_features= []):
    #creates a list that we will use with a rolling window. e.g. to predict i=2 (2 days ahead) we have features [GrowthRate_t-5, GrowthRate_t-4,... GrowthRate_t-1, pred_forward_day_0, pred_forward_day_1]
    feature_choices = ['GrowthRate_t-' + str(i) for i in [memory-i for i in range(memory)]] + ['pred_forward_day_' + str(i) for i in range(forward_days)]

    # on first iteration (i=0) previous_final_test is the original df, on future iterations (i>0) it contains the predictions for t+0 through t+i-1
    previous_final_test = df
    for i in range(forward_days):
        #ADDING external features to both features and real_features
        features = feature_choices[i:i+memory] + extra_features #adding the extra features as features
        real_features = ['GrowthRate_t-' + str(j+1) for j in range(memory)] + extra_features #adding the extra features as features

        # current_final_test is the df where we add all the state predictions for the current iteration (day)
        current_final_test = pd.DataFrame()
        for chosen_region in list_states:
            # the distinction between in state and out of state only has an effect when the day_0 is before the split_date
            #for in state train data, we can use anything before the day_0
            train_data_in_state = get_in_date_range(df.loc[df[region_col] == chosen_region], first_date=start_date[chosen_region], last_date = day_0, date_col = date_col)
            # for out of state train data, we can use anything before the split_date'
            train_data_out_of_state = (get_in_date_range(df.loc[df[region_col] != chosen_region], first_date= day1 , last_date = split_date, date_col=date_col))
            train = pd.concat([train_data_in_state, train_data_out_of_state], sort = False)
            #print(len(train))

            # if clusters_map is not empty, we isolate the states that belong to the same cluster to train only to those
            if clusters_map:
                train=train.loc[train[region_col].isin(clusters_map[chosen_region])]


            # in the train rows, we use the growthrates of t-1 to t-memory to match nearest neighbors to the test row
            x_train = train[real_features]

            for ex in range(len(extra_features)):
                # for each external feature we multiply it with its optimal weight given by best_parameters
                x_train[extra_features[ex]] = x_train[extra_features[ex]]* region_best_parameters[chosen_region]['ex_weights'][ex]


            y_train = train['GrowthRate']

            test_df = previous_final_test.loc[previous_final_test[region_col] == chosen_region]
            test0 = get_in_date_range(test_df, first_date=mod_date(day_0,0), last_date=mod_date(day_0, 1), date_col=date_col)

            #we create a copy in which we will modify feature names (which include some 'pred_forward_day_x' features) to match the real_features from the train rows (all 'GrowthRate_t-x')
            test = test0.copy(deep = True)

            x_test = test[features]
            #rename_features maps 7 days before the current iteration day to GrowthRate_t-7, 7 days before to GrowthRate_t-6, etc.
            # The renaming doesnt affect the external features
            x_test = x_test.rename(columns = rename_features(i, features, memory))
            for ex in range(len(extra_features)):
                # for each external feature we multiply it with its optimal weight given by best_parameters
                x_test[extra_features[ex]] = x_test[extra_features[ex]]* region_best_parameters[chosen_region]['ex_weights'][ex]


            '''
            Getting the best parameters for that particular state/region
            '''
            n = region_best_parameters[chosen_region]['neighbors']
            threshold = region_best_parameters[chosen_region]['threshold']
            p = region_best_parameters[chosen_region]['p_norm']
            func = region_best_parameters[chosen_region]['function']
            
            if len(x_train)<n:
                nn = KNeighborsRegressor(n_neighbors=len(x_train), weights = lambda x: func(x, threshold), p = p)
            else:
                nn = KNeighborsRegressor(n_neighbors=n, weights = lambda x: func(x, threshold),p = p)

            nn.fit(x_train, y_train)
            

            distances, indexes = nn.kneighbors(x_test)
            weights = func(distances)[0]
            weights_not_zero = np.ma.masked_not_equal(weights, 0.0, copy=False).mask
            
            values = train.iloc[indexes[0]]['GrowthRate'][weights_not_zero]

            if use_mob:
                mobs = train.iloc[indexes[0]][mob_col][weights_not_zero]
                
                mob_indexes = list(mobs.argsort())
                val_indexes = list(values.argsort())
                
                if hfilter:
                    hmatch = mob_indexes[len(mob_indexes) - 1]
                    val_indexes = val_indexes[:val_indexes.index(hmatch) + 1]
                # print('GR:',val_indexes, 'mobility:', mob_indexes,'\n')

                #Change start_index to change where we start on the left side of matching
                if starting_point == 'high':
                    start_index = len(val_indexes) - 1
                elif starting_point == 'low':
                    start_index = 0
                elif starting_point =='mid':
                    start_index = round( len(val_indexes)/2 )

                matched_indexes = get_matching_indexes(start_index, val_indexes, mob_indexes)
                
                # print(matched_indexes)

                selected_weights = np.array(weights)[matched_indexes]
                selected_values = np.array(values)[matched_indexes]
                # selected_mobs = np.array(mobs)[matched_indexes]
      
            # values is the GrowthRate of the n nearest neighbors
            # values = np.array(y_train.iloc[indexes[0]])


            # weights_not_zero = np.ma.masked_not_equal(weights, 0.0, copy=False).mask
            # valid_values = values[weights_not_zero]
            

            test0['pred_high_day_'+str(i)] = max(values)
            test0['pred_low_day_'+str(i)] = min(values)

            if deterministic:
                if use_mob:               
                    scaled_weights = selected_weights * 1/sum(selected_weights) # scale remaining weights so they add to 1
                    y_pred = np.dot(scaled_weights, selected_values)
                else:
                    y_pred = nn.predict(x_test)
 
            else:
                y_pred = choices(values, weights)

            test0['pred_forward_day_'+str(i)] = y_pred # add the new prediction as a new column
            #pred_high_day_i and pred_low_day_i
#             x_test = x_test.rename(columns = undo_rename(features)) # make sure that original column names are not changed when they are changed in the copy

            current_final_test = pd.concat([current_final_test, test0], sort = False)
        previous_final_test = current_final_test

    # after the final iteration, previous_final_test contains predictions for the next forward_days starting from day_0
    return previous_final_test

def cassandra_fit(df, list_states, start_date, memory = 10, forward_days = 14, split_date = '2020-05-01', day_0 = '2020-05-01',
                  day1='2020-04-10', real_GR = False, deterministic = True, r = 1, date_col='date', region_col='state',
                  target_col='cases', mob_col = 'mob', use_mob = False, starting_point = 'high', hfilter = False ,
                  clusters_map= {}, active_cases = False , extra_features = []):
    '''
    everything between start date and split_date is train
    '''
    #This section of code creates the forward and back features
    #remove some states/territories with late or small number of cases
    #CHANGE TO KEEPING STATES - hard copy in this code maybe global variable
    df = df.loc[~df[region_col].isin(['District of Columbia','Puerto Rico','American Samoa', 'Diamond Princess','Grand Princess','Guam','Northern Mariana Islands','Virgin Islands'])]
    df = df.sort_values(by=[region_col, date_col])#has to be sorted by days to create growth rates
    
    if active_cases:
        df['two_week_'+target_col] = df[target_col] - df[target_col].shift(14)
        target_col = 'two_week_'+target_col

    df['GrowthRate'] = df.groupby(region_col)[target_col].shift(0) / (df[target_col].shift(1)) - 1 #group by state so that consecutive rows are consecutive days in a single state
    #create the t-1 to t-memory growth rates
    for i in range(memory):
        df['GrowthRate_t-' + str(i+1)] = df.groupby(region_col)['GrowthRate'].shift(i+1)

    df[target_col+'_t-1'] = df[target_col].shift(1)

    #this is used only if we are using the alternate method where we run nearest neighbors on predictions in the train set
    if real_GR:
        for i in range(forward_days):
            df['GrowthRate_t+' + str(i)] = df.groupby(region_col)['GrowthRate'].shift(-i)

        for i in range(forward_days):
            df['actual_growth_for_next_{}days'.format(i+1)] = (df[target_col].shift(-i)/df[target_col].shift(1)) - 1
    '''
    threshold: multiplier on the nearest distance that we cut off at when assigning weights, e.g. a point outside the threshold gets a weight of 0
    n: maximum number of nearest neighbors
    p: either L1 norm (manhattan) or L2 norm
    func: get_weights or get_weights_sq, whether the distance norm will be squared or not
    coeffs: the weights of the extra_features if any.
    '''
    
    region_best_parameters = {}
    for region in list_states:
        threshold, n, p, func , ex_weights = get_best_parameters(df, region ,  memory, split_date, forward_days, r, day1,
                                                    region_col= region_col, date_col=date_col, mob_col = mob_col,
                                                    use_mob = use_mob, starting_point = starting_point,
                                                    hfilter = hfilter , clusters_map= clusters_map,
                                                    extra_features= extra_features)
        region_best_parameters[region] = {'threshold': threshold , 'neighbors': n , 'p_norm':p,
                                          'function': func, 'ex_weights' : ex_weights}

    return region_best_parameters


def cassandra_predict(df, region_best_parameters, list_states, start_date, memory = 10, forward_days = 14,
                      split_date = '2020-05-01', day_0 = '2020-05-01', day1='2020-04-10', real_GR = False,
                      deterministic = True, r = 1, date_col='date', region_col='state', target_col='cases',
                      mob_col = 'mob', use_mob = False, starting_point = 'high', hfilter = False,
                      clusters_map= {}, active_cases = False, extra_features = []):
    
    #This section of code creates the forward and back features
    #remove some states/territories with late or small number of cases
    #CHANGE TO KEEPING STATES - hard copy in this code maybe global variable
    df = df.loc[~df[region_col].isin(['District of Columbia','Puerto Rico','American Samoa', 'Diamond Princess','Grand Princess','Guam','Northern Mariana Islands','Virgin Islands'])]
    df = df.sort_values(by=[region_col, date_col])#has to be sorted by days to create growth rates
    
    if active_cases:
        df['two_week_'+target_col] = df[target_col] - df[target_col].shift(14)
        old_target_col = target_col
        target_col = 'two_week_'+target_col

    df['GrowthRate'] = df.groupby(region_col)[target_col].shift(0) / (df[target_col].shift(1)) - 1 #group by state so that consecutive rows are consecutive days in a single state
    #create the t-1 to t-memory growth rates
    for i in range(memory):
        df['GrowthRate_t-' + str(i+1)] = df.groupby(region_col)['GrowthRate'].shift(i+1)

    df[target_col+'_t-1'] = df[target_col].shift(1)

    #this is used only if we are using the alternate method where we run nearest neighbors on predictions in the train set
    if real_GR:
        for i in range(forward_days):
            df['GrowthRate_t+' + str(i)] = df.groupby(region_col)['GrowthRate'].shift(-i)

        for i in range(forward_days):
            df['actual_growth_for_next_{}days'.format(i+1)] = (df[target_col].shift(-i)/df[target_col].shift(1)) - 1
    
    # df.to_csv('growthrate_check.csv', index = False)
    '''
    All this function can be simplified in just running the match_to_real_growth with region_best_parameters
    Everything that was done before and after is just feature engineering to calculate GR and create the case output
    '''    
    predictions = match_to_real_growth(df, region_best_parameters, list_states, start_date, memory, forward_days,
                                       day_0, split_date, day1, deterministic, region_col=region_col,
                                       mob_col = mob_col, use_mob = use_mob, starting_point = starting_point,
                                       hfilter = hfilter, extra_features = extra_features)

    #we have finished producing predictions, and move on to converting predicted growth rates into predicted cases

    #convert growth rates to cumulative growth rates -- here we need to add 1 to each predicted growth rate so that when multiplied they represent growth rate over multiple days
    #the cumulative growth rate over n days starting today = (1+ GR_0) * (1+GR_1) * ... * (1+ GR_n-1)
    predictions['pred_growth_for_next_1days'] = predictions['pred_forward_day_0'] + 1
    predictions['pred_high_growth_for_next_1days'] = predictions['pred_high_day_0'] + 1
    predictions['pred_low_growth_for_next_1days'] = predictions['pred_low_day_0'] + 1

    for i in range(1,forward_days):
        predictions['pred_growth_for_next_{}days'.format(i+1)] = predictions['pred_growth_for_next_{}days'.format(i)]*(predictions['pred_forward_day_'+ str(i)] + 1)
        predictions['pred_high_growth_for_next_{}days'.format(i+1)] = predictions['pred_growth_for_next_{}days'.format(i)]*(predictions['pred_high_day_'+ str(i)] + 1)
        predictions['pred_low_growth_for_next_{}days'.format(i+1)] = predictions['pred_growth_for_next_{}days'.format(i)]*(predictions['pred_low_day_'+ str(i)] + 1)
    for i in range(forward_days):
        predictions['pred_growth_for_next_{}days'.format(i+1)] = predictions['pred_growth_for_next_{}days'.format(i+1)] - 1
        predictions['pred_high_growth_for_next_{}days'.format(i+1)] = predictions['pred_high_growth_for_next_{}days'.format(i+1)] - 1
        predictions['pred_low_growth_for_next_{}days'.format(i+1)] = predictions['pred_low_growth_for_next_{}days'.format(i+1)] - 1

    #convert cumulative growth rates to cases
    for i in range(forward_days):
        predictions[target_col + '_predicted_day_' + str(i)] = np.round(predictions[target_col+'_t-1']*(predictions['pred_growth_for_next_{}days'.format(i+1)]+1))
        predictions[target_col+'_high_predicted_day_' + str(i)] = np.round(predictions[target_col+'_t-1']*(predictions['pred_high_growth_for_next_{}days'.format(i+1)]+1))
        predictions[target_col+'_low_predicted_day_' + str(i)] = np.round(predictions[target_col+'_t-1']*(predictions['pred_low_growth_for_next_{}days'.format(i+1)]+1))

    columns_to_keep = [region_col, date_col, target_col] + [target_col+'_predicted_day_' + str(i) for i in range(forward_days)] + [target_col+'_low_predicted_day_' + str(i) for i in range(forward_days)] + [target_col+'_high_predicted_day_' + str(i) for i in range(forward_days)]
    simple_output = predictions[columns_to_keep]

    #transpose simple output to have forward_days*50 rows
    transposed_simple_output = transpose_case_df(simple_output, forward_days, day_0, date_col, region_col, target_col)

    if active_cases:
        transposed_simple_output = convert_active_to_total(df, forward_days= forward_days ,active_df = transposed_simple_output, region_col = region_col, date_col = date_col, target_col = target_col, old_target_col = old_target_col)

    return transposed_simple_output, predictions