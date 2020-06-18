# -*- coding: utf-8 -*-
"""
Created on Fri May 29 18:14:33 2020

@author: omars
"""
#############################################################################
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from random import choices
#############################################################################


#############################################################################
def wmape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred).astype('float')
    return sum((np.abs(y_true - y_pred)) * 100) / sum(y_true)

def get_in_date_range(dataset, first_date = '2020-01-01', last_date = '2020-12-31'):
    return dataset.loc[(dataset.date.astype('datetime64') >= np.datetime64(first_date)) & (dataset.date.astype('datetime64') < np.datetime64(last_date))]

def mod_date(date, interval):
    return str(np.datetime64(date) + np.timedelta64(interval, 'D'))

def get_weights(arr, threshold = 100):
    return np.apply_along_axis(lambda x: weight_func(x, threshold), 1, arr)

def weight_func(row, threshold = 100):
    min_dist = min(row)
    if (min_dist == 0):
        return [1/len(row) for elt in row]
    filt_arr = []
    count = 0
    for dist in row:
        if (dist <= threshold*min_dist):
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

def get_weights_sq(arr, benchmark = 100):
    return np.apply_along_axis(lambda x: weight_func_sq(x, benchmark), 1, arr)

def weight_func_sq(row, benchmark = 100):
    sq_row = np.apply_along_axis(lambda x: x**2, axis= 0, arr = row)
    min_dist = min(sq_row)
    if (min_dist == 0):
        return [1/len(sq_row) for elt in sq_row]
    filt_arr = []
    count = 0
    for dist in sq_row:
        if (dist <= benchmark*min_dist):
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


def rename_features(i, features, memory):
    dictionary = {}
    for j in range(memory):
        dictionary[features[j]] = 'GrowthRate_t-' +str(memory-j)
    return dictionary

def transpose_case_df(simple_output, forward_days, day_0, target='cases'):

    dates = []
    cases = []
    low_cases = []
    high_cases = []
    states = []
    for i in range(forward_days):
        date = mod_date(day_0, i)

        dates.extend([date for i in range(len(simple_output))])
        cases.extend(simple_output[target+ '_predicted_day_' + str(i)])
        low_cases.extend(simple_output[target+'_low_predicted_day_' + str(i)])
        high_cases.extend(simple_output[target+'_high_predicted_day_' + str(i)])
        states.extend(simple_output['state'])

    df = pd.DataFrame({'date':dates,'state': states, 'pred_'+target:cases, 'pred_'+target+'_low':low_cases, 'pred_'+target+'_high': high_cases})
    return df

def get_best_parameters(df, memory, split_date):
    output = []
    features = ['GrowthRate_t-'+ str(i+1) for i in range(memory)]
    for threshold in [1.1, 1.5, 2, 5, 10, 20, 50, 100]:
        for n in [1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,30,40,50,60,70,80,90,100]:
            for p in [1,2]:
                # once we determine first date for each state, we will modify the March 22 hard coding
                train = get_in_date_range(df, first_date = '2020-03-22', last_date = mod_date(split_date, -1))
                test = get_in_date_range(df, first_date = mod_date(split_date, -1), last_date = split_date)
                test0 = test.copy() # maybe this is not needed
                x_train = train[features]
                y_train = train.GrowthRate
                x_test = test[features]
                y_test = test.GrowthRate
                if (test.shape[0] != 0 and train.shape[0] != 0):
                    if (n <= train.shape[0]):
                        # we create a model using each weight method, one for non-squared and one for squared
                        model = KNeighborsRegressor(n_neighbors=n, weights = lambda x: get_weights(x, threshold), p=p)
                        modelsq = KNeighborsRegressor(n_neighbors=n, weights = lambda x: get_weights_sq(x, threshold), p=p)
                        model.fit(x_train, y_train) #fit is not actually training anything, just assigns each y value to corresponding x vector
                        modelsq.fit(x_train, y_train)
                        pred = model.predict(x_test) #returns the sum of products of weights and y values for the n nearest neighbors
                        pred_sq = modelsq.predict(x_test)
                        output.append([[threshold,n,p,get_weights], wmape(y_test, pred)])
                        output.append([[threshold,n,p,get_weights_sq], wmape(y_test, pred_sq)])
    return output[np.argmin([x[1] for x in output])][0]

def match_to_real_growth(df, threshold, n, p, func, memory, forward_days, day_0, split_date, deterministic):
    #creates a list that we will use with a rolling window. e.g. to predict i=2 (2 days ahead) we have features [GrowthRate_t-5, GrowthRate_t-4,... GrowthRate_t-1, pred_forward_day_0, pred_forward_day_1]
    feature_choices = ['GrowthRate_t-' + str(i) for i in [memory-i for i in range(memory)]] + ['pred_forward_day_' + str(i) for i in range(forward_days)]

    # on first iteration (i=0) previous_final_test is the original df, on future iterations (i>0) it contains the predictions for t+0 through t+i-1
    previous_final_test = df
    for i in range(forward_days):

        features = feature_choices[i:i+7]
        real_features = ['GrowthRate_t-' + str(j+1) for j in range(memory)]

        # current_final_test is the df where we add all the state predictions for the current iteration (day)
        current_final_test = pd.DataFrame()
        for state in df.state.unique():
            # when we have a specific first_date for each state, we will update the hard coded march 22


            # the distinction between in state and out of state only has an effect when the day_0 is before the split_date
            #for in state train data, we can use anything before the day_0
            train_data_in_state = get_in_date_range(df.loc[df.state == state],first_date='2020-03-22', last_date = day_0)
            # for out of state train data, we can use anything before the split_date
            train_data_out_of_state = get_in_date_range(df.loc[df.state != state], first_date='2020-03-22', last_date = split_date)

            train = pd.concat([train_data_in_state, train_data_out_of_state], sort = False)

            # in the train rows, we use the growthrates of t-1 to t-memory to match nearest neighbors to the test row
            x_train = train[real_features]
            y_train = train['GrowthRate']

            test_df = previous_final_test.loc[previous_final_test.state == state]
            test0 = get_in_date_range(test_df, first_date=day_0, last_date=mod_date(day_0, 1))

            #we create a copy in which we will modify feature names (which include some 'pred_forward_day_x' features) to match the real_features from the train rows (all 'GrowthRate_t-x')
            test = test0.copy(deep = True)

            x_test = test[features]
            #rename_features maps 7 days before the current iteration day to GrowthRate_t-7, 7 days before to GrowthRate_t-6, etc.
            x_test = x_test.rename(columns = rename_features(i, features, memory))

            nn = KNeighborsRegressor(n_neighbors=n, weights = lambda x: func(x, threshold), p=p)
            nn.fit(x_train, y_train)
            distances, indexes = nn.kneighbors(x_test)
            weights = func(distances)[0]
            # values is the GrowthRate of the n nearest neighbors
            values = np.array(y_train.iloc[indexes[0]])

            weights_not_zero = np.ma.masked_not_equal(weights, 0.0, copy=False).mask
            valid_values = values[weights_not_zero]


            test0['pred_high_day_'+str(i)] = max(valid_values)
            test0['pred_low_day_'+str(i)] = min(valid_values)

            if deterministic:
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

def predict_covid(df, memory = 7, forward_days = 7, split_date = '2020-05-01', day_0 = '2020-05-01', real_GR = False, deterministic = True):
    '''
    everything before split_date is train

    '''
    #This section of code creates the forward and back features

    df0 = df.copy(deep=True) # deep copy might not be needed, just for security

    #remove some states/territories with late or small number of cases
    #CHANGE TO KEEPING STATES - hard copy in this code maybe global variable
    df0 = df0.loc[~df0['state'].isin(['West Virginia','District of Columbia','Puerto Rico','American Samoa', 'Diamond Princess','Grand Princess','Guam','Northern Mariana Islands','Virgin Islands'])]

    df0 = df0.sort_values(by=['state', 'date']) #has to be sorted by days to create growth rates
    df0['GrowthRate'] = (df0.groupby('state')[target].shift(0) / df0[target].shift(1) - 1) #group by state so that consecutive rows are consecutive days in a single state

    #create the t-1 to t-memory growth rates
    for i in range(memory):
        df0['GrowthRate_t-' + str(i+1)] = df0.groupby('state')['GrowthRate'].shift(i+1)

    df0[target+'_t-1'] = df0[target].shift(1)

    #this is used only if we are using the alternate method where we run nearest neighbors on predictions in the train set
    if real_GR:
        for i in range(forward_days):
            df0['GrowthRate_t+' + str(i)] = df0.groupby('state')['GrowthRate'].shift(-i)

        for i in range(forward_days):
            df0['actual_growth_for_next_{}days'.format(i+1)] = (df0[target].shift(-i)/df0[target].shift(1)) - 1
    '''
    threshold: multiplier on the nearest distance that we cut off at when assigning weights, e.g. a point outside the threshold gets a weight of 0
    n: maximum number of nearest neighbors
    p: either L1 norm (manhattan) or L2 norm
    func: get_weights or get_weights_sq, whether the distance norm will be squared or not
    '''
    df0 = df0.dropna()
    threshold, n, p, func = get_best_parameters(df0, memory, split_date)

    #this is to choose which method to use. If only using real growth matching, we do not need
#     if method == 'match_to_predictions':
#         predictions = match_to_predictions(df0, threshold, n, p, func, memory, forward_days, split_date, last_test_date)
#     else:
#         predictions = match_to_real_growth(df0, threshold, n, p, func, memory, forward_days, split_date, last_test_date)

    #we run the method using the best parameters according to the split date
    predictions = match_to_real_growth(df0, threshold, n, p, func, memory, forward_days, day_0, split_date, deterministic)

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
        predictions[target+'_predicted_day_' + str(i)] = np.round(predictions[target+'_t-1']*(predictions['pred_growth_for_next_{}days'.format(i+1)]+1))
        predictions[target+'_high_predicted_day_' + str(i)] = np.round(predictions[target+'_t-1']*(predictions['pred_high_growth_for_next_{}days'.format(i+1)]+1))
        predictions[target+'_low_predicted_day_' + str(i)] = np.round(predictions[target+'_t-1']*(predictions['pred_low_growth_for_next_{}days'.format(i+1)]+1))

    columns_to_keep = ['state', 'date', target] + [target+'_predicted_day_' + str(i) for i in range(forward_days)] + [target+'_low_predicted_day_' + str(i) for i in range(forward_days)] + [target+'_high_predicted_day_' + str(i) for i in range(forward_days)]
    simple_output = predictions[columns_to_keep]
    # print(simple_output.iloc[0])

    #transpose simple output to have forward_days*50 rows
    transposed_simple_output = transpose_case_df(simple_output, forward_days, day_0, target)


    return transposed_simple_output, predictions
#############################################################################