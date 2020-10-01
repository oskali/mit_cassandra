
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 20:15:47 2020
@authors: Yiannis, Bibha, omars
"""

#%% Libraries
import pandas as pd
import numpy as np
from random import choices
import tensorflow as tf
from keras.layers import Layer
from keras.models import Sequential, load_model

from keras.losses import MSE
from keras.callbacks import EarlyStopping
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers import TimeDistributed
from keras.layers import Flatten
from keras.layers import Dense, Dropout

#%% Helper Functions

def wmape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred).astype('float')
    return sum((np.abs(y_true - y_pred)) * 100) / sum(y_true)

def get_in_date_range(dataset, first_date = '2020-01-01', last_date = '2020-12-31', date_col='date'):
    return dataset.loc[(dataset[date_col].astype('datetime64') >= np.datetime64(first_date)) & (dataset[date_col].astype('datetime64') < np.datetime64(last_date))]

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

def transpose_case_df(simple_output, forward_days, day_0, date_col='date', region_col='county', target_col='cases'):

    dates = []
    cases = []
    states = []
    for i in range(forward_days):
        date = mod_date(day_0, i)

        dates.extend([date for i in range(len(simple_output))])
        cases.extend(simple_output[target_col+'_predicted_day_' + str(i)])
        states.extend(simple_output[region_col])

    df = pd.DataFrame({date_col:dates, region_col: states, 'pred_'+ target_col:cases})#, 'pred_cases_low':low_cases, 'pred_cases_high': high_cases})
    return df

def get_best_parameters(df, memory, split_date, forward_days, r, col_date='date'):
    
    model_winner = ''
    mode_winner = ''
    epochs_winner = -1
    wmape_min = 10**10
    
    features = ['GrowthRate_t-'+ str(i+1) for i in range(memory)]

    for mode in ['concat','ave']:
        
        # once we determine first date for each state, we will modify the March 22 hard coding
        train = get_in_date_range(df, first_date = '2020-04-10', last_date = mod_date(split_date, -r), date_col=col_date)
        test = get_in_date_range(df, first_date = mod_date(split_date, -r), last_date = split_date, date_col=col_date)
        test0 = test.copy() # maybe this is not needed
        
        # remove nans and infs from each line of dataframes
        train = train[~train.isin([np.nan, np.inf, -np.inf]).any(1)]
        test = test[~test.isin([np.nan, np.inf, -np.inf]).any(1)]
        
        # create dataset
        x_train = train[features]
        y_train = train.GrowthRate
                
        x_test = test[features]
        y_test = test.GrowthRate
        
        # transform for lstm
        X_train = np.reshape(x_train.to_numpy(), (x_train.to_numpy().shape[0], x_train.to_numpy().shape[1], 1))
        X_test = np.reshape(x_test.to_numpy(), (x_test.to_numpy().shape[0], x_test.to_numpy().shape[1], 1))

        # set hyperparams
        act = 'relu'
        epochs = 100       
        patience = 3

        # model1
        model = Sequential()
        model.add(Bidirectional(LSTM(250, activation=act, return_sequences = False), merge_mode=mode, input_shape=(memory, 1)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse', metrics=['mse'])
        
        es = EarlyStopping(monitor='mse', verbose=1, patience = patience)
        model.fit(X_train, y_train.to_numpy(), validation_data=(X_test, y_test.to_numpy()), epochs=epochs, verbose=1,callbacks=[es])  
        preds = model.predict(X_test)
        
        if (wmape(y_test.to_numpy(),preds.flatten()) <wmape_min):
            wmape_min = wmape(y_test.to_numpy(),preds.flatten())
            model_winner = 'model_1'
            mode_winner = mode
            epochs_winner = es.stopped_epoch+patience
            
        # model2
        model = Sequential()
        model.add(Bidirectional(LSTM(250, activation=act, return_sequences = True), merge_mode=mode, input_shape=(memory, 1)))
        model.add(Bidirectional(LSTM(150, activation=act, return_sequences = False), merge_mode=mode))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse', metrics=['mse'])
        
        es = EarlyStopping(monitor='mse', verbose=1, patience = patience)
        model.fit(X_train, y_train.to_numpy(), validation_data=(X_test, y_test.to_numpy()), epochs=epochs, verbose=1,callbacks=[es])  
        preds = model.predict(X_test)
        
        if (wmape(y_test.to_numpy(),preds.flatten()) <wmape_min):
            wmape_min = wmape(y_test.to_numpy(),preds.flatten())
            model_winner = 'model_2'
            mode_winner = mode
            epochs_winner = es.stopped_epoch+patience
            
        # model3
        model = Sequential()
        model.add(Bidirectional(LSTM(250, activation=act, return_sequences = True), merge_mode=mode, input_shape=(memory, 1)))
        model.add(Bidirectional(LSTM(150, activation=act, return_sequences = True), merge_mode=mode))
        model.add(Bidirectional(LSTM(100, activation=act), merge_mode=mode))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse', metrics=['mse'])
        
        es = EarlyStopping(monitor='mse', verbose=1, patience = patience)
        model.fit(X_train, y_train.to_numpy(), validation_data=(X_test, y_test.to_numpy()), epochs=epochs, verbose=1,callbacks=[es])  
        preds = model.predict(X_test)
        
        if (wmape(y_test.to_numpy(),preds.flatten()) <wmape_min):
            wmape_min = wmape(y_test.to_numpy(),preds.flatten())
            model_winner = 'model_3'
            mode_winner = mode
            epochs_winner = es.stopped_epoch+patience
           
    return (model_winner,mode_winner,epochs_winner)

from tensorflow.keras.models import load_model
def match_to_real_growth(df, start_date, model_winner,mode_winner,epochs_winner, memory, forward_days, day_0, split_date, deterministic, region_col='county', date_col='date'):
    # creates a list that we will use with a rolling window. e.g. to predict i=2 (2 days ahead) we have features 
    # [GrowthRate_t-5, GrowthRate_t-4,... GrowthRate_t-1, pred_forward_day_0, pred_forward_day_1]
    feature_choices = ['GrowthRate_t-' + str(i) for i in [memory-i for i in range(memory)]] + ['pred_forward_day_' + str(i) for i in range(forward_days)]

    # on first iteration (i=0) previous_final_test is the original df, on future iterations (i>0) it contains the predictions for t+0 through t+i-1
    previous_final_test = df
    
    i=0
    features = feature_choices[i:i+memory]
    real_features = ['GrowthRate_t-' + str(j+1) for j in range(memory)] 
        
    # current_final_test is the df where we add all the state predictions for the current iteration (day)
    current_final_test = pd.DataFrame()
        
    train = get_in_date_range(df, first_date = '2020-04-10', last_date = day_0, date_col = date_col)
    # remove nans and infs from each line of dataframes
    train = train[~train.isin([np.nan, np.inf, -np.inf]).any(1)]
    
    
    x_train = train[real_features]
   
    y_train = train['GrowthRate']
        
    X_train = np.reshape(x_train.to_numpy(), (x_train.to_numpy().shape[0], x_train.to_numpy().shape[1], 1))
    # model = load_model("./bidir_lstm")
    act = 'relu'
    mode = mode_winner
    epochs = epochs_winner

    if (model_winner=='model_1'):
        model = Sequential()
        model.add(Bidirectional(LSTM(250, activation=act, return_sequences = False), merge_mode=mode, input_shape=(memory, 1)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse', metrics=['mse'])        
        model.fit(X_train, y_train.to_numpy(), epochs=epochs, verbose=1)
        model.save("./bidir_lstm")
        
    elif (model_winner=='model_2'):
        model = Sequential()
        model.add(Bidirectional(LSTM(250, activation=act, return_sequences = True), merge_mode=mode, input_shape=(memory, 1)))
        model.add(Bidirectional(LSTM(150, activation=act, return_sequences = False), merge_mode=mode))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse', metrics=['mse']) 
        model.fit(X_train, y_train.to_numpy(), epochs=epochs, verbose=1)
        model.save("./bidir_lstm")
    else:
        model = Sequential()
        model.add(Bidirectional(LSTM(250, activation=act, return_sequences = True), merge_mode=mode, input_shape=(memory, 1)))
        model.add(Bidirectional(LSTM(150, activation=act, return_sequences = True), merge_mode=mode))
        model.add(Bidirectional(LSTM(100, activation=act), merge_mode=mode))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse', metrics=['mse'])
        model.fit(X_train, y_train.to_numpy(), epochs=epochs, verbose=1)
        model.save("./bidir_lstm")
        
    for i in range(forward_days):
        features = feature_choices[i:i+memory]
        real_features = ['GrowthRate_t-' + str(j+1) for j in range(memory)]
       
        current_final_test = pd.DataFrame()
        
        for state1 in start_date:
            
            test_df = previous_final_test.loc[previous_final_test[region_col] == state1]
            test0 = get_in_date_range(test_df, first_date=day_0, last_date=mod_date(day_0, 1), date_col=date_col)

                #we create a copy in which we will modify feature names (which include some 'pred_forward_day_x' features) to match the real_features from the train rows (all 'GrowthRate_t-x')
            test = test0.copy(deep = True)
            test = test[~test.isin([np.nan, np.inf, -np.inf]).any(1)]
            x_test = test[features]
                 #rename_features maps 7 days before the current iteration day to GrowthRate_t-7, 7 days before to GrowthRate_t-6, etc.
            x_test = x_test.rename(columns = rename_features(i, features, memory))

            X_test = np.reshape(x_test.to_numpy(), (x_test.to_numpy().shape[0], x_test.to_numpy().shape[1],1))
            preds = model.predict(X_test)

            test0['pred_forward_day_'+str(i)] = preds.flatten() # add the new prediction as a new column
                #pred_high_day_i and pred_low_day_i
    #             x_test = x_test.rename(columns = undo_rename(features)) # make sure that original column names are not changed when they are changed in the copy

            current_final_test = pd.concat([current_final_test, test0], sort = False)
        previous_final_test = current_final_test
        
    # after the final iteration, previous_final_test contains predictions for the next forward_days starting from day_0
    return previous_final_test

def predict_covid(df, start_date, memory = 10, forward_days = 14, split_date = '2020-05-01', day_0 = '2020-05-01', real_GR = False, deterministic = True, r = 1,date_col='date', region_col='state', target_col='cases'):
    '''
    everything before split_date is train 
    
    '''
    
    #This section of code creates the forward and back features
    
    df0 = df.copy(deep=True) # deep copy might not be needed, just for security
    
    #remove some states/territories with late or small number of cases
    #CHANGE TO KEEPING STATES - hard copy in this code maybe global variable
    df0 = df0.loc[~df0[region_col].isin(['West Virginia','District of Columbia','Puerto Rico','American Samoa', 'Diamond Princess','Grand Princess','Guam','Northern Mariana Islands','Virgin Islands'])]
    
    df0 = df0.sort_values(by=[region_col, date_col])#has to be sorted by days to create growth rates
    #is_100 = df0['cases']>=100
    #df0 = df0[is_100]
    df0['GrowthRate'] = (df0.groupby(region_col)[target_col].shift(0) / df0[target_col].shift(1) - 1) #group by state so that consecutive rows are consecutive days in a single state

    #create the t-1 to t-memory growth rates
    for i in range(memory):
        df0['GrowthRate_t-' + str(i+1)] = df0.groupby(region_col)['GrowthRate'].shift(i+1)
        
    df0[target_col+'_t-1'] = df0[target_col].shift(1)

    #this is used only if we are using the alternate method where we run nearest neighbors on predictions in the train set
    if real_GR:
        for i in range(forward_days):
            df0['GrowthRate_t+' + str(i)] = df0.groupby(region_col)['GrowthRate'].shift(-i)

        for i in range(forward_days):
            df0['actual_growth_for_next_{}days'.format(i+1)] = (df0[target_col].shift(-i)/df0[target_col].shift(1)) - 1
    '''
    threshold: multiplier on the nearest distance that we cut off at when assigning weights, e.g. a point outside the threshold gets a weight of 0
    n: maximum number of nearest neighbors
    p: either L1 norm (manhattan) or L2 norm
    func: get_weights or get_weights_sq, whether the distance norm will be squared or not
    '''
    df0=df0.fillna(0)
    
    (model_winner,mode_winner,epochs_winner) = get_best_parameters(df0, memory, split_date,forward_days, r, date_col)
    
    # (model_winner,mode_winner,epochs_winner) = ('model_3','concat',20)
    #we run the method using the best parameters according to the split date
    predictions = match_to_real_growth(df0, start_date,model_winner,mode_winner,epochs_winner, memory, forward_days, day_0, split_date, deterministic,region_col, date_col)
    
    #we have finished producing predictions, and move on to converting predicted growth rates into predicted cases
    
    #convert growth rates to cumulative growth rates -- here we need to add 1 to each predicted growth rate so that when multiplied they represent growth rate over multiple days
    #the cumulative growth rate over n days starting today = (1+ GR_0) * (1+GR_1) * ... * (1+ GR_n-1)
    predictions['pred_growth_for_next_1days'] = predictions['pred_forward_day_0'] + 1
    for i in range(1,forward_days): 
        predictions['pred_growth_for_next_{}days'.format(i+1)] = predictions['pred_growth_for_next_{}days'.format(i)]*(predictions['pred_forward_day_'+ str(i)] + 1)

    for i in range(forward_days):
        predictions['pred_growth_for_next_{}days'.format(i+1)] = predictions['pred_growth_for_next_{}days'.format(i+1)] - 1

    #convert cumulative growth rates to cases
    for i in range(forward_days):
        predictions[target_col + '_predicted_day_' + str(i)] = np.round(predictions[target_col+'_t-1']*(predictions['pred_growth_for_next_{}days'.format(i+1)]+1))
    
    columns_to_keep = [region_col, date_col, target_col] + [target_col+'_predicted_day_' + str(i) for i in range(forward_days)]
   
    simple_output = predictions[columns_to_keep]
    transposed_simple_output = transpose_case_df(simple_output, forward_days, day_0, date_col, region_col, target_col)

    return transposed_simple_output, predictions
