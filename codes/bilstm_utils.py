
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
import matplotlib.pyplot as plt
import keras
from keras import backend as K

from keras.layers import Layer
from keras.models import Sequential, load_model

from keras.losses import MSE
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers import TimeDistributed
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from tensorflow.keras.models import load_model
from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans
from kneed import KneeLocator
#%% Helper Functions
from params import model_path, result_path

def wmape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred).astype('float')
    return sum((np.abs(y_true - y_pred)) ) * 100 / sum(y_true)

def get_in_date_range(dataset, first_date = '2020-01-01', last_date = '2020-12-31', date_col='date'):
    return dataset.loc[(dataset[date_col].astype('datetime64') >= np.datetime64(first_date)) & (dataset[date_col].astype('datetime64') < np.datetime64(last_date))]

def mod_date(date, interval):
    return str(np.datetime64(date) + np.timedelta64(interval, 'D'))

# def rename_features(i, features, memory):
#     dictionary = {}
#     for j in range(memory):
#         dictionary[features[3*j]] = 'GrowthRate_t-' +str(memory-j)
#         dictionary[features[3*j+1]] = 'temperature_t-' +str(memory-j)
#         dictionary[features[3*j+2]] = 'growthcaseslag_t-' +str(memory-j)
#     return dictionary

def rename_features(i, features, memory):
    dictionary = {}
    for j in range(memory):
        dictionary[features[2*j]] = 'GrowthRate_t-' +str(memory-j)
        dictionary[features[2*j+1]] = 'temperature_t-' +str(memory-j)
       
    return dictionary

# def rename_features(i, features, memory):
#     dictionary = {}
#     for j in range(memory):
#         dictionary[features[j]] = 'GrowthRate_t-' +str(memory-j)
#     return dictionary

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

def time_clustering(state_df, day_0, days_before=30, date_col='date', region_col='county', target_col='cases'):

    """
    input:
    
    state_df: pandas dataframe that contains the data
    day_0: int first day of prediction
    days_before: how many days before day_0, you will use data for time clustering
    
    output:
    
    clusters: list of lists of clusters
    
    """
    
    
    cluster_state_df = state_df.copy()
    cluster_state_df['GrowthRate'] = (state_df.groupby(region_col)[target_col].shift(0) / state_df.groupby(region_col)[target_col].shift(1) - 1)
    



    cluster_state_df = get_in_date_range(cluster_state_df,
                                         first_date = mod_date(day_0, -days_before),
                                         last_date = mod_date(day_0, 0),
                                         date_col = date_col)


    cluster_state_df = cluster_state_df.loc[:, cluster_state_df.columns.intersection([region_col,date_col,'GrowthRate'])]

    cluster_state_df = cluster_state_df[~cluster_state_df.isin([np.nan, np.inf, -np.inf]).any(1)].copy(deep=True)

    time_series = cluster_state_df.groupby(region_col)['GrowthRate'].apply(list)
    time_series_list = to_time_series_dataset([t for t in time_series])

    regions = cluster_state_df[region_col].unique().tolist()
    number_of_regions = len(regions)-1

    inertias = []

    for k in range(1,number_of_regions,1):
        print("k is: ", k)
        model = TimeSeriesKMeans(n_clusters=k, metric="dtw", max_iter=100, dtw_inertia=True, n_jobs=-1)
        model.fit(time_series_list)
        inertias.append(model.inertia_)

    kn = KneeLocator(range(1,number_of_regions,1), inertias, curve='convex', direction='decreasing')

    print("Optimal value of clusters is: ", kn.knee)
    
    model = TimeSeriesKMeans(n_clusters= kn.knee, metric="dtw", max_iter=100, dtw_inertia=True, n_jobs=-1)
    model.fit(time_series_list)
    
    clusters = [[] for _ in range(0,kn.knee,1)]
    for i in range(len(model.labels_)):
        clusters[model.labels_[i]].append(regions[i])

    return clusters


def get_best_parameters(df, cluster, use_auto, memory, split_date, r, region_col='state', col_date='date'):
    

    K.clear_session()
    auto_model = None
    model_winner = ''
    mode_winner = ''
    epochs_winner = -1
    wmape_min = 10**10
    
    LOSS = 'mse'
    BATCH = 8
    LR = 1e-4
    SCALE = 1
    n_features = 2
    features = []
    for i in [memory-i for i in range(memory)]:
        features.append('GrowthRate_t-'+ str(i))
        features.append('temperature_t-'+ str(i))
        # features.append('growthcaseslag_t-'+ str(i))
        
    
    for mode in ['ave']:
        
        # once we determine first date for each state, we will modify the March 22 hard coding

        # df = df[df['deaths'] >= 100]
        train = get_in_date_range(df, first_date = '2020-04-01', last_date = mod_date(split_date, -r), date_col=col_date)
        test = get_in_date_range(df, first_date = mod_date(split_date, -r), last_date = split_date, date_col=col_date)

        train = train[~train[features].isin([np.nan, np.inf, -np.inf]).any(1)].copy()  
        test = test[~test[features].isin([np.nan, np.inf, -np.inf]).any(1)].copy()

        if (len(cluster)>20):
            train = train.loc[train[region_col].isin(cluster)].copy(deep=True)
        
        test = test.loc[test[region_col].isin(cluster)].copy(deep=True)        
        
        # create dataset
        x_train = train[features]*SCALE
        y_train = train['GrowthRate']*SCALE
                
        x_test = test[features]*SCALE
        y_test = test['GrowthRate']*SCALE

        # transform for lstm
        # X_train = np.reshape(x_train.to_numpy(), (x_train.to_numpy().shape[0], x_train.to_numpy().shape[1]//1, 1))
        # X_test = np.reshape(x_test.to_numpy(), (x_test.to_numpy().shape[0], x_test.to_numpy().shape[1]//1, 1))
        X_train = np.reshape(x_train.to_numpy(), (x_train.to_numpy().shape[0], x_train.to_numpy().shape[1]//n_features, n_features))
        X_test = np.reshape(x_test.to_numpy(), (x_test.to_numpy().shape[0], x_test.to_numpy().shape[1]//n_features, n_features))

        # set hyperparams
        act = 'relu'
        epochs = 100
        epochs_ = 100      
        patience = 0
        patience_ = 0
        auto_epochs = 1
        # autoencoder
        timesteps = memory
        

        if (use_auto):

            a_train = get_in_date_range(df, first_date = '2020-04-01', last_date = mod_date(split_date, -r), date_col=col_date)
            a_test = get_in_date_range(df, first_date = mod_date(split_date, -r), last_date = split_date, date_col=col_date)

            a_train = a_train[~a_train[features].isin([np.nan, np.inf, -np.inf]).any(1)].copy()  
            a_test = a_test[~a_test[features].isin([np.nan, np.inf, -np.inf]).any(1)].copy()
     
            
            # create dataset
            a_x_train = a_train[features]*SCALE
            a_y_train = a_train['GrowthRate']*SCALE
            a_x_test = a_test[features]*SCALE
            a_y_test = a_test['GrowthRate']*SCALE

            a_X_train = np.reshape(a_x_train.to_numpy(), (a_x_train.to_numpy().shape[0], a_x_train.to_numpy().shape[1]//n_features, n_features))
            a_X_test = np.reshape(a_x_test.to_numpy(), (a_x_test.to_numpy().shape[0], a_x_test.to_numpy().shape[1]//n_features, n_features))

            auto_model = Sequential()
            auto_model.add(LSTM(600, activation='relu',  return_sequences=True, input_shape=(timesteps,n_features)))
            auto_model.add(LSTM(300, activation='relu', return_sequences=False))
            auto_model.add(RepeatVector(timesteps))
            auto_model.add(LSTM(300, activation='relu', return_sequences=True))
            auto_model.add(LSTM(600, activation='relu', return_sequences=True))
            auto_model.add(TimeDistributed(Dense(n_features)))
            opt = keras.optimizers.Adam(learning_rate=LR)
            auto_model.compile(optimizer=opt, loss=LOSS)

            es = EarlyStopping(monitor='val_loss', verbose=1, patience = patience_)
            mc = ModelCheckpoint(model_path+"/auto_lstm.h5", monitor='val_loss', mode='min', verbose=1, save_best_only=True)
            auto_model.fit(a_X_train, a_X_train, batch_size = BATCH, validation_data=(X_test, X_test), epochs=epochs_, verbose=1, callbacks=[es,mc]) 
            auto_model = load_model(model_path+"/auto_lstm.h5")
            auto_epochs = es.stopped_epoch+patience_

            X_train = auto_model.predict(X_train)

        # model1
        # K.clear_session()
        # model = Sequential()
        # model.add(Bidirectional(LSTM(300, activation=act, return_sequences = False), merge_mode=mode, input_shape=(memory, 2)))
        # model.add(Dense(1))
        # opt = keras.optimizers.Adam(learning_rate=LR)
        # model.compile(optimizer=opt, loss='mse', metrics=['mse'])
        
        # es = EarlyStopping(monitor='val_loss', verbose=1, patience = patience)
        # mc = ModelCheckpoint(model_path+"/bidir_lstm_1.h5", monitor='val_loss', mode='min', verbose=1, save_best_only=True)
        # model.fit(X_train, y_train.to_numpy(), batch_size = BATCH, validation_data=(X_test, y_test.to_numpy()), epochs=epochs, verbose=1, callbacks=[es,mc])  
        # model = load_model(model_path+"/bidir_lstm_1.h5")
        # preds = model.predict(X_test)
        
        # if (wmape(y_test.to_numpy(),preds.flatten()) <wmape_min):
        #     wmape_min = wmape(y_test.to_numpy(),preds.flatten())
        #     model_winner = 'model_1'
        #     mode_winner = mode
        #     epochs_winner = es.stopped_epoch+patience

           
        # model2
        K.clear_session()
        model = Sequential()
        model.add(Bidirectional(LSTM(300, activation=act, return_sequences = True), merge_mode=mode, input_shape=(memory, n_features)))
        model.add(Bidirectional(LSTM(200, activation=act, return_sequences = False), merge_mode=mode))
        model.add(Dense(1,activation='relu'))
        opt = keras.optimizers.Adam(learning_rate=LR)
        model.compile(optimizer=opt, loss='mse', metrics=['mse'])
        
        es = EarlyStopping(monitor='val_loss', verbose=1, patience = patience)
        mc = ModelCheckpoint(model_path+"/bidir_lstm_2.h5", monitor='val_loss', mode='min', verbose=1, save_best_only=True)
        model.fit(X_train, y_train.to_numpy(), batch_size = BATCH, validation_data=(X_test, y_test.to_numpy()), epochs=epochs, verbose=1, callbacks=[es,mc])  
        model = load_model(model_path+"/bidir_lstm_2.h5")
        preds = model.predict(X_test)
        
        if (wmape(y_test.to_numpy(),preds.flatten()) <wmape_min):
            wmape_min = wmape(y_test.to_numpy(),preds.flatten())
            model_winner = 'model_2'
            mode_winner = mode
            epochs_winner = es.stopped_epoch 

        # # model3
        # K.clear_session()
        # model = Sequential()
        # model.add(Bidirectional(LSTM(750, activation=act, return_sequences = True), merge_mode=mode, input_shape=(memory, n_features)))
        # model.add(Bidirectional(LSTM(500, activation=act, return_sequences = True), merge_mode=mode))
        # model.add(Bidirectional(LSTM(250, activation=act), merge_mode=mode))
        # model.add(Dense(1))
        # opt = keras.optimizers.Adam(learning_rate=LR)
        # model.compile(optimizer=opt, loss=LOSS, metrics=['mse'])
        
        # es = EarlyStopping(monitor='val_loss', verbose=1, patience = patience)
        # mc = ModelCheckpoint(model_path+"/bidir_lstm_3.h5", monitor='val_loss', mode='min', verbose=1, save_best_only=True)
        # model.fit(X_train, y_train.to_numpy(), batch_size = BATCH, validation_data=(X_test, y_test.to_numpy()), epochs=epochs, verbose=1, callbacks=[es,mc])  
        # model = load_model(model_path+"/bidir_lstm_3.h5")
        # preds = model.predict(X_test)
        
        # if (wmape(y_test.to_numpy(),preds.flatten()) <wmape_min):
        #     wmape_min = wmape(y_test.to_numpy(),preds.flatten())
        #     model_winner = 'model_3'
        #     mode_winner = mode
        #     epochs_winner = es.stopped_epoch+patience

        # # model4
        # K.clear_session()
        # model = Sequential()
        # model.add(Bidirectional(LSTM(750, activation=act, return_sequences = True), merge_mode=mode, input_shape=(memory, 2)))
        # model.add(Bidirectional(LSTM(500, activation=act, return_sequences = True), merge_mode=mode))
        # model.add(Bidirectional(LSTM(250, activation=act, return_sequences = True), merge_mode=mode))
        # model.add(Bidirectional(LSTM(100, activation=act ), merge_mode=mode))
        # model.add(Dense(1))
        # opt = keras.optimizers.Adam(learning_rate=LR)
        # model.compile(optimizer=opt, loss=LOSS, metrics=['mse'])
        
        # es = EarlyStopping(monitor='val_loss', verbose=1, patience = patience)
        # mc = ModelCheckpoint(model_path+"/bidir_lstm_4.h5", monitor='val_loss', mode='min', verbose=1, save_best_only=True)
        # model.fit(X_train, y_train.to_numpy(), batch_size = BATCH, validation_data=(X_test, y_test.to_numpy()), epochs=epochs, verbose=1, callbacks=[es,mc])  
        # model = load_model(model_path+"/bidir_lstm_4.h5")
        # preds = model.predict(X_test)
        
        # if (wmape(y_test.to_numpy(),preds.flatten()) <wmape_min):
        #     wmape_min = wmape(y_test.to_numpy(),preds.flatten())
        #     model_winner = 'model_4'
        #     mode_winner = mode
        #     epochs_winner = es.stopped_epoch+patience

    print(model_winner)
    print(mode_winner)    
    print(wmape_min)   

    if (wmape_min > 70):
        print(cluster)

    return (model_winner,mode_winner,epochs_winner,auto_epochs)

def find_best_model_covid(df, start_date, cluster, use_auto = False, memory = 10, split_date = '2020-04-01', day_0 = '2020-04-01', real_GR = False, deterministic = True, r = 1,date_col='date', region_col='state', target_col='cases'):
    
    return get_best_parameters(df, cluster, use_auto, memory, split_date, r, region_col, date_col)

def fit_covid(df, start_date, cluster, use_auto, cluster_index, model_winner,mode_winner,epochs_winner,auto_epochs, memory, day_0, split_date, deterministic, region_col='county', date_col='date', argument = ''):
    
    K.clear_session()

    # df = df[df['deaths'] >= 100]

    auto_model = None 
    
    LOSS = 'mse'
    BATCH = 8
    LR = 1e-4
    SCALE = 1
    n_features = 2
    # on first iteration (i=0) previous_final_test is the original df, on future iterations (i>0) it contains the predictions for t+0 through t+i-1
    previous_final_test = df
    
    features = []
    for i in [memory-i for i in range(memory)]:
        features.append('GrowthRate_t-'+ str(i))
        features.append('temperature_t-'+ str(i))
        # features.append('growthcaseslag_t-'+ str(i))
    
    real_features = features.copy()
    
    train = get_in_date_range(df, first_date = '2020-04-01', last_date = day_0, date_col = date_col)
    train = train[~train[features].isin([np.nan, np.inf, -np.inf]).any(1)].copy()  
    if (len(cluster)>20):
        train = train.loc[train[region_col].isin(cluster)].copy(deep=True)
        
    
    x_train = train[real_features]*SCALE
    y_train = train['GrowthRate']*SCALE
        
    X_train = np.reshape(x_train.to_numpy(), (x_train.to_numpy().shape[0], x_train.to_numpy().shape[1]//n_features, n_features))
    # model = load_model("./bidir_lstm")

    # autoencoder
    timesteps = memory


    if (use_auto):

        a_train = get_in_date_range(df, first_date = '2020-04-01', last_date = day_0, date_col = date_col)
        a_train = a_train[~a_train[features].isin([np.nan, np.inf, -np.inf]).any(1)].copy()  
        
        a_x_train = a_train[real_features]*SCALE
        a_y_train = a_train['GrowthRate']*SCALE
            
        a_X_train = np.reshape(a_x_train.to_numpy(), (a_x_train.to_numpy().shape[0], a_x_train.to_numpy().shape[1]//2, 2))

        auto_model = Sequential()
        auto_model.add(LSTM(600, activation='relu',  return_sequences=True, input_shape=(timesteps,n_features)))
        auto_model.add(LSTM(300, activation='relu', return_sequences=False))
        auto_model.add(RepeatVector(timesteps))
        auto_model.add(LSTM(300, activation='relu', return_sequences=True))
        auto_model.add(LSTM(600, activation='relu', return_sequences=True))
        auto_model.add(TimeDistributed(Dense(n_features)))
        opt = keras.optimizers.Adam(learning_rate=LR)
        auto_model.compile(optimizer=opt, loss=LOSS)

        auto_model.fit(a_X_train, a_X_train, batch_size = BATCH, epochs=auto_epochs, verbose=1)

        auto_model.save(model_path+"/auto_bidir_lstm_cluster_"+str(cluster_index)+"_"+argument+".h5")

        X_train = auto_model.predict(X_train)
        

    act = 'relu'
    mode = mode_winner
    epochs = epochs_winner

    if (model_winner=='model_1'):
        model = Sequential()
        model.add(Bidirectional(LSTM(300, activation=act, return_sequences = False), merge_mode=mode, input_shape=(memory, n_features)))
        model.add(Dense(1))
        opt = keras.optimizers.Adam(learning_rate=LR)
        model.compile(optimizer=opt, loss=LOSS, metrics=['mse'])
        model.fit(X_train, y_train.to_numpy(), batch_size = BATCH, epochs=epochs, verbose=1)
        model.save(model_path+"/bidir_lstm_cluster_"+str(cluster_index)+"_"+argument+".h5")
        
    elif (model_winner=='model_2'):
        model = Sequential()
        model.add(Bidirectional(LSTM(300, activation=act, return_sequences = True), merge_mode=mode, input_shape=(memory, n_features)))
        model.add(Bidirectional(LSTM(200, activation=act, return_sequences = False), merge_mode=mode))
        model.add(Dense(1, activation='relu'))
        opt = keras.optimizers.Adam(learning_rate=LR)
        model.compile(optimizer=opt, loss=LOSS, metrics=['mse'])
        model.fit(X_train, y_train.to_numpy(), batch_size = BATCH, epochs=epochs, verbose=1)
        model.save(model_path+"/bidir_lstm_cluster_"+str(cluster_index)+"_"+argument+".h5")

    elif (model_winner=='model_3'):
        model = Sequential()
        model.add(Bidirectional(LSTM(750, activation=act, return_sequences = True), merge_mode=mode, input_shape=(memory, n_features)))
        model.add(Bidirectional(LSTM(500, activation=act, return_sequences = True), merge_mode=mode))
        model.add(Bidirectional(LSTM(250, activation=act, return_sequences = False), merge_mode=mode))
        model.add(Dense(1))
        opt = keras.optimizers.Adam(learning_rate=LR)
        model.compile(optimizer=opt, loss=LOSS, metrics=['mse'])
        model.fit(X_train, y_train.to_numpy(), batch_size = BATCH, epochs=epochs, verbose=1)
        model.save(model_path+"/bidir_lstm_cluster_"+str(cluster_index)+"_"+argument+".h5")
    else:
        model = Sequential()
        model.add(Bidirectional(LSTM(750, activation=act, return_sequences = True), merge_mode=mode, input_shape=(memory, n_features)))
        model.add(Bidirectional(LSTM(500, activation=act, return_sequences = True), merge_mode=mode))
        model.add(Bidirectional(LSTM(250, activation=act, return_sequences = True), merge_mode=mode))
        model.add(Bidirectional(LSTM(100, activation=act ), merge_mode=mode))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss=LOSS, metrics=['mse'])
        model.fit(X_train, y_train.to_numpy(),batch_size = BATCH, epochs=epochs, verbose=1)
        model.save(model_path+"/bidir_lstm_cluster_"+str(cluster_index)+"_"+argument+".h5")

    return None


def predict_covid(df, start_date, model, auto_model, cluster, use_auto, memory = 10, forward_days = 14, split_date = '2020-04-01', day_0 = '2020-04-01', real_GR = False, deterministic = True, r = 30,date_col='date', region_col='state', target_col='cases'):


    SCALE = 1
    n_features = 2
    features = []
    for i in [memory-i for i in range(memory)]:
        features.append('GrowthRate_t-'+ str(i))
        features.append('temperature_t-'+ str(i))
        # features.append('growthcaseslag_t-'+ str(i))

    preds = []
    for i in range(forward_days):
        preds.append('pred_forward_day_' + str(i))
        preds.append('pred_forward_temp_' + str(i))
        # preds.append('pred_forward_cases_' + str(i))

    feature_choices = features + preds
    previous_final_test = df.copy(deep=True)
    
    temp_df = pd.read_csv('../data/01_05_2021_states_combined_temperature_demographics_and_holidays_v3_lags_pred_sep.csv')

    for i in range(forward_days):
        print("day", i)
        # features = feature_choices[2*i:2*i+2*memory]
        features = feature_choices[n_features*i:n_features*i+n_features*memory]
        real_features = []
        for j in range(memory):
            real_features.append('GrowthRate_t-' + str(j+1))
            real_features.append('temperature_t-' + str(j+1))
            # real_features.append('growthcaseslag_t-' + str(j+1))
        # current_final_test is the df where we add all the state predictions for the current iteration (day)
        current_final_test = pd.DataFrame()
        
        for state1 in start_date:
            # print(state1)    
            if (state1 in cluster):

                test_df = previous_final_test.loc[previous_final_test[region_col] == state1]
                test0 = get_in_date_range(test_df, first_date=day_0, last_date=mod_date(day_0, 1), date_col=date_col)

                test = test0.copy(deep = True)
                # test = test[~test.isin([np.nan, np.inf, -np.inf]).any(1)]
                
                x_test = test[features]*SCALE
                #rename_features maps 7 days before the current iteration day to GrowthRate_t-7, 7 days before to GrowthRate_t-6, etc.
                x_test = x_test.rename(columns = rename_features(i, features, memory))

                X_test = np.reshape(x_test.to_numpy(), (x_test.to_numpy().shape[0], x_test.to_numpy().shape[1]//n_features,n_features))
                if(use_auto):
                    X_test = auto_model.predict(X_test)
                preds = model.predict(X_test)

                if(preds.flatten()[0]<=0):
                    test0['pred_forward_day_'+str(i)] = abs(np.random.normal(0,1))/300 # add the new prediction as a new column
                else:
                    test0['pred_forward_day_'+str(i)] = preds.flatten()[0]/SCALE

                
                temperature = float(temp_df.loc[ (temp_df[region_col] == state1) & (temp_df['date'] ==  str(mod_date(day_0, i-20))[0:10])]['temperature'])
                # grpred = float(temp_df.loc[ (temp_df[region_col] == state1) & (temp_df['date'] ==  str(mod_date(day_0, i-20))[0:10])]['growthcasespred'])
                test0['pred_forward_temp_'+str(i)] = temperature
                # test0['pred_forward_cases_'+str(i)] = grpred

                current_final_test = pd.concat([current_final_test, test0], sort = False)
        previous_final_test = current_final_test
     

    predictions = previous_final_test

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

    return transposed_simple_output
