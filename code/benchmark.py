import pandas as pd
import numpy as np
import os
import math
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt

###########################
## USER INPUTS ############
###########################

month = 'June' #Currently 'June' or 'September' or 'November'
metric = 'Deaths' #'Deaths' or 'Cases'
path_to_mit_results = '../Predictions/backtest_tr_cutoff_state_deaths_2020-06-01_2020-07-01.csv'
exclude_states = ['US','District of Columbia','Hawaii','Georgia','Maine','Vermont','Alaska','Arkansas','Utah','Wyoming','West Virginia','South Dakota','North Dakota','Montana','Idaho']
model_to_plot = 'agg'
models_to_rank = ['agg','mdp','bilstm','knn','sir']
plot_filename = 'benchmark_plot_' + month + '_' + metric + '_' + model_to_plot + '.jpg'
rank_table_path = '../Processed data/model_ranks_' + month + '_' + metric + '_' + model_to_plot + '.csv'

# grab all the CDC models
full_model_list = os.listdir('../covid19-forecast-hub/data-processed')
# here are the ones I've been leaving out because they skew the graph scale
bad_models = ['USC-SI_kJalpha_RF','IQVIA_ACOE-STAN','STH-3PU','Caltech-CS156','CMU-TimeSeries', 'NotreDame-FRED','UCM_MESALab-FoGSEIR']

###########################
## FUNCTIONS ##############
###########################

# A function to load a dataframe from a csv to match location ids and state names
# This csv was made from scratch and needs to be included in the directory
# Inputs: none
# Returns: a pandas dataframe with the contents of the csv
def load_locations():
    locs = pd.read_csv("../code/locations.csv", dtype = {'location': str})
    return(locs)


# A function to convert the location-matching dataframe into a dictionary
# Dictionary will have state names as keys and IDs as values
# Note, it also includes the key 'US' with an assigned value of 'US'
# This dictionary will be used to allow users to input desired state names rather than ID numbers
# Inputs: data frame with a column of state names and a column of state ids produced by the load_locaations() function
# Returns Dictionary described above
def build_location_dict(locations_frame):
    ids = locations_frame.id.to_list()
    states = locations_frame.state.to_list()
    #dict_idtostate = {ids[i]:states[i] for i in range(len(ids))}
    dict_statetoid = {states[i]:ids[i] for i in range(len(ids))}
    return(dict_statetoid)


# A function to load the observed values for covid statistics
# Also filters results to include only state- and nation-wide results and exclude county-wide results using the dictionary created by the build_location_dict() function
# Inputs:
# locations_dict - dictionary with state names as keys and state ids as values (created by the build_location_dict() function)
# metric - what covid measure to evaluate:
# 'Cumulative Deaths' [DEFAULT]
# 'Cumulative Cases'
# 'Incident Deaths'
# 'Incident Cases'
def load_truth(locations_dict, metric='Cumulative Deaths'):
    # Check for valid metric name
    if metric not in ['Cumulative Deaths', 'Cumulative Cases', 'Incident Deaths', 'Incident Cases']:
        print("Please input a valid covid metric: 'Cumulative Deaths','Cumulative Cases', 'Incident Deaths', 'Incident Cases'")
        return (None)

    # load data
    path = "../covid19-forecast-hub/data-truth/truth-" + metric + ".csv"
    truth = pd.read_csv(path, dtype={'location': str})

    # filter out county values
    truth = truth.loc[truth.location.isin(locations_dict.values()), :]

    return (truth)


# A function to load the predictions produced by our models
# Inputs: a path to the location of the csv file
# Outputs: a pandas dataframe containing the model predictions
def load_mit_model(path):
    if path is None:
        print("Please input a valid path to covid predictions")
        return(None)

    model = pd.read_csv(path)
    return (model)


# A function to load the predictions output by one of the many covid models and filter results by date, prediction scope, location, and prediction type
# Inputs:
# model_name - which covid model to evaluate results--must match exactly the directory name in the covid19-forecast repository
# start_date - date on which to start looking for model predictions--must be a string in 'YYYY-MM-DD' format
# end_date - date on which to end looking for model predictions (inclusive)--must be a string in 'YYYY-MM-DD' format
# target - list of desired model output
# '1 wk ahead cum death' [DEFAULT]
# '2/3/4/... wk ahead cum death'
# '1/2/3/... wk ahead inc deaths'
# '1/2/3/... wk ahead inc cases'
# location_list - the states (full names) for which model predictions will be returned--default is None and all states + nationwide (if available) will be returned
# ['US'] will return just nationwide predictions if available--Note: will not equal sum of 50 states + DC because 'US' includes territories and many models do not
# pred_type - the prediction type to return, either 'point' or 'quantile'
# quantile - if pred_type == 'quantile' this field will determine which quantile to return, if left blank will default to 0.5
# Returns: A pandas dataframe with predictions matching the criteria above
def load_predictions(model_name=['CovidAnalytics-DELPHI'], start_date='2020-04-27', end_date='2021-01-11',
                     target=['1 wk ahead cum death'], location_list=None, pred_type='point', quantile=None):
    # initialize the dataframe for later
    preds_full = None
    for model in model_name:
        # construct path to model data
        loc = "../covid19-forecast-hub/data-processed/" + model

        # convert to date objects
        today = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()

        # for each day in the range, load the data and append it to the full dataframe
        preds = None
        while today <= end:
            # construct the rest of the filepath
            today_str = str(today)
            file = today_str + "-" + model
            path = loc + "/" + file + ".csv"

            # try to open the file if it's there
            try:
                today_preds = pd.read_csv(path, dtype={'location': str, 'quantile': str})

                # reorder columns
                proper_col_order = ['forecast_date', 'target_end_date', 'target', 'location', 'type', 'quantile',
                                    'value']
                today_preds = today_preds[proper_col_order]

                # The first time through, replace the dataframe, otherwise stack them on top of each other
                if preds is None:
                    preds = today_preds
                else:
                    preds = pd.concat([preds, today_preds], axis=0)
            # if the file doesn't exist, just skip it
            except:
                pass

            # increment date
            today += timedelta(days=1)

        # check to make sure input target is valid then filter results (cheating for now)
        try:
            preds = preds.loc[preds.target.isin(target), :]
        except:
            pass
        try:
            # check to make sure input prediction type and quantile are valid then filter results
            if pred_type == 'point':
                preds = preds.loc[preds.type == pred_type, :]
            elif pred_type == 'quantile':
                if quantile is None:
                    quantile = '0.5'  # Default to 0.5 if no quantile is specified
                elif quantile not in preds['quantile'].to_list():  # otherwise make sure it's a valid quantile
                    print("Please input valid quantile: 0.01, 0.05,0.1,...,0.95,0.99")
                    return (None)
                preds = preds.loc[((preds.type == pred_type) & (preds['quantile'] == quantile)), :]
            else:
                print("Please input a value prediction type: either point or quantile")
                return (None)
        except:
            pass

        try:
            # check to make sure location_list is valid and then filter results
            if location_list is None:  # default is none, so don't filter by location
                pass
            elif location_list == ['US']:  # if asking for just nationwide results, check first to make sure that the model has data for 'US'
                if 'US' in preds.location.to_list():
                    preds = preds.loc[preds.location.isin(location_list), :]
                else:
                    print("Warning: this model has not made predictions for the US as a whole--returning all locations")
            else:  # otherwise, convert state names to IDs and filter
                # load location mapping and build dictionary to convert
                locs = load_locations()
                location_dict_statetoid = build_location_dict(locs)

                # build a list of IDs or throw an error if an invalid state name is input
                try:
                    id_list = [location_dict_statetoid[i] for i in location_list]
                except KeyError:
                    print("Invalid location listed")
                    return (None)
                # filter by location
                preds = preds.loc[preds.location.isin(id_list), :]

            # Add a column with the model name
            preds['model'] = [model] * len(preds)

            if preds_full is None:
                preds_full = preds
            else:
                preds_full = pd.concat([preds_full, preds], axis=0)
        except:
            pass

    return (preds_full)


# A function to join CDC model predictions with ground truth
# Inputs:
# truth_dataframe - a dataframe with ground truth predictions (built using the load_truth() function)
# predictions_dataframe - a dataframe with model predictions (built using the load_predictions() function)
# Returns: a single dataframe with ground truth values matched to model predictions
def merge_predictions(truth_dataframe, predictions_dataframe, metric='Cumulative Deaths'):
    # Check what we're trying to merge
    if metric == 'Cumulative Cases':
        print("Warning: CDC models do not predict Cumulative Cases. You probably want Incident Cases instead.")
    elif metric == 'Incident Cases':
        # incident cases in the truth dataset are day-by-day but they're weekly in the predictions so we need to roll them up
        truth_dataframe.sort_values(by=['location_name', 'date'], ascending=True, inplace=True)
        truth_dataframe = truth_dataframe.assign(value_rolling7=truth_dataframe['value'].rolling(7).sum())

    # left join predictions with truth--may lead to null values for ground truth if no matching entry in that table
    merged_df = pd.merge(left=predictions_dataframe, right=truth_dataframe,
                         how="left", left_on=['target_end_date', 'location'],
                         right_on=['date', 'location'])
    # tidy up: rename columns, drop duplicates, reorder columns, sort by state name
    if metric == 'Incident Cases':
        # Some weird manipulation to deal with the rolled up data that's probably overcomplicated
        merged_df.rename(columns={"value_x": "expected", "value_rolling7": "observed"}, inplace=True)
        merged_df.sort_values(by=['model', 'location_name', 'target_end_date'], ascending=True, inplace=True)
        date_col = merged_df[['target_end_date']].reset_index()['target_end_date']
        temp = merged_df.groupby(['model', 'location_name'], as_index=True)[
            ['target_end_date', 'expected', 'observed']].expanding().sum().reset_index()
        temp = pd.concat([temp[['model', 'location_name', 'expected', 'observed']], date_col], axis=1)
        merged_df = temp
    else:
        merged_df.rename(columns={"value_x": "expected", "value_y": "observed"}, inplace=True)

    try:
        # a little more cleanup if necessary
        merged_df.drop(['date'], axis=1, inplace=True)
        merged_df = merged_df[
            ['model', 'forecast_date', 'target_end_date', 'target', 'location', 'location_name', 'type',
             'quantile', 'expected', 'observed']]
        merged_df.sort_values(by=['model', 'target', 'forecast_date', 'location_name'], ascending=True, inplace=True)
    except:
        pass

    return (merged_df)


# A function to join MIT model predictions with ground truth
# Inputs:
# truth_dataframe - a dataframe with ground truth predictions (built using the load_truth() function)
# predictions_dataframe - a dataframe with model predictions (built using the load_mit_predictions() function)
# Returns: a single dataframe with ground truth values matched to model predictions
def rearrange_predictions_mit(predictions_dataframe, truth_dataframe=None, metric='Cumulative Deaths'):
    # in some versions the truth is uploaded with the model predictions, but it's better to merge with the truth here
    if truth_dataframe is not None:
        if metric == 'Cumulative Cases':
            # For cumulative cases, we need to convert to incremental cases to match with the CDC data
            # This means subtracting the ground truth value from the day before the predictions were made

            # Find out what day the predictions start
            date_list = predictions_dataframe.date.tolist()
            date_list = list(map(lambda x: datetime.strptime(x, "%Y-%m-%d").date(), date_list))
            # Take the min and subtract 1
            first_date = min(date_list) - timedelta(days=1)

            # This smaller dataframe has baseline truth values
            truth_small = truth_dataframe.loc[truth_dataframe.date == str(first_date), :]
            truth_small.rename(columns={"value": "cum"}, inplace=True)
        # Merge (left join) the predictions with the ground truth on each day
        predictions_dataframe = pd.merge(left=predictions_dataframe, right=truth_dataframe,
                                         how="left", left_on=['date', 'state'],
                                         right_on=['date', 'location_name'])
        predictions_dataframe = predictions_dataframe[['state', 'date', 'knn', 'sir', 'mdp', 'bilstm', 'agg', 'value']]
        # predictions_dataframe.rename(columns = {"value": "deaths"}, inplace = True)

        # if we're doing cumulative cases we need another merge with the filtered ground truth created above
        if metric == 'Cumulative Cases':
            predictions_dataframe = pd.merge(left=predictions_dataframe, right=truth_small,
                                             how="left", left_on=['state'], right_on=['location_name'])
            predictions_dataframe.drop(['date_y', 'location', 'location_name'], axis=1, inplace=True)
            predictions_dataframe.rename(columns={"date_x": "date"}, inplace=True)

            # subtract the baseline cases from each of the modeled predictions and the cumulative truth value
            predictions_dataframe = predictions_dataframe.assign(
                knn=predictions_dataframe['knn'] - predictions_dataframe['cum'])
            predictions_dataframe = predictions_dataframe.assign(
                sir=predictions_dataframe['sir'] - predictions_dataframe['cum'])
            predictions_dataframe = predictions_dataframe.assign(
                mdp=predictions_dataframe['mdp'] - predictions_dataframe['cum'])
            predictions_dataframe = predictions_dataframe.assign(
                bilstm=predictions_dataframe['bilstm'] - predictions_dataframe['cum'])
            predictions_dataframe = predictions_dataframe.assign(
                agg=predictions_dataframe['agg'] - predictions_dataframe['cum'])
            predictions_dataframe = predictions_dataframe.assign(
                value=predictions_dataframe['value'] - predictions_dataframe['cum'])

    # convert from wide to long form
    predictions_dataframe.rename(columns={"value": "truth"}, inplace=True)
    merged_df = predictions_dataframe.melt(id_vars=['state', 'date', 'truth'],
                                           value_vars=['sir', 'knn', 'mdp', 'bilstm', 'agg'])

    # tidy up: rename columns, drop duplicates, reorder columns, sort by state name
    # merged_df.drop(['date','location','location_name'], axis=1, inplace = True)
    merged_df.rename(columns={"variable": "model", "value": "expected", "truth": "observed", "state": "location_name",
                              "date": "target_end_date"}, inplace=True)
    merged_df = merged_df[['model', 'location_name', 'target_end_date', 'expected', 'observed']]
    merged_df.sort_values(by=['model', 'target_end_date', 'location_name'], ascending=True, inplace=True)

    #print(merged_df)
    return (merged_df)


# A function to calculate errors for model predictions
# Inputs:
    # merged_df - a pandas dataframe containing both modeled predictions and matched ground truth values (built_using the merge_predictions() function)
    # error_type
        # 'percent' [DEFAULT] - absolute error as a percentage of the ground truth value: abs(expected - observed) / observed
        # 'absolute' - absoute difference between ground truth and modeled values
        # 'squared' - squared difference between ground truth and modeled values
def calculate_prediction_errors(merged_df, errortype = 'percent'):
    if errortype == 'percent':
        merged_df['error'] = abs(merged_df.expected - merged_df.observed) / merged_df.observed
    elif errortype == 'absolute':
        merged_df['error'] = abs(merged_df.expected - merged_df.observed)
    elif errortype == 'squared':
        merged_df['error'] = (merged_df.expected - merged_df.observed)**2
    else:
        print("Please input valid error type: 'percent', 'absolute', 'squared'")
        return (None)
    return(merged_df)


# A function that combines each of the above helper functions to output comparisons between model predictions and ground truth values
# Inputs:
# metric - what covid measure to evaluate:
# 'Cumulative Deaths' [DEFAULT]
# 'Cumulative Cases'
# 'Incident Deaths'
# 'Incident Cases'
# model_name - which covid model to evaluate results--must match exactly the directory name in the covid19-forecast repository
# start_date - date on which to start looking for model predictions--must be a string in 'YYYY-MM-DD' format
# end_date - date on which to end looking for model predictions (inclusive)--must be a string in 'YYYY-MM-DD' format
# target - desired model output
# '1 wk ahead cum death' [DEFAULT]
# '2/3/4/... wk ahead cum death'
# '1/2/3/... wk ahead inc deaths'
# '1/2/3/... wk ahead inc cases'
# location_list - the states (full names) for which model predictions will be returned--default is None and all states + nationwide (if available) will be returned
# ['US'] will return just nationwide predictions if available--Note: will not equal sum of 50 states + DC because 'US' includes territories and many models do not
# pred_type - the prediction type to return, either 'point' or 'quantile'
# quantile - if pred_type == 'quantile' this field will determine which quantile to return, if left blank will default to 0.5
# error_type
# 'percent' [DEFAULT] - error as a percentage of the ground truth value: (expected - observed) / observed--will be negative if predicted values are lower than ground truth values
# 'absolute' - absoute difference between ground truth and modeled values
# 'squared' - squared difference between ground truth and modeled values
# Returns:
# A pandas dataframe with the following columns:
# forecast_date - date on which the forecast was made
# target_end_date - date on which the forecast is trying to predict outcomes
# target - description of what and when the model is predicting
# location - location code or 'US'
# location_name - name of state or 'US'
# type - point or quantile predictions
# quantile - what quantile prediction is being made if applicable
# expected - model output
# observed - ground truth value
# error - measure of model error
def model_evaluation(metric='Cumulative Deaths', model_name=['CovidAnalytics-DELPHI'], start_date='2021-04-27',
                     end_date='2021-01-11',
                     target=['1 wk ahead cum death'], location_list=None, pred_type='point',
                     quantile=None, error_type='percent'):
    # load location info and build dictionary
    locs = load_locations()
    location_dict_statetoid = build_location_dict(locs)

    # load ground truth values
    truth = load_truth(location_dict_statetoid, metric=metric)
    if truth is None:
        return ()

    # load and filter model predictions
    preds = load_predictions(model_name=model_name, start_date=start_date, end_date=end_date, target=target,
                             location_list=location_list, pred_type=pred_type, quantile=quantile)
    if preds is None:
        return ()

    # compare model predictions and ground truth values
    merged = merge_predictions(truth, preds, metric)
    final = calculate_prediction_errors(merged, error_type)
    if final is None:
        return ()

    return (final)


# A function that combines each of the above helper functions specialized to handle our team's predictions.
# Inputs:
# path - a path to the csv containing the predictions
# metric - what covid measure to evaluate:
# 'Cumulative Deaths' [DEFAULT]
# 'Cumulative Cases'
# 'Incident Deaths'
# 'Incident Cases'
# error_type
# 'percent' [DEFAULT] - error as a percentage of the ground truth value: (expected - observed) / observed--will be negative if predicted values are lower than ground truth values
# 'absolute' - absoute difference between ground truth and modeled values
# 'squared' - squared difference between ground truth and modeled values
# exclude_locations - a list of state names (and territories) to exclude from the results
# Returns: a pandas dataframe with prediction errors by location/model/date that can be fed into plotting functions
def mit_model_evaluation(path, metric="Cumulative Deaths",
                         error_type='percent', exclude_locations=['US']):
    # load location info and build dictionary
    locs = load_locations()
    location_dict_statetoid = build_location_dict(locs)

    # load ground truth values
    truth = load_truth(location_dict_statetoid, metric)
    if truth is None:
       return()

    # load model predictions
    mit_test_results = load_mit_model(path)
    if mit_test_results is None:
        return()

    # compare model predictions and ground truth values
    preds = calculate_prediction_errors(rearrange_predictions_mit(mit_test_results, truth, metric),
                                        errortype=error_type)

    return (preds)


# A function to build a specific table for graphing
# Can be used for both MIT modeled results and CDC modeled results
# Inputs: a pandas dataframe with predictions and errors from a set of covid models (built by model_evaluation() function)
# Outputs: a pandas dataframe with the following columns:
# model - which covid model(s) we're evaluating
# target_end_date - the date on which the predictions are targeting
# target - the name corresponding to that date as well as the metric (deaths, cases, etc.)
# error - The average error across locations
def build_plot_table(merged_data, exclude_locations=['US'], weighted = True):
    # remove rows for excluded locations
    preds = merged_data.loc[~merged_data.location_name.isin(exclude_locations), :]
    # Get rid of a possible extra column
    try:
        preds.drop(['target'])
    except:
        pass

    # for wMAPES
    def my_agg(x):
        names = {'error': (x['observed'] * x['error']).sum() / x['observed'].sum()}
        return pd.Series(names, index=['error'])

    # group by model, target end date to get [weighted] average across locations
    if weighted:
        preds = preds.groupby(['model', 'target_end_date'], as_index = False).apply(my_agg).reset_index(drop = True)
    else:
        preds = preds.groupby(['model', 'target_end_date'], as_index = False).mean()

    # select columns
    preds = preds[['model', 'target_end_date', 'error']]

    return (preds)


# A function to build a table that displays the rank of each model based on its relative performance
# Inputs:
# benchmark_table - a pandas dataframe with the errors of CDC models (from the build_plot_table() function)
# mit_table - a pandas dataframe with the errors of our team's model (from the build_plot_table() function)
# mit_model - a list of component models to include in the final ranking selected from ['agg','mdp','bilstm','knn','sir']
# Outputs:
# A pandas dataframe with models for each prediction target day ranked by error
def build_rank_table(benchmark_table, mit_table, mit_model=None):
    # our model makes predictions daily so filter out all of the dates that do not have corresponding CDC results
    benchmark_dates = benchmark_table.target_end_date.unique()
    mit_table = mit_table.loc[mit_table.target_end_date.isin(benchmark_dates), :]

    # filter to use only the component models we care about
    if mit_model is not None:
        mit_table = mit_table.loc[mit_table.model.isin(mit_model), :]

    # stack the tables then sort and rank
    combo_table = pd.concat([mit_table, benchmark_table], axis=0)
    combo_table.sort_values(by=['target_end_date', 'error'], inplace=True)
    ranks = combo_table.groupby(['target_end_date'], as_index=False).cumcount() + 1
    combo_table['rank'] = ranks
    combo_table.sort_values(by=['target_end_date', 'rank'], inplace=True)

    return (combo_table)


# A function to plot our model vs the CDC benchmarks
# Inputs:
# benchmark_table - table with errors for CDC models from the build_plot_table() function
# mit_table - table with errors for our team's model from the build_plot_table() function
# first_day - the first date for which the CDC models are making predictions in 'YYYY-MM-DD' format
# mit_model - the component model or aggregate model to compare. Must be one of ['agg','mdp','bilstm','knn','sir']
# topn - how many CDC models to highlight [DEFAULT = 5]
# filename - name the plot will be saved as
# Outputs:
# A plot
def benchmark_plot(benchmark_table, mit_table, first_day, mit_model='agg', topn=5, filename='temp.png'):
    # pick the right dates
    benchmark_dates = benchmark_table.target_end_date.unique()
    mit = mit_table.loc[mit_table.target_end_date.isin(benchmark_dates), :]
    # pick the right model
    mit = mit.loc[mit.model == mit_model, :]

    # some organizing
    bench = benchmark_table
    bench.sort_values(by=['target_end_date', 'error'], inplace=True)
    ranks = bench.groupby(['target_end_date'], as_index=False).cumcount() + 1
    bench['rank'] = ranks

    # formatting
    fig, ax = plt.subplots(figsize=(20, 10), nrows=1, sharex=True)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(15)

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(15)

    # Where to start the plot
    temp = bench.loc[bench.target_end_date == first_day, :]
    # split into the best models (to be highlighted) and the less good models
    low_ranks = temp.loc[temp['rank'] <= topn, :]
    high_ranks = temp.loc[temp['rank'] > topn, :]
    good_models = low_ranks.model.unique()
    bad_models = high_ranks.model.unique()

    # plot a line for each model
    models = bench.model.unique()
    for m in bad_models:
        small = bench.loc[bench.model == m, ['model', 'target_end_date', 'error', 'rank']]
        model_name = small.model.tolist()[0]
        plt.plot(small.target_end_date, small.error, color='#d3d3d3')
        # , label = f"({model_name})")

    for m in good_models:
        small = bench.loc[bench.model == m, ['model', 'target_end_date', 'error', 'rank']]
        model_name = small.model.tolist()[0]
        plt.plot(small.target_end_date, small.error, color='firebrick')

    # plot a line for our team's model
    plt.plot(mit.target_end_date, mit.error, label=mit_model, color='black', linewidth=3)

    # legend
    custom_lines = [plt.Line2D([0], [0], color='black', lw=4),
                    plt.Line2D([0], [0], color='firebrick', lw=3),
                    plt.Line2D([0], [0], color='#d3d3d3', lw=3)]
    ax.grid(True)
    plt.legend(loc='upper left', prop={'size': 20}, handles=custom_lines,
               labels=[mit_model + ' Model', 'Top 5 Benchmark Models', 'Other Benchmark Models'])
    # plt.title("{} prediction, {}".format(target_col, state_name))
    # more formatting
    plt.ylabel("wMAPE", fontsize=20)
    plt.xlabel("Prediction Target Date", fontsize=20)
    # plt.show()
    plt.savefig('../Figures/' + filename)


#####################################################################
#####################################################################
## K, let's run it now ##
#####################################################################
#####################################################################

if month == 'June':
    start_date = '2020-05-31'
    end_date = '2020-06-01'
    first_day = '2020-06-06'
elif month == 'September':
    start_date = '2020-08-30'
    end_date = '2020-08-31'
    first_day = '2020-09-05'
elif month == 'November':
    start_date = '2020-11-01'
    end_date = '2020-11-02'
    first_day = '2020-11-07'

if metric == 'Cases':
    mit_metric = 'Cumulative Cases'
    cdc_metric = 'Incident Cases'
    target = ['1 wk ahead inc case','2 wk ahead inc case','3 wk ahead inc case','4 wk ahead inc case','5 wk ahead inc case']
else:
    mit_metric = 'Cumulative Deaths'
    cdc_metric = 'Cumulative Deaths'
    target = ['1 wk ahead cum death','2 wk ahead cum death','3 wk ahead cum death','4 wk ahead cum death','5 wk ahead cum death']

# Benchmark results
cdc_table = build_plot_table(model_evaluation(#model_name = ['CovidAnalytics-DELPHI'],
                           model_name = full_model_list,
                           metric = cdc_metric, start_date = start_date, end_date = end_date,
                           target = target,
                           pred_type = 'point', error_type = 'percent',
                           location_list = ['Alabama','Alaska','Arizona','Arkansas','California','Colorado','Connecticut','Delaware',
                                           'Florida','Georgia','Hawaii','Idaho','Illinois','Indiana','Iowa','Kansas','Kentucky','Louisiana',
                                           'Maine','Maryland','Massachusetts','Michigan','Minnesota','Mississippi','Missouri','Montana',
                                           'Nebraska','Nevada','New Hampshire','New Jersey','New Mexico','New York','North Carolina','North Dakota',
                                           'Ohio','Oklahoma','Oregon','Pennsylvania','Rhode Island','South Carolina','South Dakota','Tennessee',
                                           'Texas','Utah','Vermont','Virginia','Washington','West Virginia','Wisconsin','Wyoming']),
                            exclude_locations = exclude_states)
cdc_table = cdc_table.loc[~cdc_table.model.isin(bad_models),:]

# MIT results
mit_table = build_plot_table(mit_model_evaluation(path = path_to_mit_results, metric = mit_metric),
                              exclude_locations = exclude_states)


# Plot the results
benchmark_plot(cdc_table, mit_table, mit_model = model_to_plot, first_day = first_day, filename = plot_filename)

# Rank the models
rank_table = build_rank_table(cdc_table, mit_table, mit_model = models_to_rank)
rank_table.to_csv(rank_table_path, index = False)