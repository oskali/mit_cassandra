# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 21:24:44 2020

@author: omars
"""

#%% Libraries

import pickle
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from params import (target_col, date_col, region_col, training_cutoff,
                         df_path, default_path, nmin, restriction_dict, region_exceptions_dict)
import os
import warnings
warnings.filterwarnings("ignore")

#%% Helper Functions

def save_model(model,
               filename):
    file_pi = open(filename, 'wb')
    pickle.dump(model, file_pi)

def load_model(filename):
    filehandler = open(filename, 'rb')
    return(pickle.load(filehandler))

def load_data(file=df_path,
              target=target_col,
              date=date_col,
              region=region_col,
              training_cutoff=training_cutoff,
              validation_cutoff=None,
              nmin=nmin,
              restriction_dict=restriction_dict[region_col],
              region_exceptions=region_exceptions_dict[region_col],
              default_path=default_path):
    if file is None:
        df = get_public_data(default_path)
    else:
        df = pd.read_csv(file)
    df.columns = map(str.lower, df.columns)

    # restrict to a subset of obervations
    if not (restriction_dict is None):
        masks = []
        for col, values in restriction_dict.items():
            try:
                masks.append(df[col].isin(values))
            except:
                pass
        if masks:
            mask_ = masks.pop(0)
            for other_mask in masks:
                mask_ = (mask_ | other_mask)
            df = df[mask_].copy()

    # delete excepctions
    if not (region_exceptions is None):
        df = df[~df[region].isin(region_exceptions)].copy()

    df = df[df[target] >= nmin[region]]

    df.sort_values(by=[region, date], inplace=True)
    try:
        df["cases_nom"] = df["cases"] / df["population"]
        df["deaths_nom"] = df["deaths"] / df["population"]
    except KeyError:
        pass
    df["cases_pct3"] = df.groupby(region)["cases"].pct_change(3).values
    df["cases_pct5"] = df.groupby(region)["cases"].pct_change(5).values
    try:
        df[date] = df[date].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    except:
        df[date] = df[date].apply(lambda x: datetime.strptime(x, '%m/%d/%Y'))
    df = df.sort_values(by=[region, date])
    df_train = df[df[date] <= training_cutoff]
    print("Training set contains {} {}.".format(df[region].nunique(), region))
    if validation_cutoff is None:
        df_test = df[df[date] > training_cutoff]
    else:
        df_test = df[[a and b for a, b in zip(df[date] > training_cutoff, df[date] <= validation_cutoff)]]
        df = df[df[date] <= validation_cutoff]
    return(df, df_train, df_test)

def dict_to_df(output,
               df_validation,
               region_col=region_col,
               date_col=date_col,
               target_col=target_col):
    models = list(output.keys())
    regions = list(set(df_validation[region_col]))
    dates = list(set(df_validation[date_col]))
    predictions_rows = []
    for region in regions:
        for date in dates:
            prediction = [region, date]
            for model in models:
                if region in output[model].keys():
                    try:
                        prediction.append(output[model][region].loc[date])
                    except:
                        prediction.append(np.nan)
                else:
                    prediction.append(np.nan)
            predictions_rows.append(prediction)
    df_predictions = pd.DataFrame(predictions_rows, columns=[region_col, date_col] + models)
    df_agg = df_predictions.merge(df_validation.loc[:, [region_col, date_col, target_col]], how='left', on=[region_col, date_col])
    return df_agg

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))

def get_mapes(df,
              models,
              region_col='state',
              target_col='cases'):

    results = []
    for region in set(df[region_col]):
        df_sub = df[df[region_col] == region]
        results.append([region] + [mape(df_sub[target_col], df_sub[model]) for model in models])
    results.append(['Average'] + [mape(df[target_col], df[model]) for model in models])
    return(pd.DataFrame(results, columns=[region_col] + ['MAPE_' + model for model in models]))

def get_public_data(path=df_path):
    # Import the latest data using the raw data urls
    meas_url = 'https://raw.githubusercontent.com/COVID19StatePolicy/SocialDistancing/master/data/USstatesCov19distancingpolicy.csv'
    case_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv'
    deaths_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv'
    mob_url = 'https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv'
    measures = pd.read_csv(meas_url,
                           encoding="ISO-8859-1")
    cases = pd.read_csv(case_url,
                        encoding='utf-8')
    deaths = pd.read_csv(deaths_url,
                         encoding='utf-8')
    mobility = pd.read_csv(mob_url,
                           encoding='utf-8')
    #John Hopkins University daily reports
    last_available_date = (datetime.today() - timedelta(1)).strftime('%Y-%m-%d')
    dates = pd.date_range(start='2020-04-12', end=datetime.today() - timedelta(3)).strftime('%m-%d-%Y').tolist()
    daily_df = pd.DataFrame()
    for date in dates:
        daily_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports_us/' + date + '.csv'
        this_daily = pd.read_csv(daily_url,encoding='utf-8')
        this_daily['date'] = date  # fill the date column with the date from the file name
        daily_df = daily_df.append(this_daily)
    daily_df.drop(['Country_Region', 'Last_Update', 'Lat', 'Long_', 'UID', 'ISO3'], axis=1, inplace=True)
    daily_df.columns = daily_df.columns.str.replace(
        'Province_State', 'state').str.replace(
        'Confirmed','cases').str.replace(
        'Deaths', 'deaths')
    daily_df['date'] = pd.to_datetime(daily_df['date'], format="%m-%d-%Y")
    # Make sure cases and deaths have the same number of rows
    try:
        assert cases.shape[0] == deaths.shape[0]
    except AssertionError:
        print("Different number of rows in cases and deaths dataframes")
        print(cases.shape)
        print(deaths.shape)
    # keep only US mobility data
    mobility_us = mobility[mobility.country_region_code == 'US']
    # make a new df with no missing values in state column, where sub_region_1=states, sub_region_2 = counties
    mobility_counties = mobility_us[mobility_us['sub_region_1'].notna()]
    mobility_states = mobility_counties[mobility_counties['sub_region_2'].isna()]
    # drop columns with countries and counties; Show list of unique states in final df
    mobility_states.drop(
        ['country_region_code', 'country_region', 'sub_region_2', 'iso_3166_2_code', 'census_fips_code'], axis=1,
        inplace=True)
    # Check that we have the right number of states
    mob_states = sorted(mobility_states.sub_region_1.unique())
    try:
        assert len(mob_states) == 51
    except AssertionError:
        print("Number of states in mobility data is not 51")
        print(len(mob_states))
        print(mob_states)

    measures[['DateIssued', 'DateEnacted', 'DateExpiry', 'DateEased', 'DateEnded']] = measures[
        ['DateIssued', 'DateEnacted', 'DateExpiry', 'DateEased', 'DateEnded']].apply(pd.to_datetime, format="%Y%m%d")
    # Remove suffix from mobility column names
    mobility_states.columns = [col.replace('_percent_change_from_baseline', '') for col in mobility_states.columns]
    # check for any missing or misspelled states before joining cases, measures and mobility
    mset = set(mobility_states.loc[:, 'sub_region_1'])
    cset = set(cases.loc[:, 'Province_State'])
    pset = set(measures.loc[:, 'StateName'])
    dset = set(daily_df.loc[:, 'state'])
    daily_diff = dset - cset
    # emove state rows that are not in cases df
    daily_df = daily_df[~daily_df.state.isin(list(daily_diff))]
    #check for the same number of states in daily_df and cases df
    try:
        assert len(daily_df.state.unique()) == len(cset)
    except AssertionError:
        print("Number of states in daily_df and in cases df is not the same")
        print(len(daily_df.state.unique()))
        print(len(cset))
    # Select columns from measures df to merge with cases and deaths dfs
    measures = measures[~measures.StateFIPS.isnull()]  # drop rows with empty StateFIPS values
    # Select columns in measures for modeling
    meas_sel = measures[['StateName', 'StatePolicy', 'DateEnacted', 'DateEnded']]
    # drop columns not used in models
    cases.drop(['iso2', 'iso3', 'Country_Region', 'Combined_Key', 'UID', 'code3', 'FIPS', 'Lat', 'Long_'], axis=1,
               inplace=True)
    deaths.drop(['iso2', 'iso3', 'Country_Region', 'Combined_Key', 'UID', 'code3', 'FIPS', 'Lat', 'Long_'], axis=1,
                inplace=True)
    # Reshape cases and deaths df from wide to tall format
    c_melt = cases.melt(id_vars=['Province_State', 'Admin2'], var_name='date', value_name='cases')
    d_melt = deaths.melt(id_vars=['Province_State', 'Admin2', 'Population'], var_name='date', value_name='deaths')
    # merge cases and deaths df on state and date columns
    case_death_df = pd.merge(c_melt, d_melt, how='left', on=['Province_State', 'Admin2', 'date'])
    # convert date colum from str to date
    case_death_df['date'] = pd.to_datetime(case_death_df['date'], format="%m/%d/%y")
    # Drop rows with Population = 0 (correction, out-of-state, unassigned)
    tmp = case_death_df[case_death_df['Population'] != 0]
    # get total state Population column after grouping each state on an arbitrary date
    pop = tmp.loc[tmp.date == '2020-05-08'].groupby(['Province_State'], as_index=False)[['Population']].sum()
    # Group cases and death data by state
    cd_state = case_death_df.groupby(['Province_State', 'date'], as_index=False)[['cases', 'deaths']].sum()
    # Merge Population column
    cdp_state = pd.merge(cd_state, pop, how='left', left_on=['Province_State'], right_on=['Province_State'])
    #Add measures categorical columns
    # Add columns with 0s for each measure to the main df
    dfzeros = pd.DataFrame(np.zeros((len(cdp_state), len(meas_sel.StatePolicy.unique()))),
                           columns=list(meas_sel.StatePolicy.unique())).astype(int)
    tseries = pd.concat([cdp_state, dfzeros], axis=1)
    tseries.columns = tseries.columns.str.replace('Province_State', 'state').str.replace('Admin2', 'county')
    # Loop over states and measures. Plug  1s in the rows when measures were enacted, leave the rest as 0s
    for state in meas_sel.StateName.unique():
        for i, meas in enumerate(meas_sel.StatePolicy.unique()):
            # select rows by state and  measure
            mask1 = (meas_sel.StateName == state) & (meas_sel.StatePolicy == meas)
            if not meas_sel[mask1].empty:
                # date policy enacted
                start = meas_sel.loc[mask1, "DateEnacted"].values[0]
                # date policy ended
                end = meas_sel.loc[mask1, "DateEnded"].values[0]
            else:
                # print(state+ " is missing " + meas)
                continue
            if pd.notnull(start) & pd.notnull(end):
                mask2 = (tseries.state == state) & (tseries.date >= start) & (tseries.date <= end)
            elif pd.notnull(start):
                mask2 = (tseries.state == state) & (tseries.date >= start)
            else:
                continue
            # set measure values to 1 after date was enacted by state
            tseries.loc[mask2, meas] = 1
    #Merge mobility and columns from daily_df reports
    tseries['date'] = tseries['date'].dt.strftime('%Y-%m-%d')
    df = pd.merge(tseries, mobility_states, how='left', left_on=['state', 'date'], right_on=['sub_region_1', 'date'])
    # Drop duplicate state column after merge
    df.drop(['sub_region_1'], axis=1, inplace=True)
    # Select columns from daily_df and convert date to string to merge to final df
    daily_df.drop(['FIPS', 'cases', 'deaths'], axis=1, inplace=True)
    daily_df['date'] = daily_df['date'].dt.strftime('%Y-%m-%d')
    # Merge testing (daily_df) to df
    df_final = pd.merge(df, daily_df, on=["date", "state"], how="left")
    df_final.columns = df_final.columns.str.lower()
    # Export df_final to csv
    pathstr = os.path.split(path)
    datestr = datetime.now().strftime('%m_%d_%Y')
    df_final.to_csv(os.path.join(pathstr[0], datestr + '_states_combined.csv'))

    return df_final
