# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 21:24:25 2020

@author: omars
"""

if __name__ == "__main__":
    #%% Libraries and Parameters

    from data_utils import (save_model, load_model, load_data, dict_to_df)
    from params import (train_sir, train_bilstm, train_knn, train_mdp, train_agg, train_ci,
                        date_col, region_col, target_col, sir_file, knn_file, mdp_file, agg_file, ci_file,
                        validation_cutoff, training_cutoff, training_agg_cutoff,
                        per_region, ml_methods, ml_mapping, ml_hyperparams, ci_range,
                        knn_params_dict, sir_params_dict, mdp_params_dict, bilstm_params_dict,retrain,
                        train_mdp_agg, train_sir_agg, train_bilstm_agg, train_knn_agg,
                        train_bilstm_ret, train_knn_ret, train_mdp_ret, train_sir_ret,
                        load_sir, load_knn, load_bilstm, load_mdp, load_sir_agg, load_knn_agg, load_mdp_agg, load_bilstm_agg, load_agg,
                        train_preval, tests_col, population,
                        preval_file, bilstm_file, alpha)

    from sir_model import SIRModel
    from bilstm_model import BILSTMModel
    from knn_model import KNNModel
    from mdp_model import MDPModel
    from agg_model import AGGModel
    from confidence_intervals import CI
    from prevalence import PrevalenceModel
    import warnings
    import pandas as pd
    import pickle
    # warnings.filterwarnings("ignore")

    #%% Load Data

    df, _, df_train, df_validation = load_data(validation_cutoff=validation_cutoff)
    _, _, df_train_agg, df_validation_agg = load_data(training_cutoff=training_agg_cutoff,
                                                   validation_cutoff=training_cutoff)
    _, _, df_train_ret, df_validation_ret = load_data(training_cutoff=validation_cutoff,
                                                   validation_cutoff=None)

    #%% Train and Save Models

    models = []
    regions_val = list(set(df_validation[region_col]))
    dates_val = sorted(list(set(df_validation[date_col])))
    validation_predictions = {}

    if train_bilstm:
        bilstm = BILSTMModel(**bilstm_params_dict)
        bilstm.fit(df_train, 'first')
        models.append('bilstm')
        save_model(bilstm, bilstm_file.replace(".pickle", "_train.pickle"))
    if load_bilstm:
        bilstm = load_model(bilstm_file.replace(".pickle", "_train.pickle"))
        validation_predictions['bilstm'] = bilstm.predict(regions_val, dates_val, 'first')
        models.append('bilstm')

    if train_sir:
        sir = SIRModel(**sir_params_dict)
        sir.fit(df_train)
        save_model(sir, sir_file.replace(".pickle", "_train.pickle"))

    if load_sir:
        sir = load_model(sir_file.replace(".pickle", "_train.pickle"))
        try:
            validation_predictions['sir'] = sir.predict(regions_val, dates_val)
        except:
            pass
        models.append('sir')

    if train_knn:
        knn = KNNModel(**knn_params_dict)
        knn.fit(df_train)
        models.append('knn')
        save_model(knn, knn_file.replace(".pickle", "_train.pickle"))
    if load_knn:
        knn = load_model(knn_file.replace(".pickle", "_train.pickle"))
        try:
            validation_predictions['knn'] = knn.predict(regions_val, dates_val)
        except:
            pass
        models.append('knn')

    if train_mdp:
        mdp = MDPModel(**mdp_params_dict)
        mdp.fit(df_train)
        models.append('mdp')
        save_model(mdp, mdp_file.replace(".pickle", "_train.pickle"))
    if load_mdp:
        mdp = load_model(mdp_file.replace(".pickle", "_train.pickle"))
        try:
            validation_predictions['mdp'] = mdp.predict(regions_val, dates_val)
        except:
            pass
        models.append('mdp')

    if load_sir * load_bilstm * load_knn * load_mdp:
        # convert predictions as a dataframe and save
        df_tmp = dict_to_df(validation_predictions,
                            df_validation)
        df_tmp.to_csv("../results/prediction_intermediate_{}_{}.csv".format(str(min(dates_val))[:10], str(max(dates_val))[:10]))

        with open("../results/prediction_intermediate_{}_{}.pickle".format(str(min(dates_val))[:10], str(max(dates_val))[:10]), "wb") as file:
            pickle.dump(validation_predictions, file)

    if train_agg:

        models_agg = []
        validation_predictions_agg = {}
        regions_agg = list(set(df_validation_agg[region_col]))
        dates_agg = sorted(list(set(df_validation_agg[date_col])))

        # train SEIRD
        if train_sir_agg:
            sir_agg = SIRModel(**sir_params_dict)
            sir_agg.fit(df_train_agg)
            save_model(sir_agg, sir_file.replace(".pickle", "_agg.pickle"))
        if load_sir_agg:
            sir_agg = load_model(sir_file.replace(".pickle", "_agg.pickle"))
            validation_predictions_agg['sir'] = sir_agg.predict(regions_agg, dates_agg)
            models_agg.append('sir')

        # train kNN
        if train_knn_agg:
            knn_agg = KNNModel(**knn_params_dict)
            knn_agg.fit(df_train_agg)
            save_model(knn_agg, knn_file.replace(".pickle", "_agg.pickle"))
        if load_knn_agg:
            knn_agg = load_model(knn_file.replace(".pickle", "_agg.pickle"))
            validation_predictions_agg['knn'] = knn_agg.predict(regions_agg, dates_agg)
            models_agg.append('knn')

        # train bi lstm
        if train_bilstm_agg:
            bilstm_agg = BILSTMModel(**bilstm_params_dict)
            bilstm_agg.fit(df_train_agg,'agg')
            save_model(bilstm_agg, bilstm_file.replace(".pickle", "_agg.pickle"))
        if load_bilstm_agg:
            bilstm_agg = load_model(bilstm_file.replace(".pickle", "_agg.pickle"))
            validation_predictions_agg['bilstm'] = bilstm_agg.predict(regions_agg, dates_agg, 'agg')
            models_agg.append('bilstm')

        # train MDP
        if train_mdp_agg:
            mdp_agg = MDPModel(**mdp_params_dict)
            mdp_agg.fit(df_train_agg)
            save_model(mdp_agg, mdp_file.replace(".pickle", "_agg.pickle"))
        if load_mdp_agg:
            mdp_agg = load_model(mdp_file.replace(".pickle", "_agg.pickle"))
            validation_predictions_agg['mdp'] = mdp_agg.predict(regions_agg, dates_agg)
            models_agg.append('mdp')

        if load_sir_agg * load_bilstm_agg * load_knn_agg * load_mdp_agg:
            df_agg = dict_to_df(validation_predictions_agg,
                                df_validation_agg)

            df_agg.to_csv("../results/prediction_aggregate_{}_{}.csv".format(str(min(dates_agg))[:10], str(max(dates_agg))[:10]))

        df_agg = pd.read_csv("../results/prediction_aggregate_{}_{}.csv".format(str(min(dates_agg))[:10], str(max(dates_agg))[:10]),
                             index_col=0, parse_dates=["date"])

        agg = AGGModel(date=date_col,
                       region=region_col,
                       target=target_col,
                       models=["sir", "knn", "bilstm", "mdp"],
                       per_region=per_region,
                       ml_methods=ml_methods,
                       ml_mapping=ml_mapping,
                       ml_hyperparams=ml_hyperparams)
        agg.fit(df_agg)
        save_model(agg, agg_file)

    if load_agg:
        with open("../results/prediction_intermediate_{}_{}.pickle".format(str(min(dates_val))[:10], str(max(dates_val))[:10]), "rb") as file:
            validation_predictions = pickle.load(file)
        agg = load_model(agg_file)
        validation_predictions['agg'] = agg.predict(regions_val, dates_val, validation_predictions)
        models.append('agg')

        df_agg = dict_to_df(validation_predictions,
                            df_validation)
        df_agg.to_csv("../results/prediction_aggregate_{}_{}_with_agg.csv".format(str(min(dates_val))[:10], str(max(dates_val))[:10]))

    if train_ci:
        prediction_train = pd.read_csv("../results/prediction_aggregate_{}_{}_with_agg.csv".format(str(min(dates_val))[:10], str(max(dates_val))[:10]),
                             index_col=0, parse_dates=["date"])
        prediction_train.dropna(subset=["agg"], inplace=True)
        ci = CI(region_col=region_col,
                target_col=target_col,
                ci_range=ci_range,
                models=models)
        ci.fit(prediction_train)
        save_model(ci, ci_file)

    if train_preval:
        preval = PrevalenceModel(region_col=region_col,
                                 date_col=date_col,
                                 tests_col=tests_col,
                                 population_col=population,
                                 alpha=alpha)
        preval.fit(df_train)
        save_model(preval, preval_file)

    if retrain:
        if train_sir_ret:
            sir = SIRModel(**sir_params_dict)
            sir.fit(df_train_ret)
            save_model(sir, sir_file)

        if train_bilstm_ret:
            bilstm = BILSTMModel(**bilstm_params_dict)
            bilstm.fit(df_train_ret)
            save_model(bilstm, bilstm_file)

        if train_knn_ret:
            knn = KNNModel(**knn_params_dict)
            knn.fit(df_train_ret)
            save_model(knn, knn_file)

        if train_mdp_ret:
            mdp = MDPModel(**mdp_params_dict)
            mdp.fit(df_train_ret)
            save_model(mdp, mdp_file)

        if train_preval:
            preval = PrevalenceModel(region_col=region_col,
                                     date_col=date_col,
                                     tests_col=tests_col,
                                     population_col=population,
                                     alpha=alpha)
            preval.fit(df)
            save_model(preval, preval_file)
