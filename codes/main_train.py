# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 21:24:25 2020

@author: omars
"""

if __name__ == "__main__":
    #%% Libraries and Parameters

    from data_utils import (save_model, load_model, load_data, dict_to_df, get_mapes,  get_last_day_mapes)
    from params import (df_path,infection_period, severe_infections, new_cases, train_sir, train_bilstm, train_knn, train_mdp, train_agg, train_ci,
                        date_col, region_col, target_col, sir_file, knn_file, mdp_file, agg_file, ci_file,
                        validation_cutoff, training_cutoff, training_agg_cutoff,
                        per_region, ml_methods, ml_mapping, ml_hyperparams, ci_range,
                        knn_params_dict, sir_params_dict, mdp_params_dict, bilstm_params_dict,
                        retrain, train, batcktest, test,
                        train_mdp_agg, train_sir_agg, train_bilstm_agg, train_knn_agg,
                        train_bilstm_ret, train_knn_ret, train_mdp_ret, train_sir_ret,
                        load_preval, load_sir, load_knn, load_bilstm, load_mdp, load_sir_agg, load_knn_agg, load_mdp_agg, load_bilstm_agg, load_agg,
                        train_preval, tests_col, population, load_ci, out_of_sample_dates,
                        # NEW
                        train_rmdp, train_rmdp_agg, train_rmdp_ret, load_rmdp, load_rmdp_agg, rmdp_params_dict, rmdp_file,

                        preval_file, bilstm_file, alpha, result_path, model_path, n_samples, random_state, load_new_cases)
    from uszipcode import SearchEngine
    from sir_model import SIRModel
    from bilstm_model import BILSTMModel
    from knn_model import KNNModel
    from mdp_model import MDPModel
    from randMDPModel.rmdp_model import rMDPModel
    from agg_model import AGGModel
    from confidence_intervals import CI
    from prevalence import PrevalenceModel
    import warnings
    import pandas as pd
    import pickle
    import os
    from datetime import datetime
    import json
    # warnings.filterwarnings("ignore")

    # STEP 1 : Load the datatest
    print(
f"""

BEGINNING OF THE TRAINING
#########################

-------------------------
1) Loading of the data :

    target : {target_col}
    region : {region_col}
    
    Chosen cutoffs: 

        - training_agg_cutoff : {training_cutoff}
        - training_cutoff : {training_agg_cutoff}
        - validation_cutoff : {validation_cutoff}

    Additional information >>>

        - features MDP : 
            {mdp_params_dict}

    Loading datasets : running ...

""")
    #%% Load Data

    df, _, df_train, df_validation = load_data(validation_cutoff=validation_cutoff)
    _, _, df_train_agg, df_validation_agg = load_data(training_cutoff=training_agg_cutoff,
                                                   validation_cutoff=training_cutoff)
    _, _, df_train_ret, df_validation_ret = load_data(training_cutoff=validation_cutoff,
                                                   validation_cutoff=None)

    print(
f"""
    Loading datasets : done.

""")

    # %% Train and Save Models
    if train:
        print(
f"""

-------------------------------
2) Training and Loading models (TRAIN) :

    Training and predictions : running ...

""")

        models = []
        regions_train_agg = list(set(df_train_agg[region_col]))
        regions_train = list(set(df_train[region_col]))
        regions_val = list(set(df_validation[region_col]))
        dates_val = sorted(list(set(df_validation[date_col])))
        validation_predictions = {}

        if train_bilstm:
            now = datetime.now()
            print(f"""
    ({now.strftime("%d/%m/%Y %H:%M:%S")}) Training Bi-LSTM : running ...""")
            bilstm = BILSTMModel(**bilstm_params_dict)
            bilstm.fit(df_train, 'first')
            
            now = datetime.now()
            print(f"""
    ({now.strftime("%d/%m/%Y %H:%M:%S")}) Training Bi-LSTM : done.""")
            models.append('bilstm')
            save_model(bilstm, bilstm_file.replace(".pickle", "_train.pickle"))
            print(f""" 
    bi-lstm model saved.""")
    
        if load_bilstm:
            now = datetime.now()
            print(f"""
    ({now.strftime("%d/%m/%Y %H:%M:%S")}) Predictions Bi-LSTM : running ...""")
            bilstm = load_model(bilstm_file.replace(".pickle", "_train.pickle"))
            validation_predictions['bilstm'] = bilstm.predict(regions_val, dates_val, 'first')
            now = datetime.now()
            print(f"""
    ({now.strftime("%d/%m/%Y %H:%M:%S")}) Predictions Bi-LSTM : done.""")
            models.append('bilstm')
            
        if train_knn:
            now = datetime.now()
            print(f"""
    ({now.strftime("%d/%m/%Y %H:%M:%S")}) Training Cassandra : running ...""")
            knn = KNNModel(**knn_params_dict)
            knn.fit(df_train, regions_train)
            now = datetime.now() 
            print(f"""
    ({now.strftime("%d/%m/%Y %H:%M:%S")}) Training Cassandra : done.""")
            models.append('knn')
            save_model(knn, knn_file.replace(".pickle", "_train.pickle"))
            print(f"""
    seird model saved.""")
        if load_knn:
            now = datetime.now()
            print(f"""
    ({now.strftime("%d/%m/%Y %H:%M:%S")}) Predictions Cassandra : running ...""")
            knn = load_model(knn_file.replace(".pickle", "_train.pickle"))
            try:
                validation_predictions['knn'] = knn.predict(regions_val, dates_val)
            except:
                pass
            now = datetime.now()
            print(f"""
    ({now.strftime("%d/%m/%Y %H:%M:%S")}) Predictions Cassandra : done.""")
            models.append('knn')
        
        if train_mdp:
            # df = df[df[target] >= 1]
            now = datetime.now()
            print(f"""
    ({now.strftime("%d/%m/%Y %H:%M:%S")}) Training MDP : running ...""")
            mdp = MDPModel(**mdp_params_dict)
            # mdp.fit(df_train[df_train['cases'] >= 200])
            mdp.fit(df_train)
            now = datetime.now() 
            print(f"""
    ({now.strftime("%d/%m/%Y %H:%M:%S")}) Training MDP : done.""")
            models.append('mdp')
            save_model(mdp, mdp_file.replace(".pickle", "_train.pickle"))
            print(f"""
    mdp model saved.""")
        if load_mdp:
            now = datetime.now()
            print(f"""
    ({now.strftime("%d/%m/%Y %H:%M:%S")}) Predictions MDP : running ...""")
            mdp = load_model(mdp_file.replace(".pickle", "_train.pickle"))
            try:
                validation_predictions['mdp'] = mdp.predict(regions_val, dates_val)
            except:
                pass
            now = datetime.now()
            print(f"""
    ({now.strftime("%d/%m/%Y %H:%M:%S")}) Predictions MDP : done.""")
            models.append('mdp')

        if train_rmdp:
            # df = df[df[target] >= 1]
            now = datetime.now()
            print(f"""
    ({now.strftime("%d/%m/%Y %H:%M:%S")}) Training rMDP : running ...""")
            rmdp = rMDPModel(**rmdp_params_dict)
            # rmdp.fit(df_train[df_train['cases'] >= 200])
            mode = "TIME_CV"
            rmdp.fit(df_train, mode=mode)
            now = datetime.now() 
            print(f"""
    ({now.strftime("%d/%m/%Y %H:%M:%S")}) Training rMDP : done.""")
            models.append('rmdp')
            save_model(rmdp, rmdp_file.replace(".pickle", "_train.pickle"))
            print(f"""
    rmdp model saved.""")
        if load_rmdp:
            now = datetime.now()
            print(f"""
    ({now.strftime("%d/%m/%Y %H:%M:%S")}) Predictions rMDP : running ...""")
            rmdp = load_model(rmdp_file.replace(".pickle", "_train.pickle"))
            rmdp.verbose = 1
            try:
                validation_predictions['rmdp'] = rmdp.predict(regions_val, dates_val)
            except:
                pass
            now = datetime.now()
            print(f"""
    ({now.strftime("%d/%m/%Y %H:%M:%S")}) Predictions rMDP : done.""")
            models.append('rmdp')

        if train_sir:
            now = datetime.now()
            print(f"""
({now.strftime("%d/%m/%Y %H:%M:%S")}) Training SEIRD : running ...""")
            sir = SIRModel(**sir_params_dict)
            sir.fit(df_train)

            now = datetime.now()
            print(f"""
({now.strftime("%d/%m/%Y %H:%M:%S")}) Training Bi-LSTM : done.""")
            save_model(sir, sir_file.replace(".pickle", "_train.pickle"))
            print(f""" 
        seird model saved.""")

        if load_sir:
            now = datetime.now()
            print(f"""
    ({now.strftime("%d/%m/%Y %H:%M:%S")}) Predictions SEIRD : running ...""")
            sir = load_model(sir_file.replace(".pickle", "_train.pickle"))
            try:
                validation_predictions['sir'] = sir.predict(regions_val, dates_val)
            except:
                pass
            now = datetime.now()
            print(f"""
    ({now.strftime("%d/%m/%Y %H:%M:%S")}) Predictions SEIRD : done.""")
            models.append('sir')


        print(f"""
    Training and predictions (~AGG) : done.
    
    """)
        if load_bilstm + load_knn + load_mdp + load_rmdp >= 3:
        
            print(f"""
    Additional saving of the predicitons (~AGG) as a csv.""")

            # convert predictions as a dataframe and save
            df_tmp = dict_to_df(validation_predictions,
                            df_validation)

            print(df_tmp)
            df_tmp.to_csv(
                os.path.join(result_path, 
                "prediction_intermediate_{}_{}_{}_{}.csv".format(region_col, target_col, str(min(dates_val))[:10], str(max(dates_val))[:10])))

            with open(
            os.path.join(result_path, 
            "prediction_intermediate_{}_{}_{}_{}.pickle".format(region_col, target_col, str(min(dates_val))[:10], str(max(dates_val))[:10])), "wb") as file:
                pickle.dump(validation_predictions, file)

    if train_agg:
        print(f"""
    -------------------------------
    2-bis) Training and Loading models (AGG) :

        Training and predictions : running ...

    """)
        models_agg = []
        validation_predictions_agg = {}
        regions_agg = list(set(df_validation_agg[region_col]))
        dates_agg = sorted(list(set(df_validation_agg[date_col])))

        # train SEIRD
        if train_sir_agg:

            now = datetime.now()
            print(f"""
        ({now.strftime("%d/%m/%Y %H:%M:%S")}) Training SEIRD (AGG) : running ...""")
            sir_agg = SIRModel(**sir_params_dict)
            sir_agg.fit(df_train_agg)
            now = datetime.now() 
            print(f"""
        ({now.strftime("%d/%m/%Y %H:%M:%S")}) Training SEIRD (AGG) : done.""")
            save_model(sir_agg, sir_file.replace(".pickle", "_agg.pickle"))
            print(f"""
        seird model saved.""")

        if load_sir_agg:

            now = datetime.now()
            print(f"""
        ({now.strftime("%d/%m/%Y %H:%M:%S")}) Predictions SEIRD (AGG) : running ...""")
            sir_agg = load_model(sir_file.replace(".pickle", "_agg.pickle"))
            validation_predictions_agg['sir'] = sir_agg.predict(regions_agg, dates_agg)

            now = datetime.now() 
            print(f"""
        ({now.strftime("%d/%m/%Y %H:%M:%S")}) Predictions SEIRD (AGG) : done.""")
            models_agg.append('sir')

        # train kNN
        if train_knn_agg:
            now = datetime.now()
            print(f"""
        ({now.strftime("%d/%m/%Y %H:%M:%S")}) Training Cassandra (AGG) : running ...""")
            knn_agg = KNNModel(**knn_params_dict)
            knn_agg.fit(df_train_agg, regions_train_agg)

            now = datetime.now() 
            print(f"""
        ({now.strftime("%d/%m/%Y %H:%M:%S")}) Training Cassandra (AGG) : done.""")
            save_model(knn_agg, knn_file.replace(".pickle", "_agg.pickle"))
            print(f"""
        Cassandra model saved.""")

        if load_knn_agg:
            now = datetime.now()
            print(f"""
        ({now.strftime("%d/%m/%Y %H:%M:%S")}) Predictions Cassandra (AGG) : running ...""")
            knn_agg = load_model(knn_file.replace(".pickle", "_agg.pickle"))
            validation_predictions_agg['knn'] = knn_agg.predict(regions_agg, dates_agg)
            now = datetime.now() 
            print(f"""
        ({now.strftime("%d/%m/%Y %H:%M:%S")}) Predictions Cassandra (AGG) : done.""")
            models_agg.append('knn')

        # train BI LSTM
        if train_bilstm_agg:
            now = datetime.now()
            print(f"""
        ({now.strftime("%d/%m/%Y %H:%M:%S")}) Training Bi-LSTM (AGG) : running ...""")
            bilstm_agg = BILSTMModel(**bilstm_params_dict)
            bilstm_agg.fit(df_train_agg,'agg')
            now = datetime.now() 
            print(f"""
        ({now.strftime("%d/%m/%Y %H:%M:%S")}) Training Bi-LSTM (AGG) : done.""")
            save_model(bilstm_agg, bilstm_file.replace(".pickle", "_agg.pickle"))
            print(f"""
        bi-lstm model saved.""")
        if load_bilstm_agg:
            now = datetime.now()
            print(f"""
        ({now.strftime("%d/%m/%Y %H:%M:%S")}) Predictions Bi-LSTM (AGG) : running ...""")
            bilstm_agg = load_model(bilstm_file.replace(".pickle", "_agg.pickle"))
            validation_predictions_agg['bilstm'] = bilstm_agg.predict(regions_agg, dates_agg, 'agg')
            now = datetime.now() 
            print(f"""
        ({now.strftime("%d/%m/%Y %H:%M:%S")}) Predictions Bi-LSTM (AGG) : done.""")
            models_agg.append('bilstm')

        # train MDP
        if train_mdp_agg:
            now = datetime.now()
            print(f"""
        ({now.strftime("%d/%m/%Y %H:%M:%S")}) Training MDP (AGG) : running ...""")
            mdp_agg = MDPModel(**mdp_params_dict)
            # mdp_agg.fit(df_train_agg[df_train_agg['cases'] >= 200])
            mdp_agg.fit(df_train_agg)
            now = datetime.now() 
            print(f"""
        ({now.strftime("%d/%m/%Y %H:%M:%S")}) Training MDP (AGG) : done.""")
            save_model(mdp_agg, mdp_file.replace(".pickle", "_agg.pickle"))
            print(f"""
        mdp model saved.""")

        if load_mdp_agg:
            now = datetime.now()
            print(f"""
        ({now.strftime("%d/%m/%Y %H:%M:%S")}) Predictions MDP (AGG) : running ...""")
            mdp_agg = load_model(mdp_file.replace(".pickle", "_agg.pickle"))
            validation_predictions_agg['mdp'] = mdp_agg.predict(regions_agg, dates_agg)
            now = datetime.now()
            print(f"""
        ({now.strftime("%d/%m/%Y %H:%M:%S")}) Predictions MDP (AGG) : done.""")
            models_agg.append('mdp')

        # train rMDP
        if train_rmdp_agg:
            now = datetime.now()
            print(f"""
        ({now.strftime("%d/%m/%Y %H:%M:%S")}) Training rMDP (AGG) : running ...""")
            rmdp_agg = rMDPModel(**rmdp_params_dict)
            mode = "TIME_CV"
            rmdp_agg.fit(df_train_agg, mode=mode)
            now = datetime.now() 
            print(f"""
        ({now.strftime("%d/%m/%Y %H:%M:%S")}) Training rMDP (AGG) : done.""")
            save_model(rmdp_agg, rmdp_file.replace(".pickle", "_agg.pickle"))
            print(f"""
        rmdp model saved.""")

        if load_rmdp_agg:
            now = datetime.now()
            print(f"""
        ({now.strftime("%d/%m/%Y %H:%M:%S")}) Predictions rMDP (AGG) : running ...""")
            rmdp_agg = load_model(rmdp_file.replace(".pickle", "_agg.pickle"))
            rmdp_agg.verbose = 0
            validation_predictions_agg['rmdp'] = rmdp_agg.predict(regions_agg, dates_agg)
            now = datetime.now()
            print(f"""
        ({now.strftime("%d/%m/%Y %H:%M:%S")}) Predictions rMDP (AGG) : done.""")
            models_agg.append('rmdp')

        if load_bilstm_agg + load_knn_agg + load_mdp_agg + load_rmdp_agg >= 3:
            print(f"""
        [AGG] Additional saving of the predictions (~AGG) as a pickle and csv.""")

            with open(
                    os.path.join( 
                        result_path, 
                        "prediction_aggregate_{}_{}_{}_{}.pickle".format(region_col, target_col, str(min(dates_val))[:10], str(max(dates_val))[:10])), "wb") as file:
                pickle.dump(validation_predictions_agg, file)

            df_agg = dict_to_df(validation_predictions_agg,
                                df_validation_agg)

            df_agg.to_csv(
                os.path.join( 
                    result_path, 
                    "prediction_aggregate_{}_{}_{}_{}.csv".format(region_col, target_col, str(min(dates_agg))[:10], str(max(dates_agg))[:10])))


        df_agg = pd.read_csv(os.path.join( 
                    result_path, 
                    "prediction_aggregate_{}_{}_{}_{}.csv".format(region_col, target_col, str(min(dates_agg))[:10], str(max(dates_agg))[:10])),
                             index_col=0, parse_dates=["date"])


        now = datetime.now()
        print(f"""
    ({now.strftime("%d/%m/%Y %H:%M:%S")}) Training AGG : running ...""")
        agg = AGGModel(date=date_col,
                       region=region_col,
                       target=target_col,
                       models=["knn", "rmdp", "bilstm"],  # ["knn", "mdp", "bilstm"]
                       per_region=per_region,
                       ml_methods=ml_methods,
                       ml_mapping=ml_mapping,
                       ml_hyperparams=ml_hyperparams)
        agg.fit(df_agg)
        now = datetime.now()
        print(f"""
    ({now.strftime("%d/%m/%Y %H:%M:%S")}) Training AGG : done. """)
        save_model(agg, agg_file)
        print(f"""
    agg model saved.""")

    if load_agg:
        now = datetime.now()
        print(f"""
    ({now.strftime("%d/%m/%Y %H:%M:%S")}) Predictions AGG : running...""")
        with open(
                os.path.join( 
                    result_path, 
                    "prediction_intermediate_{}_{}_{}_{}.pickle".format(region_col, target_col, str(min(dates_val))[:10], str(max(dates_val))[:10])), "rb") as file:
            validation_predictions = pickle.load(file)
            models = list(validation_predictions.keys())
        agg = load_model(agg_file)
        validation_predictions['agg'] = agg.predict(regions_val, dates_val, validation_predictions)
        now = datetime.now()
        print(f"""
    ({now.strftime("%d/%m/%Y %H:%M:%S")}) Predictions AGG : done.""")
        models.append('agg')

        df_agg = dict_to_df(validation_predictions,
                            df_validation).dropna(subset=[ "rmdp", "bilstm", "agg"])
        df_agg.to_csv(os.path.join( 
                        result_path,
                        "prediction_intermediate_{}_{}_{}_{}_with_agg.csv".format(region_col, target_col, str(min(dates_val))[:10], str(max(dates_val))[:10])))

        print(f"""
        Additional saving of the predictions as a csv.""")

    if train_ci:
        now = datetime.now()
        print(f"""
    ({now.strftime("%d/%m/%Y %H:%M:%S")}) Training Confidence Interval : running ...""")
        prediction_train = pd.read_csv(os.path.join( 
                        result_path,
                        "prediction_intermediate_{}_{}_{}_{}_with_agg.csv".format(region_col, target_col, str(min(dates_val))[:10], str(max(dates_val))[:10])),
                             index_col=0, parse_dates=["date"])
        prediction_train.dropna(subset=["agg"], inplace=True)
        models = ["mdp", "bilstm", "agg"]
        ci = CI(region_col=region_col,
                target_col=target_col,
                ci_range=ci_range,
                models=models)
        ci.fit(prediction_train)
        now = datetime.now()
        print(f"""
    ({now.strftime("%d/%m/%Y %H:%M:%S")}) Training Confidence Interval : done.""")
        save_model(ci, ci_file)

    if train_preval:
        now = datetime.now()
        print(f"""
    ({now.strftime("%d/%m/%Y %H:%M:%S")}) Training Prevalence : running ...""")
        preval = PrevalenceModel(region_col=region_col,
                                 date_col=date_col,
                                 tests_col=tests_col,
                                 population_col=population,
                                 alpha=alpha)
        preval.fit(df_train)
        now = datetime.now()
        print(f"""
    ({now.strftime("%d/%m/%Y %H:%M:%S")}) Training Prevalence : done.""")
        save_model(preval, preval_file)

    if retrain:
    # %% Train and Save Models
        print(
f"""

-------------------------------
3) Training and (re-TRAIN) :

    Training and predictions : running ...

""")

        if train_sir_ret:
            now = datetime.now()
            print(f"""
        ({now.strftime("%d/%m/%Y %H:%M:%S")}) Training SEIRD : running ...""")
            sir = SIRModel(**sir_params_dict)
            sir.fit(df_train_ret)
            now = datetime.now()
            print(f"""
        ({now.strftime("%d/%m/%Y %H:%M:%S")}) Training SEIRD : done.""")
            save_model(sir, sir_file)

        if train_bilstm_ret:
            now = datetime.now()
            print(f"""
        ({now.strftime("%d/%m/%Y %H:%M:%S")}) Training Bi-LSTM : running ...""")
            bilstm = BILSTMModel(**bilstm_params_dict)
            bilstm.fit(df_train_ret)
            now = datetime.now()
            print(f"""
        ({now.strftime("%d/%m/%Y %H:%M:%S")}) Training Bi-LSTM : done.""")
            save_model(bilstm, bilstm_file)

        if train_knn_ret:
            now = datetime.now()
            print(f"""
        ({now.strftime("%d/%m/%Y %H:%M:%S")}) Training kNN : running ...""")
            knn = KNNModel(**knn_params_dict)
            knn.fit(df_train_ret, regions_val)
            now = datetime.now()
            print(f"""
        ({now.strftime("%d/%m/%Y %H:%M:%S")}) Training kNN : done.""")
            save_model(knn, knn_file)

        if train_mdp_ret:
            now = datetime.now()
            print(f"""
        ({now.strftime("%d/%m/%Y %H:%M:%S")}) Training MDP : running ...""")
            mdp = MDPModel(**mdp_params_dict)
            mdp.fit(df_train_ret)
            now = datetime.now()
            print(f"""
        ({now.strftime("%d/%m/%Y %H:%M:%S")}) Training MDP : done.""")
            save_model(mdp, mdp_file)

        if train_preval:
            now = datetime.now()
            print(f"""
        ({now.strftime("%d/%m/%Y %H:%M:%S")}) Training Prevalence : running ...""")
            preval = PrevalenceModel(region_col=region_col,
                                     date_col=date_col,
                                     tests_col=tests_col,
                                     population_col=population,
                                     alpha=alpha)
            preval.fit(df)
            now = datetime.now()
            print(f"""
        ({now.strftime("%d/%m/%Y %H:%M:%S")}) Training Prevalence : done.""")
            save_model(preval, preval_file)

    print("""
###################
END OF THE TRAINING
-------------------
""")

    if batcktest:
        print(
"""

BEGINNING OF THE BATCKTESTING
############################# """)

        df, _, df_train, df_test = load_data(training_cutoff=training_cutoff, validation_cutoff=validation_cutoff)

        regions = list(set(df_test[region_col]))
        dates = sorted(list(set(df_test[date_col])))
        #%% Load Models and Make Predictions

        output = {}
        models = []
        if load_sir:
            sir = load_model(sir_file.replace(".pickle", "_train.pickle"))
            output['sir'] = sir.predict(regions, dates)
            models.append('sir')
        
        if load_knn:
            knn = load_model(knn_file.replace(".pickle", "_train.pickle"))
            output['knn'] = knn.predict(regions, dates)
            models.append('knn')

        if load_mdp:
            mdp = load_model(mdp_file.replace(".pickle", "_train.pickle"))
            output['mdp'] = mdp.predict(regions, dates)
            models.append('mdp')

        if load_rmdp:
            rmdp = load_model(rmdp_file.replace(".pickle", "_train.pickle"))
            rmdp.verbose = 0
            output['rmdp'] = rmdp.predict(regions, dates)
            models.append('rmdp')

        if load_bilstm:
            bilstm = load_model(bilstm_file.replace(".pickle", "_train.pickle"))
            output['bilstm'] = bilstm.predict(regions, dates, "first")
            models.append('bilstm')

        print("DEBUG regions:", regions)
        if load_agg:
            agg = load_model(agg_file)
            output['agg'] = agg.predict(regions, dates, output)
            models.append('agg')

        if load_ci:
            ci = load_model(ci_file)
            sampled_output = ci.sample(output)
            low_output = ci.sample(output, how='low')
            high_output = ci.sample(output, how='high')
            for model in models:
                output[model + '_low'] = low_output[model]
                output[model + '_high'] = high_output[model]
                output[model + '_sample'] = sampled_output[model]

        df_agg = dict_to_df(output,
                            df_test)
        df_agg.to_csv(os.path.join(result_path, "backtest_tr_cutoff_{}_{}_{}_{}.csv".format(region_col, target_col, training_cutoff, validation_cutoff)))

        results = get_mapes(df_agg.dropna(subset=models),
                            models,
                            region_col=region_col,
                            target_col=target_col)
        results.to_csv(os.path.join(result_path, "mapes_{}_{}_{}_{}.csv".format(region_col, target_col, training_cutoff, validation_cutoff)))

        results = get_last_day_mapes(df_agg.dropna(subset=models),
                            models,
                            region_col=region_col,
                            target_col=target_col)
        results.to_csv(os.path.join(result_path, "last_day_mapes_{}_{}_{}_{}.csv".format(region_col, target_col, training_cutoff, validation_cutoff)))


        export = df.merge(df_agg.iloc[:, :-1], how='left', on=[region_col, date_col])
        export.to_csv(os.path.join(result_path, "export_{}_{}_{}_{}.csv".format(region_col, target_col, training_cutoff, validation_cutoff)))
    
        print("""
######################
END OF THE BACKTESTING
----------------------
""")

    if test:
        print(
    """

BEGINNING OF THE MAIN TEST
########################## """)

    #%% Load Models and Make Predictions
        from copy import deepcopy
        if any([datetime.strptime(training_cutoff, '%Y-%m-%d') > date for date in out_of_sample_dates]):
            raise Exception('Prediction dates appear in the training data. Please make predictions for a date after ' + training_cutoff)

        search = SearchEngine(simple_zipcode=True)

        # if use_zips:
        #     fips = pd.read_csv(fips_path)
        #     regions_dic = {zipcode_number:fips.query('ZIP == @zipcode_number').iloc[0, 1] for zipcode_number in zip_codes}
        #     regions = list(set(regions_dic.values()))
        #     ratio_dic = {}
        # for zipcode_number in zip_codes:
        #     fips_number = regions_dic[zipcode_number]
        #     zip_list = set(fips.query('STCOUNTYFP == @fips_number')['ZIP'])
        #     total_population = sum([search.by_zipcode(z).population for z in zip_list if search.by_zipcode(z).population is not None])
        #     population = search.by_zipcode(zipcode_number).population
        #     ratio = population/total_population
        #
        #     ratio_dic[zipcode_number] = ratio

        output = {}

        if load_knn:
            knn = load_model(knn_file)
            output['knn'] = knn.predict(regions, out_of_sample_dates)

        if load_sir:
            sir = load_model(sir_file)
            output['sir'] = sir.predict(regions, out_of_sample_dates)

        if load_mdp:
            mdp = load_model(mdp_file)
            output['mdp'] = mdp.predict(regions, out_of_sample_dates)

        if load_rmdp:
            rmdp = load_model(rmdp_file)
            rmdp.verbose =0
            output['rmdp'] = rmdp.predict(regions, out_of_sample_dates)

        if load_bilstm:
            bilstm = load_model(bilstm_file)
            output['bilstm'] = bilstm.predict(regions, out_of_sample_dates)

        if load_agg:
            agg = load_model(agg_file)
            output['agg'] = agg.predict(regions, out_of_sample_dates, output)


        with open(os.path.join(result_path, "out_of_sample_pred_{}_{}_{}_{}.pickle".format(region_col, target_col, validation_cutoff, len(out_of_sample_dates))), 'wb') as fp:
            pickle.dump(output, fp)

        if load_ci:
            ci = load_model(ci_file)
            sampled_output = ci.sample(output)
            low_output = ci.sample(output, how='low')
            high_output = ci.sample(output, how='high')
            for model in models:
                output[model + '_low'] = low_output[model]
                output[model + '_high'] = high_output[model]
                output[model + '_sample'] = sampled_output[model]

            with open(os.path.join(result_path, "out_of_sample_pred_{}_{}_{}_{}_with_ci.pickle".format(region_col, target_col, validation_cutoff, len(out_of_sample_dates))), 'wb') as fp:
                pickle.dump(output, fp)
        # TO UPDATE
        # df_agg = dict_to_df(output,
        #                     df_test)
        # df_agg.to_csv(os.path.join(result_path, "backtest_tr_cutoff_{}_{}_{}_{}.csv".format(region_col, target_col, training_cutoff, validation_cutoff)))

        if load_new_cases:
            ci = load_model(ci_file)
            sampled_output = ci.sample(output, n_samples, random_state)
            #Generate JSONs with random samples per model
            pathstr = os.path.split(df_path)
            for model_type in sampled_output.keys():
                prediction_distribution = dict.fromkeys(regions)
                for state in regions:
                    predictions = sampled_output[model_type][state]
                    prediction_distribution[state] = predictions
                all_samples = []
                
                print(sampled_output)
                print(model_type)
                print(regions)
                print(regions[0])

                # import pdb; pdb.set_trace()
                date_list = sampled_output[model_type][regions[0]][0].index.strftime('%Y-%m-%d').tolist()
                samples = dict(dates=date_list, samples=None)
                for t_i in range(n_samples):
                    sample_dict = dict.fromkeys(regions)
                    for state in regions:
                        sample_dict[state] = list(prediction_distribution[state][t_i])
                    all_samples.append(sample_dict)
                samples['samples'] = all_samples
                samples['samples'] = [[{state:samples['samples'][i][state][date] for state in samples['samples'][0].keys()} for date in range(len(samples['dates']))] for i in range(n_samples)]

                if load_preval:
                    preval = load_model(preval_file)
                    confirmed_samples = deepcopy(samples)
                    samples = preval.convert(samples)
                    print(samples)

                if new_cases:
                    new_samples = {}
                    new_samples['dates'] = samples['dates'][infection_period:]
                    #new_samples['samples'] = []
                    tags = ['Total_uncontained_infections', 'Total_contained_infections', 'Total_confirmed_infections', 'New_uncontained_infections', 'New_contained_infections', 'New_confirmed_infections']

                    l = []

                    for i in range(len(samples['samples'])):
                        cl_dic = {}
                        for state in samples['samples'][i][0].keys():
                            sub_sample = [samples['samples'][i][j][state] -samples['samples'][i][j-1][state] for j in range(1, len(out_of_sample_dates))]
                            total_new = [sum(sub_sample[i+j] for i in range(infection_period)) for j in range(len(out_of_sample_dates)-infection_period)]

                            total = [samples['samples'][i][j][state] for j in range(1, len(out_of_sample_dates))]

                            total_severe = [severe_infections*a for a in total]
                            total_mild = [(1-severe_infections)*a for a in total]

                            total_new_severe = [severe_infections*a for a in total_new]
                            total_new_mild = [(1-severe_infections)*a for a in total_new]
                            confirmed_sub_sample = [confirmed_samples['samples'][i][j][state] -samples['samples'][i][j-1][state] for j in range(1, len(out_of_sample_dates))]
                            confirmed_total_new = [sum(confirmed_sub_sample[i+j] for i in range(infection_period)) for j in range(len(out_of_sample_dates)-infection_period)]

                            confirmed_sub_sample_total = [confirmed_samples['samples'][i][j][state] for j in range(1, len(out_of_sample_dates))]
                            confirmed_total = [sum(confirmed_sub_sample_total[i+j] for i in range(infection_period)) for j in range(len(out_of_sample_dates)-infection_period)]

                            cl_dic[state] = (total_severe, total_mild, confirmed_total, total_new_severe, total_new_mild, confirmed_total_new)
                        sample_dic = [{state: {tags[i]: cl_dic[state][i][j] for i in range(len(tags))} for state in samples['samples'][i][0].keys()} for j in range(len(out_of_sample_dates)-infection_period)]
                        if use_zips:
                            sample_dic = [{zipcode_number: {tags[i]: cl_dic[regions_dic[zipcode_number]][i][j]*ratio_dic[zipcode_number] for i in range(len(tags))} for zipcode_number in zip_codes} for j in range(len(out_of_sample_dates)-infection_period)]
                        l.append(sample_dic)
                    new_samples['samples'] = l
                    samples = new_samples

                with open(os.path.join(result_path, "out_of_sample_prevalence_{}_{}_{}_{}.json".format(region_col, target_col, validation_cutoff, len(out_of_sample_dates))), 'w') as fp:
                    json.dump(output, fp)


        print("""
####################
END OF THE MAIN TEST
--------------------
""")