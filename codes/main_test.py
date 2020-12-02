# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 22:02:43 2020

@author: omars
"""

#%% Libraries and Parameters

from codes.data_utils import (load_model, load_data, dict_to_df, get_mapes)
from codes.params import (training_cutoff, validation_cutoff, region_col, date_col)
import warnings
import os
import pickle
warnings.filterwarnings("ignore")


df, df_train, df_test = load_data(training_cutoff=training_cutoff, validation_cutoff=None)

regions = list(set(df_test[region_col]))
dates = list(set(df_test[date_col]))
dates = sorted(dates)
# regions = ["Massachusetts"]

output = {}
models = []

if __name__ == "__main__":  # required for running multiple kernels
    #%% Load Models and Make Predictions

    # mdp_file = r"C:\Users\david\Desktop\MIT\Courses\Research internship\results\20 - 20200819 - Massachusetts with Boosted MDP new pred\MDPs_without_actions\TIME_CV\mdp__target_cases__h5__davg3__cdt_10pct__n_iter200__ClAlg_Rando__errhoriz_cv4_nbfs2\mdp_backup_save.pkl"
    # mdp_file = r"C:\Users\david\Desktop\MIT\Courses\Research internship\results\37 - 20201113 - testing risk without actions\MDPs_with_actions\TIME_CV\mdpR__target_cases__h8__davg3__cdt_30pct__n_iter150__ClAlg_Decis__errhoriz_cv30_nbfs20\mdp_20200909_case - Copy.pkl"
    # mdp_file = r"C:\Users\david\Desktop\MIT\Courses\Research internship\results\38 - 20201201 - testing risk without actions\MDPs_with_actions\TIME_CV\mdpR__target_cases__h8__davg7__cdt_30pct__n_iter80__ClAlg_Decis__errhoriz_cv2_nbfs20\mdp_20200909_cases_state.pkl"
    mdp_file = r"C:\Users\david\Desktop\MIT\Courses\Research internship\results\38 - 20201201 - testing risk without actions\MDPs_with_actions\TIME_CV\mdpR__target_cases__h8__davg7__cdt_30pct__n_iter130__ClAlg_Decis__errhoriz_cv5_nbfs20\mdp_20200909_cases_state.pkl"
    mdp = load_model(mdp_file)
    mdp.verbose = 1

    # output['mdp_exp_mean'] = mdp.predict(regions, dates[:45], from_first=False, model_key="exp_mean_diff", n_jobs=-1)
    output['mdp_opt'] = mdp.predict(regions, dates[:60], from_first=False, model_key="k_opt", n_jobs=-1)
    models.append('mdp_opt')
    # output['mdp_exp_mean'] = mdp.predict(regions, dates[:45], from_first=False, model_key="exp_mean", n_jobs=-1)
    output['mdp_exp_mean'] = mdp.predict(regions, dates[:60], from_first=False, model_key="exp_mean", n_jobs=-1)
    models.append('mdp_exp_mean')
    # output['mdp_logmean'] = mdp.predict(regions, dates[:60], from_first=False, model_key="log_mean", n_jobs=-1)
    # models.append('mdp_logmean')
    output['mdp_median'] = mdp.predict(regions, dates[:60], from_first=False, model_key="median", n_jobs=-1)
    models.append('mdp_median')

    df_agg = dict_to_df(output=output, df_validation=df_test).sort_values(by=["state", "date"]).dropna()
    results = get_mapes(df_agg, models)

    #%% Load Models and Make Predictions
    output['mdp_mean'] = mdp.predict(regions, dates[:45], from_first=False, model_key="mean", n_jobs=-1)
    output['mdp_median'] = mdp.predict(regions, dates[:45], from_first=False, model_key="median", n_jobs=-1)
    output['mdp_std'] = mdp.predict(regions, dates[:45], from_first=False, model_key="std", n_jobs=-1)
    output['mdp_std_diff'] = mdp.predict(regions, dates[:45], from_first=False, model_key="std_diff", n_jobs=-1)
    #
    # with open(os.path.join(os.path.dirname(mdp_file), 'output_predictions_country_rob2.pickle'), 'wb') as fp:
    #     pickle.dump(output, fp)
    print("ok")

