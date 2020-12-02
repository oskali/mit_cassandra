# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 21:24:25 2020

@author: david
"""

#%% Libraries and Parameters

from codes.data_utils import (save_model, load_data, load_model)
from codes.params import train_mdp, train_mdp_gs, mdp_params_dict, mdp_gs_params_dict, mdp_file, mdp_gs_file, \
    validation_cutoff, calibrate_mdp, training_cutoff, training_cutoff_agg, train_mdp_cal


from codes.mdp_model import MDPModel, MDPGridSearch
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":  # required for running multiple kernels

    #%% Load Data
    df, df_train, _ = load_data(validation_cutoff=validation_cutoff)
    df_cal, df_train_cal, df_calibrate = load_data(training_cutoff=training_cutoff_agg, validation_cutoff=training_cutoff)

    if train_mdp_cal:

        # train
        mdp_cal = MDPModel(**mdp_params_dict)
        mode = "TIME_CV"
        mdp_cal.fit(df_train_cal, mode=mode)
        try:
            save_model(mdp_cal, mdp_file(mode+"_CAL", str(mdp_cal)))
        except:
            import pickle
            file_pi = open("mdp_cal_backup_save.pkl", 'wb')
            pickle.dump(mdp_cal, file_pi)

    if calibrate_mdp:
        mode = "TIME_CV"
        mdp_cal = MDPModel(**mdp_params_dict)
        mdp_cal = load_model(mdp_file(mode+"_CAL", str(mdp_cal)))
        mdp_cal.verbose = 1
        mdp_cal.calibrate(df_calibrate)
        try:
            save_model(mdp_cal, mdp_file(mode+"_CAL", str(mdp_cal)))
        except:
            import pickle
            file_pi = open("mdpcal_backup_save.pkl", 'wb')
            pickle.dump(mdp_cal, file_pi)

    if train_mdp:
        mdp = MDPModel(**mdp_params_dict)
        mode = "TIME_CV"
        mdp.fit(df_train, mode=mode)
        try:
            save_model(mdp, mdp_file(mode, str(mdp)))
        except:
            import pickle
            file_pi = open("mdp_backup_save.pkl", 'wb')
            pickle.dump(mdp, file_pi)

    if calibrate_mdp:
        mode = "TIME_CV"
        mdp = MDPModel(**mdp_params_dict)
        mdp = load_model(mdp_file(mode, str(mdp)))

        mdp_cal = MDPModel(**mdp_params_dict)
        mdp_cal = load_model(mdp_file(mode+"_CAL", str(mdp_cal)))

        # update the calibration
        mdp.calibration_dict = mdp_cal.calibration_dict
        mdp.calibrated = True

        try:
            save_model(mdp, mdp_file(mode, str(mdp)))
        except:
            import pickle
            file_pi = open("mdp_backup_save.pkl", 'wb')
            pickle.dump(mdp, file_pi)

    if train_mdp_gs:
        mdp_gs = MDPGridSearch(**mdp_gs_params_dict)
        mdp_gs.fit(df_train)
        save_model(mdp_gs, mdp_gs_file)
