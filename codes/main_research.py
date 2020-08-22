# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 21:24:25 2020

@author: omars
"""

#%% Libraries and Parameters

from codes.data_utils import (save_model, load_data)
from codes.params import train_mdp, train_mdp_gs, mdp_params_dict, mdp_gs_params_dict, mdp_file, mdp_gs_file, \
    validation_cutoff


from codes.mdp_model import MDPModel, MDPGridSearch
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":  # required for running multiple kernels

    #%% Load Data
    df, df_train, df_validation = load_data(validation_cutoff=validation_cutoff)

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

    if train_mdp_gs:
        mdp_gs = MDPGridSearch(**mdp_gs_params_dict)
        mdp_gs.fit(df_train)
        save_model(mdp_gs, mdp_gs_file)