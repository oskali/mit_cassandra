# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 21:24:25 2020

@author: omars
"""

#%% Libraries and Parameters

from codes.data_utils import (save_model, load_data)
from codes.params import (train_sir, train_knn, train_mdp,
                    knn_params_dict, sir_params_dict, mdp_params_dict,
                    sir_file, knn_file, mdp_file)

from codes.sir_model import SIRModel
from codes.knn_model import KNNModel
from codes.mdp_model import MDPModel
import warnings
warnings.filterwarnings("ignore")

#%% Load Data
_, df_train, _ = load_data()

#%% Train and Save Models
if train_sir:
    sir = SIRModel(**sir_params_dict)
    sir.fit(df_train)
    save_model(sir, sir_file)

if train_knn:
    knn = KNNModel(**knn_params_dict)
    knn.fit(df_train)
    save_model(knn, knn_file)

if train_mdp:
    if __name__ == "__main__":  # required for running multiple kernels
        mdp = MDPModel(**mdp_params_dict)
        mdp.fit(df_train, mode="TIME_CV")
        save_model(mdp, mdp_file)
