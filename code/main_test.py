# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 22:02:43 2020

@author: omars
"""

#%% Libraries and Parameters

from data_utils import (load_model)
from params import (load_sir, load_knn, load_mdp, sir_file,
                    knn_file, mdp_file, regions, dates)
import warnings
warnings.filterwarnings("ignore")

#%% Load Models and Make Predictions

output = {}
if load_sir:
    sir = load_model(sir_file)
    output['sir'] = sir.predict(regions, dates)

if load_knn:
    knn = load_model(knn_file)
    output['knn'] = knn.predict(regions, dates)

if load_mdp:
    mdp = load_model(mdp_file)
    output['mdp'] = mdp.predict(regions, dates)
