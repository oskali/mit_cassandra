# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 22:02:43 2020

@author: omars
"""

#%% Libraries and Parameters

from codes.data_utils import (load_model)
from codes.params import (load_mdp, regions, dates)
import warnings
warnings.filterwarnings("ignore")

#%% Load Models and Make Predictions

output = {}

if load_mdp:
    mdp = load_model(mdp_file)
    output['mdp'] = mdp.predict(regions, dates)

