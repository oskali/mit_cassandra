# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 22:02:43 2020

@author: omars
"""

#%% Libraries and Parameters

from codes.data_utils import (load_model)
from codes.params import (regions, dates)
import warnings
import os
import pickle
warnings.filterwarnings("ignore")

#%% Load Models and Make Predictions

# mdp_file = r"C:\Users\david\Desktop\MIT\Courses\Research internship\results\20 - 20200819 - Massachusetts with Boosted MDP new pred\MDPs_without_actions\TIME_CV\mdp__target_cases__h5__davg3__cdt_10pct__n_iter200__ClAlg_Rando__errhoriz_cv4_nbfs2\mdp_backup_save.pkl"
mdp_file = r"C:\Users\david\Desktop\MIT\Courses\Research internship\results\21 - 20200819 - Massachusetts with Boosted MDP new pred\MDPs_without_actions\TIME_CV\mdp__target_cases__h5__davg3__cdt_10pct__n_iter400__ClAlg_Rando__errhoriz_cv4_nbfs2\mdp_backup_save.pkl"
output = {}
mdp = load_model(mdp_file)
output['mdp'] = mdp.predict(regions, dates, from_first=False)
#
# with open(os.path.join('output_predictions_country.pickle'), 'wb') as fp:
#     pickle.dump(output, fp)
print("ok")

