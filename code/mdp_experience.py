# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 21:24:25 2020

@author: omars
"""


if __name__ == "__main__":

    import os
    from data_utils import (save_model, load_model, load_data, dict_to_df)
    from params import (train_sir, train_bilstm, train_knn, train_mdp, train_agg, train_ci,
                        date_col, region_col, target_col, sir_file, knn_file, mdp_file, agg_file, ci_file,
                        validation_cutoff, training_cutoff, training_agg_cutoff,
                        per_region, ml_methods, ml_mapping, ml_hyperparams, ci_range,
                        knn_params_dict, sir_params_dict, mdp_params_dict, bilstm_params_dict,retrain,
                        train_mdp_agg, train_sir_agg, train_bilstm_agg, train_knn_agg,
                        load_sir, load_knn, load_bilstm, load_mdp, train_preval, tests_col, population,
                        preval_file, bilstm_file, alpha)

    from mdp_model import MDPModel
    import warnings
    warnings.filterwarnings("ignore")

    EXPERIMENT_PATH = r"C:\Users\david\Desktop\MIT\Courses\Research internship\results\28 - 20201008 - mobility analysis"
    random_state = 42
    df, df_train, df_validation = load_data(validation_cutoff=validation_cutoff)
    _, df_train_agg, df_validation_agg = load_data(training_cutoff=training_agg_cutoff,
                                                   validation_cutoff=training_cutoff)

    # # %% Experiment 1
    #
    # # %% Parameters MDP
    try:
        assert False
        region_exceptions_dict = {
            "state":
                ['Guam', 'Northern Mariana Islands', 'Puerto Rico',
                 'Diamond Princess',
                 'Grand Princess', 'American Samoa', 'Virgin Islands',
                 'Hawaii', "Benin", "Ecuador",
                 "Jordan", "Lithuania", "Uganda",
                 "Georgia", "International", "Mongolia"
                 ],
            "fips":
                []}

        mdp_features_dict = \
            {
                'state':
                    {"deaths": ["cases_pct3", "cases_pct5"],
                     "cases": ["workplaces_percent_change_from_baseline_med_diff7", "cases_pct3", "cases_pct5"]},  # ["cases_nom", "cases_pct3", "cases_pct5"]},
                'fips':
                    {"deaths": [],
                     "cases": []}
            }

        mdp_params_dict = \
            {
                "days_avg": 3,
                "d_delay": 0,
                "horizon": 8,
                "n_iter": 90,
                "n_folds_cv": 4,
                "clustering_distance_threshold": 0.1,
                "splitting_threshold": 0.,
                "classification_algorithm": 'DecisionTreeClassifier',
                "clustering_algorithm": 'Agglomerative',
                "n_clusters": None,
                "action_thresh": ([-21, -6, 7, 17], 2),  # ([-250, 200], 1),
                "features_list": mdp_features_dict[region_col][target_col],
                "completion_algorithm": "relative_completion",
                "verbose": 1,
                "n_jobs": -1,
                "date_colname": date_col,
                "target_colname": target_col,
                "region_colname": region_col,
                "random_state": random_state,
                "keep_first": True,
                "save": False,
                "plot": False,
                "savepath": "",  # os.path.dirname(mdp_file),
                "region_exceptions": None
            }

        mdp = MDPModel(**mdp_params_dict)
        mdp.fit(df_train)
        save_model(mdp, os.path.join(EXPERIMENT_PATH, "mdp_exp1.pickle"))
    except:
        pass

    # %% Experiment 2

    # %% Parameters MDP
    try:
        region_exceptions_dict = {
            "state":
                ['Guam', 'Northern Mariana Islands', 'Puerto Rico',
                 'Diamond Princess',
                 'Grand Princess', 'American Samoa', 'Virgin Islands',
                 'Hawaii', "Benin", "Ecuador",
                 "Jordan", "Lithuania", "Uganda",
                 "Georgia", "International", "Mongolia"
                 ],
            "fips":
                []}

        mdp_features_dict = \
            {
                'state':
                    {"deaths": ["cases_pct3", "cases_pct5"],
                     "cases": ["workplaces_percent_change_from_baseline_med_diff7", "cases_pct3", "cases_pct5"]},  # ["cases_nom", "cases_pct3", "cases_pct5"]},
                'fips':
                    {"deaths": [],
                     "cases": []}
            }

        mdp_params_dict = \
            {
                "days_avg": 3,
                "d_delay": 7,
                "horizon": 8,
                "n_iter": 90,
                "n_folds_cv": 4,
                "clustering_distance_threshold": 0.1,
                "splitting_threshold": 0.,
                "classification_algorithm": 'DecisionTreeClassifier',
                "clustering_algorithm": 'Agglomerative',
                "n_clusters": None,
                "action_thresh": ([-21, -6, 7, 17], 2),  # ([-250, 200], 1),
                "features_list": mdp_features_dict[region_col][target_col],
                "completion_algorithm": "relative_completion",
                "verbose": 1,
                "n_jobs": -1,
                "date_colname": date_col,
                "target_colname": target_col,
                "region_colname": region_col,
                "random_state": random_state,
                "keep_first": True,
                "save": False,
                "plot": False,
                "savepath": "",  # os.path.dirname(mdp_file),
                "region_exceptions": None
            }

        mdp = MDPModel(**mdp_params_dict)
        mdp.fit(df_train)
        save_model(mdp, os.path.join(EXPERIMENT_PATH, "mdp_exp2.pickle"))

    except:
        pass

    # %% Experiment 3

    # %% Parameters MDP
    try:
        region_exceptions_dict = {
            "state":
                ['Guam', 'Northern Mariana Islands', 'Puerto Rico',
                 'Diamond Princess',
                 'Grand Princess', 'American Samoa', 'Virgin Islands',
                 'Hawaii', "Benin", "Ecuador",
                 "Jordan", "Lithuania", "Uganda",
                 "Georgia", "International", "Mongolia"
                 ],
            "fips":
                []}

        mdp_features_dict = \
            {
                'state':
                    {"deaths": ["cases_pct3", "cases_pct5"],
                     "cases": ["workplaces_percent_change_from_baseline_med_diff7", "cases_pct3", "cases_pct5"]},  # ["cases_nom", "cases_pct3", "cases_pct5"]},
                'fips':
                    {"deaths": [],
                     "cases": []}
            }

        mdp_params_dict = \
            {
                "days_avg": 3,
                "d_delay": 14,
                "horizon": 8,
                "n_iter": 90,
                "n_folds_cv": 4,
                "clustering_distance_threshold": 0.1,
                "splitting_threshold": 0.,
                "classification_algorithm": 'DecisionTreeClassifier',
                "clustering_algorithm": 'Agglomerative',
                "n_clusters": None,
                "action_thresh": ([-21, -6, 7, 17], 2),  # ([-250, 200], 1),
                "features_list": mdp_features_dict[region_col][target_col],
                "completion_algorithm": "relative_completion",
                "verbose": 1,
                "n_jobs": -1,
                "date_colname": date_col,
                "target_colname": target_col,
                "region_colname": region_col,
                "random_state": random_state,
                "keep_first": True,
                "save": False,
                "plot": False,
                "savepath": "",  # os.path.dirname(mdp_file),
                "region_exceptions": None
            }

        mdp = MDPModel(**mdp_params_dict)
        mdp.fit(df_train)
        save_model(mdp, os.path.join(EXPERIMENT_PATH, "mdp_exp3.pickle"))
    except:
        pass

    # %% Experiment 4

    # %% Parameters MDP
    # try:
    region_exceptions_dict = {
        "state":
            ['Guam', 'Northern Mariana Islands', 'Puerto Rico',
             'Diamond Princess',
             'Grand Princess', 'American Samoa', 'Virgin Islands',
             'Hawaii', "Benin", "Ecuador",
             "Jordan", "Lithuania", "Uganda",
             "Georgia", "International", "Mongolia"
             ],
        "fips":
            []}

    mdp_features_dict = \
        {
            'state':
                {"deaths": ["cases_pct3", "cases_pct5"],
                 "cases": ["workplaces_percent_change_from_baseline_med_diff7", "cases_pct3", "cases_pct5"]},  # ["cases_nom", "cases_pct3", "cases_pct5"]},
            'fips':
                {"deaths": [],
                 "cases": []}
        }

    mdp_params_dict = \
        {
            "days_avg": 3,
            "d_delay": 0,
            "horizon": 8,
            "n_iter": 90,
            "n_folds_cv": 4,
            "clustering_distance_threshold": 0.1,
            "splitting_threshold": 0.,
            "classification_algorithm": 'DecisionTreeClassifier',
            "clustering_algorithm": 'Agglomerative',
            "n_clusters": None,
            "action_thresh": ([-21, -6, 7, 17], 2),  # ([-250, 200], 1),
            "features_list": mdp_features_dict[region_col][target_col],
            "completion_algorithm": "bias_completion",
            "verbose": 1,
            "n_jobs": -1,
            "date_colname": date_col,
            "target_colname": target_col,
            "region_colname": region_col,
            "random_state": random_state,
            "keep_first": True,
            "save": False,
            "plot": False,
            "savepath": "",  # os.path.dirname(mdp_file),
            "region_exceptions": None
        }

    mdp = MDPModel(**mdp_params_dict)
    mdp.fit(df_train)
    save_model(mdp, os.path.join(EXPERIMENT_PATH, "mdp_exp4.pickle"))
    # except:
    #     pass

    # %% Experiment 5

    # %% Parameters MDP
    try:
        region_exceptions_dict = {
            "state":
                ['Guam', 'Northern Mariana Islands', 'Puerto Rico',
                 'Diamond Princess',
                 'Grand Princess', 'American Samoa', 'Virgin Islands',
                 'Hawaii', "Benin", "Ecuador",
                 "Jordan", "Lithuania", "Uganda",
                 "Georgia", "International", "Mongolia"
                 ],
            "fips":
                []}

        mdp_features_dict = \
            {
                'state':
                    {"deaths": ["cases_pct3", "cases_pct5"],
                     "cases": ["workplaces_percent_change_from_baseline_med_diff7", "cases_pct3", "cases_pct5"]},  # ["cases_nom", "cases_pct3", "cases_pct5"]},
                'fips':
                    {"deaths": [],
                     "cases": []}
            }

        mdp_params_dict = \
            {
                "days_avg": 3,
                "d_delay": 7,
                "horizon": 8,
                "n_iter": 90,
                "n_folds_cv": 4,
                "clustering_distance_threshold": 0.1,
                "splitting_threshold": 0.,
                "classification_algorithm": 'DecisionTreeClassifier',
                "clustering_algorithm": 'Agglomerative',
                "n_clusters": None,
                "action_thresh": ([-21, -6, 7, 17], 2),  # ([-250, 200], 1),
                "features_list": mdp_features_dict[region_col][target_col],
                "completion_algorithm": "bias_completion",
                "verbose": 1,
                "n_jobs": -1,
                "date_colname": date_col,
                "target_colname": target_col,
                "region_colname": region_col,
                "random_state": random_state,
                "keep_first": True,
                "save": False,
                "plot": False,
                "savepath": "",  # os.path.dirname(mdp_file),
                "region_exceptions": None
            }

        mdp = MDPModel(**mdp_params_dict)
        mdp.fit(df_train)
        save_model(mdp, os.path.join(EXPERIMENT_PATH, "mdp_exp5.pickle"))
    except:
        pass

    # %% Experiment 6

    # %% Parameters MDP
    try:
        region_exceptions_dict = {
            "state":
                ['Guam', 'Northern Mariana Islands', 'Puerto Rico',
                 'Diamond Princess',
                 'Grand Princess', 'American Samoa', 'Virgin Islands',
                 'Hawaii', "Benin", "Ecuador",
                 "Jordan", "Lithuania", "Uganda",
                 "Georgia", "International", "Mongolia"
                 ],
            "fips":
                []}

        mdp_features_dict = \
            {
                'state':
                    {"deaths": ["cases_pct3", "cases_pct5"],
                     "cases": ["workplaces_percent_change_from_baseline_med_diff7", "cases_pct3", "cases_pct5"]},  # ["cases_nom", "cases_pct3", "cases_pct5"]},
                'fips':
                    {"deaths": [],
                     "cases": []}
            }

        mdp_params_dict = \
            {
                "days_avg": 3,
                "d_delay": 14,
                "horizon": 8,
                "n_iter": 90,
                "n_folds_cv": 4,
                "clustering_distance_threshold": 0.1,
                "splitting_threshold": 0.,
                "classification_algorithm": 'DecisionTreeClassifier',
                "clustering_algorithm": 'Agglomerative',
                "n_clusters": None,
                "action_thresh": ([-21, -6, 7, 17], 2),  # ([-250, 200], 1),
                "features_list": mdp_features_dict[region_col][target_col],
                "completion_algorithm": "bias_completion",
                "verbose": 1,
                "n_jobs": -1,
                "date_colname": date_col,
                "target_colname": target_col,
                "region_colname": region_col,
                "random_state": random_state,
                "keep_first": True,
                "save": False,
                "plot": False,
                "savepath": "",  # os.path.dirname(mdp_file),
                "region_exceptions": None
            }

        mdp = MDPModel(**mdp_params_dict)
        mdp.fit(df_train)
        save_model(mdp, os.path.join(EXPERIMENT_PATH, "mdp_exp6.pickle"))
    except:
        pass

    # %% Experiment 7

    # %% Parameters MDP
    try:
        region_exceptions_dict = {
            "state":
                ['Guam', 'Northern Mariana Islands', 'Puerto Rico',
                 'Diamond Princess',
                 'Grand Princess', 'American Samoa', 'Virgin Islands',
                 'Hawaii', "Benin", "Ecuador",
                 "Jordan", "Lithuania", "Uganda",
                 "Georgia", "International", "Mongolia"
                 ],
            "fips":
                []}

        mdp_features_dict = \
            {
                'state':
                    {"deaths": ["cases_pct3", "cases_pct5"],
                     "cases": ["workplaces_percent_change_from_baseline_med_diff7", "cases_pct3", "cases_pct5"]},  # ["cases_nom", "cases_pct3", "cases_pct5"]},
                'fips':
                    {"deaths": [],
                     "cases": []}
            }

        mdp_params_dict = \
            {
                "days_avg": 3,
                "d_delay": 0,
                "horizon": 8,
                "n_iter": 90,
                "n_folds_cv": 4,
                "clustering_distance_threshold": 0.1,
                "splitting_threshold": 0.,
                "classification_algorithm": 'DecisionTreeClassifier',
                "clustering_algorithm": 'Agglomerative',
                "n_clusters": None,
                "action_thresh": ([-21, -6, 7, 17], 2),  # ([-250, 200], 1),
                "features_list": mdp_features_dict[region_col][target_col],
                "completion_algorithm": "unbias_completion",
                "verbose": 1,
                "n_jobs": -1,
                "date_colname": date_col,
                "target_colname": target_col,
                "region_colname": region_col,
                "random_state": random_state,
                "keep_first": True,
                "save": False,
                "plot": False,
                "savepath": "",  # os.path.dirname(mdp_file),
                "region_exceptions": None
            }

        mdp = MDPModel(**mdp_params_dict)
        mdp.fit(df_train)
        save_model(mdp, os.path.join(EXPERIMENT_PATH, "mdp_exp7.pickle"))
    except:
        pass

    # %% Experiment 1

    # %% Parameters MDP
    try:
        region_exceptions_dict = {
            "state":
                ['Guam', 'Northern Mariana Islands', 'Puerto Rico',
                 'Diamond Princess',
                 'Grand Princess', 'American Samoa', 'Virgin Islands',
                 'Hawaii', "Benin", "Ecuador",
                 "Jordan", "Lithuania", "Uganda",
                 "Georgia", "International", "Mongolia"
                 ],
            "fips":
                []}

        mdp_features_dict = \
            {
                'state':
                    {"deaths": ["cases_pct3", "cases_pct5"],
                     "cases": ["workplaces_percent_change_from_baseline_med_diff7", "cases_pct3", "cases_pct5"]},  # ["cases_nom", "cases_pct3", "cases_pct5"]},
                'fips':
                    {"deaths": [],
                     "cases": []}
            }

        mdp_params_dict = \
            {
                "days_avg": 3,
                "d_delay": 7,
                "horizon": 8,
                "n_iter": 90,
                "n_folds_cv": 4,
                "clustering_distance_threshold": 0.1,
                "splitting_threshold": 0.,
                "classification_algorithm": 'DecisionTreeClassifier',
                "clustering_algorithm": 'Agglomerative',
                "n_clusters": None,
                "action_thresh": ([-21, -6, 7, 17], 2),  # ([-250, 200], 1),
                "features_list": mdp_features_dict[region_col][target_col],
                "completion_algorithm": "unbias_completion",
                "verbose": 1,
                "n_jobs": -1,
                "date_colname": date_col,
                "target_colname": target_col,
                "region_colname": region_col,
                "random_state": random_state,
                "keep_first": True,
                "save": False,
                "plot": False,
                "savepath": "",  # os.path.dirname(mdp_file),
                "region_exceptions": None
            }

        mdp = MDPModel(**mdp_params_dict)
        mdp.fit(df_train)
        save_model(mdp, os.path.join(EXPERIMENT_PATH, "mdp_exp8.pickle"))
    except:
        pass

    # %% Experiment 9

    # %% Parameters MDP
    try:
        region_exceptions_dict = {
            "state":
                ['Guam', 'Northern Mariana Islands', 'Puerto Rico',
                 'Diamond Princess',
                 'Grand Princess', 'American Samoa', 'Virgin Islands',
                 'Hawaii', "Benin", "Ecuador",
                 "Jordan", "Lithuania", "Uganda",
                 "Georgia", "International", "Mongolia"
                 ],
            "fips":
                []}

        mdp_features_dict = \
            {
                'state':
                    {"deaths": ["cases_pct3", "cases_pct5"],
                     "cases": ["workplaces_percent_change_from_baseline_med_diff7", "cases_pct3", "cases_pct5"]},  # ["cases_nom", "cases_pct3", "cases_pct5"]},
                'fips':
                    {"deaths": [],
                     "cases": []}
            }

        mdp_params_dict = \
            {
                "days_avg": 3,
                "d_delay": 14,
                "horizon": 8,
                "n_iter": 90,
                "n_folds_cv": 4,
                "clustering_distance_threshold": 0.1,
                "splitting_threshold": 0.,
                "classification_algorithm": 'DecisionTreeClassifier',
                "clustering_algorithm": 'Agglomerative',
                "n_clusters": None,
                "action_thresh": ([-21, -6, 7, 17], 2),  # ([-250, 200], 1),
                "features_list": mdp_features_dict[region_col][target_col],
                "completion_algorithm": "unbias_completion",
                "verbose": 1,
                "n_jobs": -1,
                "date_colname": date_col,
                "target_colname": target_col,
                "region_colname": region_col,
                "random_state": random_state,
                "keep_first": True,
                "save": False,
                "plot": False,
                "savepath": None,  # os.path.dirname(mdp_file),
                "region_exceptions": None
            }

        mdp = MDPModel(**mdp_params_dict)
        mdp.fit(df_train)
        save_model(mdp, os.path.join(EXPERIMENT_PATH, "mdp_exp9.pickle"))
    except:
        pass
