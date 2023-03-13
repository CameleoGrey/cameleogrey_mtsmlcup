
import numpy as np
import pandas as pd
import bisect
import optuna
from pathlib import Path

from classes.paths_config import *
from classes.utils import *

from sklearn.model_selection import train_test_split, StratifiedKFold
from lightgbm import LGBMClassifier
from lightgbm import log_evaluation
from catboost import CatBoostClassifier

class GreyFeatureSelector():
    def __init__(self):
        
        self.optimal_feature_ids = None
        self.feature_names = None
        
        pass
    
    def fit(self, x, y, n_estimators=50, n_jobs=10, cv=5, sample=None, feature_names=None, opt_metric=None, opt_rounds=200, random_state=45):
        
        x = x.copy()
        y = y.copy()
        
        if sample is not None:
            sample_count = int( sample * len(x) )
            sample_ids = np.array([i for i in range(len(x))])
            np.random.seed(random_state)
            np.random.shuffle( sample_ids )
            sample_ids = sample_ids[:sample_count]
            x = x[ sample_ids ]
            y = y[ sample_ids ]
                           
        
        self.feature_names = feature_names
        
        def objective( trial ):
            weights_dict = {}
            for i in range(len(x[0])):
                weights_dict["w_{}".format(i)] = trial.suggest_int( "w_{}".format(i), 0, 1 )
            
            current_feature_ids = []
            for i, w_i in enumerate(weights_dict.keys()):
                w_i = weights_dict[ w_i ]
                if w_i == 1:
                    current_feature_ids.append( i )
            current_feature_ids = np.array( current_feature_ids )
            
            mean_cv_score = []
            """cv_folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
            for i, (ids_train, ids_test) in enumerate( cv_folds.split(x, y) ):
                
                x_train, y_train = x[ids_train], y[ids_train]
                x_test, y_test = x[ids_test], y[ids_test]
                
                x_train = x_train[:, current_feature_ids]
                x_test = x_test[:, current_feature_ids]
                
                
                model = LGBMClassifier( n_estimators=n_estimators, n_jobs=n_jobs )
                model.fit( x_train, y_train, eval_set=(x_test, y_test), callbacks=[log_evaluation(0)] )
                
                #model = CatBoostClassifier(n_estimators=n_estimators, max_depth=8, thread_count=10, task_type="GPU")
                #model.fit( x_train, y_train, verbose=False )
                
                score_i = opt_metric( model, x_test, y_test )
                mean_cv_score.append( score_i )
            score_i = opt_metric( model, x_test, y_test )"""
            
                
                
            x_train = x[:, current_feature_ids]
            y_train = y
            model = LGBMClassifier( n_estimators=n_estimators, n_jobs=n_jobs )
            model.fit( x_train, y_train, callbacks=[log_evaluation(0)] )
            #model = CatBoostClassifier(n_estimators=n_estimators, max_depth=8, thread_count=10, task_type="GPU")
            #model.fit( x_train, y_train, verbose=False )
            score_i = opt_metric( model, x_train, y_train )
            mean_cv_score.append( score_i )
            
            mean_cv_score = np.mean( mean_cv_score )
            
            return mean_cv_score
        
        study = optuna.create_study(directions=["maximize"], sampler=optuna.samplers.MOTPESampler())
        study.optimize( objective, n_trials=opt_rounds, n_jobs=n_jobs )

        # get one of the last results where f1 != 0
        best_trials = study.best_trials
        scores = []
        for i in range(len(best_trials)):
            score = best_trials[i].values[0]
            scores.append(score)
        best_trial_id = np.argmax( scores )
        best_trial = best_trials[ best_trial_id ]
        best_params = best_trial.params

        optimal_feature_ids = []
        for i, w_i in enumerate(best_params.keys()):
            w_i = best_params[ w_i ]
            if w_i == 1:
                optimal_feature_ids.append( i )
        optimal_feature_ids = np.array( optimal_feature_ids )
        self.optimal_feature_ids = optimal_feature_ids
        self.feature_names = feature_names[ optimal_feature_ids ]
        

        return self
    
    
    def transform(self, x):
        
        x = x.copy()
        
        x_transformed = x[:, self.optimal_feature_ids]
        
        return x_transformed
    
    def get_optimal_feature_names(self):
        return self.feature_names
    
    