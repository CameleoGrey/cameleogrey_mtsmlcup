
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from pprint import pprint

from catboost import CatBoostClassifier

import optuna
from optuna.samplers import MOTPESampler
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

class CatBoostMulticlassOptimal():
    def __init__(self):
        
        self.model = CatBoostClassifier(n_estimators=10000, 
                                        learning_rate=0.066245,
                                        max_depth=8, 
                                        thread_count=10, 
                                        early_stopping_rounds=100, 
                                        task_type="GPU")
        self.class_names = []
        self.proba_multipliers = []
        self.optimal_class_weights = None
        pass
    
    def fit(self, x, y, eval_set=None,
            optimize_sample_weights = True, sw_n_trials=200):
        
        uniq_classes = np.unique( y )
        for i in range(len(uniq_classes)):
            self.class_names.append( uniq_classes[i] )
            self.proba_multipliers.append( 1.0 )
        self.proba_multipliers = np.array( self.proba_multipliers )
        
        if optimize_sample_weights:
            optimal_class_weights = self.optimize_sample_weights(x, y, eval_set, sw_n_trials)
            self.optimal_class_weights = optimal_class_weights
            optimal_sample_weights = np.zeros( (len(y),), dtype=np.float32 )
            y_unique = np.unique( y )
            for i in range(len(y_unique)):
                y_i = y_unique[i]
                class_mask = y == y_i
                optimal_sample_weights[ class_mask ] = optimal_class_weights[i]
            self.model.fit( x, y, eval_set=eval_set, sample_weight=optimal_sample_weights )
        else:
            self.model.fit( x, y, eval_set=eval_set )
        
        return self
    
    def predict_proba(self, x):
        
        probas = self.model.predict_proba(x)
        
        return probas
    
    def predict(self, x, proba_input = False):
        
        if proba_input:
            probas = x
        else:
            probas = self.predict_proba( x )
        y_pred = []
        for i in range(len(probas)):
            current_probas = probas[i]
            current_probas = current_probas * self.proba_multipliers
            #current_probas = (current_probas - min(current_probas)) / (max(current_probas) - min(current_probas))
            max_proba_id = np.argmax( current_probas )
            predicted_class = self.class_names[ max_proba_id ]
            y_pred.append( predicted_class )
        y_pred = np.array( y_pred )
            
        return y_pred
    
    def optimize_sample_weights(self, x, y, eval_set=None, n_trials=200):
        
        y_unique = np.unique( y )
        max_class_weight = compute_class_weight(class_weight="balanced", classes=y_unique, y=y)
        max_class_weight = np.max( max_class_weight )
        max_class_weight = 1.0 * max_class_weight
        
        def objective(trial):
            
            current_class_weights = []
            for i in range(len(self.proba_multipliers)):
                mult_i = trial.suggest_float("class_weight_{}".format(i), 0.0, max_class_weight)
                current_class_weights.append( mult_i )
            current_class_weights = np.array( current_class_weights )
            
            sample_weights = np.zeros( (len(y),), dtype=np.float32 )
            for i in range(len(y_unique)):
                y_i = y_unique[i]
                class_mask = y == y_i
                sample_weights[ class_mask ] = current_class_weights[i]
            
            model = CatBoostClassifier(n_estimators=200, 
                                       learning_rate=0.066245,
                                       max_depth=8, 
                                       thread_count=10, 
                                       early_stopping_rounds=100, 
                                       task_type="GPU")
            
            if eval_set is None:
                model.fit( x, y, eval_set=(x, y), sample_weight=sample_weights, verbose=False )
                y_pred = model.predict( x )
                opt_score = f1_score( y, y_pred, average="weighted" )
            else:
                model.fit( x, y, eval_set=eval_set, sample_weight=sample_weights, verbose=False )
                y_pred = model.predict( eval_set[0] )
                opt_score = f1_score( eval_set[1], y_pred, average="weighted" )
            
            return opt_score
        
        study = optuna.create_study(directions=["maximize"], sampler=optuna.samplers.MOTPESampler())
        study.optimize( objective, n_trials=n_trials, n_jobs=1 )
        
        best_trials = study.best_trials
        scores = []
        for i in range(len(best_trials)):
            score = best_trials[i].values[0]
            scores.append(score)
        best_trial_id = np.argmax( scores )
        best_trial = best_trials[ best_trial_id ]
        best_params = best_trial.params
        
        optimal_class_weights = np.array(list(best_params.values()))
        pprint("Best class weights: {}".format(optimal_class_weights))

        return optimal_class_weights
    
    def optimize_proba_multipliers(self, x, y, n_trials=200):
        
        probas = self.predict_proba( x )
        
        def objective(trial):
            
            current_proba_mults = []
            for i in range(len(self.proba_multipliers)):
                mult_i = trial.suggest_float("mult_{}".format(i), 0.0, 100.0)
                current_proba_mults.append( mult_i )
            current_proba_mults = np.array( current_proba_mults )
            self.proba_multipliers = current_proba_mults
            
            y_pred = self.predict(probas, proba_input=True)
            opt_score = f1_score( y, y_pred, average="weighted" )
            return opt_score
        
        study = optuna.create_study(directions=["maximize"], sampler=optuna.samplers.MOTPESampler())
        study.optimize( objective, n_trials=n_trials, n_jobs=1 )
        
        best_trials = study.best_trials
        scores = []
        for i in range(len(best_trials)):
            score = best_trials[i].values[0]
            scores.append(score)
        best_trial_id = np.argmax( scores )
        best_trial = best_trials[ best_trial_id ]
        best_params = best_trial.params
        
        self.proba_multipliers = np.array(list(best_params.values()))
        pprint("Best proba_multipliers: {}".format(self.proba_multipliers))

        return self
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        