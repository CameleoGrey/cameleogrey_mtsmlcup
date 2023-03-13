
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import compute_sample_weight
from tqdm import tqdm
import optuna
from sklearn import metrics
from pprint import pprint

class CatBoostBinarySelfOptimized():
    def __init__(self, n_jobs=8,
                 optimize_treshold=False, treshold_opt_iters=200,
                 optimize_hyperparams=True, gbm_opt_iters=100, max_opt_time=2*60*60):

        self.n_jobs = n_jobs
        self.feature_names = None
        self.optimize_hyperparams = optimize_hyperparams
        self.optimize_treshold = optimize_treshold
        self.treshold_opt_iters = treshold_opt_iters
        self.gbm_opt_iters = gbm_opt_iters
        self.max_opt_time = max_opt_time #max time of hyperparams optimization for each grad boosting model

        # default models
        """self.model = CatBoostClassifier(n_estimators=10000, learning_rate=0.017395, 
                                        boosting_type="Plain", bootstrap_type="Bernoulli",
                                        reg_lambda=20.963662807481374,
                                        subsample=0.8227476047769066,
                                        max_depth=8, thread_count=10, early_stopping_rounds=100, task_type="GPU")"""
        self.model = CatBoostClassifier(n_estimators=10000, max_depth=8, thread_count=10, early_stopping_rounds=100, task_type="GPU")
        
        self.feature_names = None
        self.optimal_treshold = None

        pass
    
    def gini_score(self, model, x, y):
        y_pred = model.predict_proba( x )
        y_pred = y_pred[:, 1]
        score = 2.0 * metrics.roc_auc_score(y, y_pred) - 1.0
        return score

    def fit(self, x_train, y_train, eval_set, feature_names=None, sample_weight=None, opt_metric="gini"):
        
        x_val, y_val = eval_set[0], eval_set[1]

        if feature_names is not None:
            self.feature_names = np.array(feature_names)
        else:
            self.feature_names = np.array( [i for i in range(len(x_val[0]))] )
        
        if isinstance( y_train, list ):
            y_train = np.array( y_train )
        if isinstance( y_val, list ):
            y_val = np.array( y_val )


        if self.optimize_hyperparams:
            self.model = self.optimize_catboost(x_train, y_train, x_val, y_val, 
                                                sample_weight = sample_weight, 
                                                opt_metric = opt_metric )
        else:
            self.model.fit( x_train, y_train, eval_set=[(x_val, y_val)], sample_weight=sample_weight )

        if self.optimize_treshold:
            self.optimize_treshold_(x_val, y_val, n_trials=self.treshold_opt_iters)

        return self

    def predict_proba(self, x):
        probas = self.model.predict_proba( x )
        return probas

    def predict(self, x):

        y_pred_probas = self.predict_proba( x )
        y_pred = []
        for i in range( len(x) ):
            if self.optimal_treshold is None:
                y_i = np.argmax(y_pred_probas[i])
            else:
                y_i = 1 if y_pred_probas[i][1] > self.optimal_treshold else 0
            y_pred.append( y_i )
        y_pred = np.array( y_pred )

        return y_pred

    def get_feature_importance(self):

        feature_importance = self.model.feature_importances_

        return feature_importance

    def optimize_catboost(self, x_train, y_train, x_val, y_val, sample_weight=None, opt_metric=None):

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 1000, 1000),
                "learning_rate": trial.suggest_float("learning_rate", 0.017395, 0.017395),
                "use_best_model": trial.suggest_categorical("use_best_model", [True]),
                "thread_count": trial.suggest_int("thread_count", self.n_jobs, self.n_jobs),
                "early_stopping_rounds": trial.suggest_int("early_stopping_rounds", 100, 100),
                "task_type": trial.suggest_categorical("task_type", ["GPU"]),
                "logging_level": trial.suggest_categorical("logging_level", ["Silent"]),
                "objective": trial.suggest_categorical("objective", ["Logloss"]),
                "max_depth": trial.suggest_int("max_depth", 8, 8),
                "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
                "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]),
                "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 100),
            }

            if params["bootstrap_type"] == "Bayesian":
                params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
            elif params["bootstrap_type"] == "Bernoulli":
                params["subsample"] = trial.suggest_float("subsample", 0.1, 1)

            model = CatBoostClassifier(**params)
            model.fit(x_train, y_train, eval_set=[(x_val, y_val)], sample_weight=sample_weight )
            
            if opt_metric == "gini":
                opt_score = self.gini_score(model, x_val, y_val)
            elif opt_metric == "f1":
                y_pred = model.predict(x_val)
                opt_score = metrics.f1_score( y_val, y_pred, average="weighted" )
            return opt_score

        study = optuna.create_study(directions=["maximize"], sampler=optuna.samplers.MOTPESampler())
        study.optimize( objective, n_trials=self.gbm_opt_iters, timeout=self.max_opt_time, n_jobs=1 )

        best_trials = study.best_trials
        scores = []
        for i in range(len(best_trials)):
            score = best_trials[i].values[0]
            scores.append(score)
        best_trial_id = np.argmax( scores )
        best_trial = best_trials[ best_trial_id ]
        best_params = best_trial.params
        pprint("Best CatBoostClassifier params: {}".format(best_params))
        pprint("Best CatBoostClassifier score: {}".format(best_trial))
        best_params["n_estimators"] = 10000
        best_params["early_stopping_rounds"] = 100

        model = CatBoostClassifier(**best_params)
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)], sample_weight=sample_weight )
        y_pred = model.predict(x_val)
        if opt_metric == "gini":
                opt_score = self.gini_score(model, x_val, y_val)
        elif opt_metric == "f1":
            y_pred = model.predict(x_val)
            opt_score = metrics.f1_score( y_val, y_pred, average="weighted" )
        print("Total val score by retrain={}: {}".format(best_params["n_estimators"], opt_score))

        return model

    def optimize_treshold_(self, x_test, y_test, n_trials=200, opt_metric=None):


        def objective(trial):
            
            current_treshold = trial.suggest_float("treshold", 0.001, 0.999)
            self.optimal_treshold = current_treshold
            
            y_pred = self.predict(x_test)
            if (opt_metric == "f1") or (opt_metric is None):
                opt_score = metrics.f1_score( y_test, y_pred, average="weighted" )

            return opt_score

        study = optuna.create_study(directions=["maximize"], sampler=optuna.samplers.MOTPESampler())
        study.optimize( objective, n_trials=n_trials, n_jobs=1 )

        # get one of the last results where f1 != 0
        best_trials = study.best_trials
        scores = []
        for i in range(len(best_trials)):
            score = best_trials[i].values[0]
            scores.append(score)
        best_trial_id = np.argmax( scores )
        best_trial = best_trials[ best_trial_id ]
        best_params = best_trial.params
        
        self.optimal_treshold = list(best_params.values())[0]
        pprint("Optimal treshold: {}".format(self.optimal_treshold))

        return self
