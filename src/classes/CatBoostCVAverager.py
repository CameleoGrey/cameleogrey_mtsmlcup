
import numpy as np
from catboost import CatBoostClassifier
from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold

class CatBoostCVAverager():
    def __init__(self):
        
        self.base_models = []
        self.tree_counts = []
        
        pass
    
    def fit(self, x, y, cv=10, add_full_retrain=True, n_estimators=10000, learning_rate=0.017395, max_depth=8, 
            thread_count=10, early_stopping_rounds=100, task_type="GPU"):
        
        k_fold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=45)
        for train_ids, val_ids in tqdm(k_fold.split(x, y), desc="Fitting cv classifiers"):
            x_t, y_t = x[ train_ids ], y[ train_ids ]
            x_v, y_v = x[ val_ids ], y[ val_ids ]
            
            model = CatBoostClassifier( n_estimators=n_estimators, 
                                        learning_rate=learning_rate, 
                                        max_depth=max_depth, 
                                        thread_count=thread_count, 
                                        early_stopping_rounds=early_stopping_rounds, 
                                        task_type=task_type )
            model.fit( x_t, y_t, eval_set=(x_v, y_v), verbose=False )
            
            self.base_models.append( model )
            self.tree_counts.append( model.tree_count_ )
        
        if add_full_retrain:
            print("Full retraining...")
            mean_cv_tree_count = np.mean( self.tree_counts )
            full_retrain_estimators = (float(cv) / float((cv - 1))) * mean_cv_tree_count
            full_retrain_estimators = int( full_retrain_estimators )
            model = CatBoostClassifier( n_estimators=full_retrain_estimators, 
                                        learning_rate=learning_rate, 
                                        max_depth=max_depth, 
                                        thread_count=thread_count, 
                                        early_stopping_rounds=early_stopping_rounds, 
                                        task_type=task_type )
            model.fit( x, y, verbose=False )
            self.base_models.append( model )
            
    def predict_proba(self, x):
        
        y_pred_proba_batches = []
        for model in tqdm(self.base_models, desc="Predicting probas"):
            pred_proba = model.predict_proba( x )
            y_pred_proba_batches.append( pred_proba )

        y_pred_proba = []
        for j in range( len(x) ):
            mean_probas = []
            for i in range( len(y_pred_proba_batches) ):
                mean_probas.append( y_pred_proba_batches[i][j] )
            mean_probas = np.mean( mean_probas, axis=0 )
            y_pred_proba.append( mean_probas )
        y_pred_proba = np.array( y_pred_proba )
        
        return y_pred_proba
    
    def predict(self, x):

        y_pred_probas = self.predict_proba( x )
        y_pred = []
        for i in range( len(x) ):
            y_i = np.argmax(y_pred_probas[i])
            y_pred.append( y_i )
        y_pred = np.array( y_pred )

        return y_pred
            
            
            
            