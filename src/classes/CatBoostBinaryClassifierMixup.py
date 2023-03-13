
import numpy as np
from tqdm import tqdm
from catboost import CatBoostRegressor

class CatBoostBinaryClassifierMixup():
    def __init__(self):
        
        self.model = CatBoostRegressor(n_estimators=10000, max_depth=8, thread_count=10, early_stopping_rounds=100, task_type="GPU")
        self.y_min = None
        self.y_max = None
        
        pass
    
    def fit(self, x, y, eval_set=None, sample_weight=None, mixup_scaling=10.0, alpha=0.2):
        
        y = y.copy()
        y = y.astype(np.float32)
        
        x_mixup = []
        y_mixup = []
        raw_id = 0
        dataset_size = len(y)
        mixup_size = int( mixup_scaling * len(y) )
        for i in tqdm(range(mixup_size), desc="Generating mixup samples"):
            x_1 = x[raw_id]
            y_1 = y[raw_id]
            
            rand_id = int(np.random.uniform(0.0, dataset_size-1, 1))
            x_2 = x[rand_id]
            y_2 = y[rand_id]
            
            lam = np.random.beta(alpha, alpha)
            x_i = lam * x_1 + (1 - lam) * x_2
            y_i = lam * y_1 + (1 - lam) * y_2
            
            x_mixup.append( x_i )
            y_mixup.append( y_i )
            
            if raw_id == (dataset_size-1):
                raw_id = 0
            else:
                raw_id += 1
            
        x_mixup = np.array( x_mixup )
        y_mixup = np.array( y_mixup )
            
        self.model.fit( x_mixup, y_mixup, eval_set=eval_set )
        
        return self
    
    def predict_proba(self, x):
        
        probas = self.model.predict( x )
        probas = (probas - np.min(probas)) / (np.max(probas) - np.min(probas))
        
        
        return probas
    
    def predict(self, x, treshold=0.5):
        
        y_pred_probas = self.predict_proba( x )
        y_pred = []
        for i in range( len(x) ):
            y_i = 1 if y_pred_probas[i] > treshold else 0
            y_pred.append( y_i )
        y_pred = np.array( y_pred )

        return y_pred
        
