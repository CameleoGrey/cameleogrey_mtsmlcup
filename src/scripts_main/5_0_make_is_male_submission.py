
import numpy as np
import pandas as pd
import bisect

from pathlib import Path
from tqdm import tqdm

from classes.paths_config import *
from classes.utils import *
from classes.GreyFeatureSelector import GreyFeatureSelector
#from classes.GreyAutoencoder1d import GreyAutoencoder1d

from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.cluster import KMeans
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.utils import compute_sample_weight

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from classes.CatBoostBinarySelfOptimized import CatBoostBinarySelfOptimized
from classes.CatBoostBinaryClassifierMixup import CatBoostBinaryClassifierMixup
from classes.CatBoostCVAverager import CatBoostCVAverager
from classes.danet.DANetClassifier import DANetClassifier

from imblearn.over_sampling import SMOTE, ADASYN
from category_encoders import TargetEncoder

from matplotlib import pyplot as plt
import seaborn as sns

user_factor_dict_postfix = "50_100_400_20_1"
w2v_user_url_feats_postfix = "20_300_100"
graph_feats_dict_postfix = "node2vec_w20_300_100_l40_c40"
content_dict_postfix = "bert"

dataset_postfix = "_".join([
    user_factor_dict_postfix, 
    w2v_user_url_feats_postfix, 
    graph_feats_dict_postfix, 
    content_dict_postfix 
])

train_df_path = Path(interim_dir, "train_dataset_{}.feather".format(dataset_postfix))
train_df = pd.read_feather( train_df_path )

user_ids = train_df["user_id"].values
y_age = train_df["age"].values
y_is_male = train_df["is_male"].values

train_df.drop( columns=["user_id", "age", "is_male"], inplace=True )
feature_names = np.array( train_df.columns )
x = train_df.values
del train_df


submission_df_path = Path(interim_dir, "submission_dataset_{}.feather".format(dataset_postfix))
submission_df = pd.read_feather( submission_df_path )
submission_ids = submission_df["user_id"].values
del submission_df["user_id"]
submission_features = submission_df.values


def plot_feature_importance(importance, feature_names):
    
    #Create arrays from feature importance and feature names
    feature_importance = np.array( importance )
    
    #Create a DataFrame using a Dictionary
    data={'feature_names': feature_names, 'feature_importance': feature_importance}
    fi_df = pd.DataFrame(data)
    
    #Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values( by=['feature_importance'], ascending=False, inplace=True)
    
    #Define size of bar plot
    plt.figure(figsize=(10,8))
    #Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    #Add chart labels
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')

def age_score(model, x, y):
    y_pred = model.predict_proba( x )
    if len(y_pred.shape) == 2:
        y_pred = y_pred[:, 1]
    gini_score = 2.0 * roc_auc_score(y, y_pred) - 1.0
    return gini_score

########################
# is_male classification
y_is_male = y_is_male.astype(np.float64)
not_nan_mask = ~np.isnan(y_is_male)
x_is_male = x[ not_nan_mask ]
y_is_male = y_is_male[ not_nan_mask ]
y_is_male = y_is_male.astype(np.int64)

#############
# power transforming and scaling features except url embeddings 
selected_feature_ids = []
for i in range(len(feature_names)):
    current_feature_name = feature_names[i]
    #if "url" in current_feature_name:
    #    continue
    #if "factor" in current_feature_name:
    #    continue
    #if "graph" in current_feature_name:
    #    continue
    selected_feature_ids.append(i)
selected_feature_ids = np.array( selected_feature_ids )

for i in tqdm(range(len(selected_feature_ids)), desc="Scaling selected features"):
    current_feature_id = selected_feature_ids[i]
    
    #power_transformer = PowerTransformer(method="yeo-johnson", standardize=False)
    #power_transformer.fit( x_train_male[:, current_feature_id].reshape((-1, 1)))
    #x_train_male[:, current_feature_id] = power_transformer.transform( x_train_male[:, current_feature_id].reshape((-1, 1)) ).reshape(-1,)
    #x_test_male[:, current_feature_id] = power_transformer.transform( x_test_male[:, current_feature_id].reshape((-1, 1)) ).reshape(-1,)
    #x_val_male[:, current_feature_id] = power_transformer.transform( x_val_male[:, current_feature_id].reshape((-1, 1)) ).reshape(-1,)
    
    # fair scaling (only on labeled data)
    """scaler  = StandardScaler()    
    scaler.fit( x_train_male[:, current_feature_id].reshape((-1, 1)))
    x_train_male[:, current_feature_id] = scaler.transform( x_train_male[:, current_feature_id].reshape((-1, 1)) ).reshape(-1,)
    x_test_male[:, current_feature_id] = scaler.transform( x_test_male[:, current_feature_id].reshape((-1, 1)) ).reshape(-1,)
    x_val_male[:, current_feature_id] = scaler.transform( x_val_male[:, current_feature_id].reshape((-1, 1)) ).reshape(-1,)"""
    
    # contaminated version (using features from unclassified samples to improve scaling)
    scaler  = StandardScaler()
    scaler.partial_fit( x_is_male[:, current_feature_id].reshape((-1, 1)) )
    scaler.partial_fit( submission_features[:, current_feature_id].reshape((-1, 1)) )
    x_is_male[:, current_feature_id] = scaler.transform( x_is_male[:, current_feature_id].reshape((-1, 1)) ).reshape(-1,)
    submission_features[:, current_feature_id] = scaler.transform( submission_features[:, current_feature_id].reshape((-1, 1)) ).reshape(-1,)
    
print()
#############

###################################
# finding best model hyper parameters
x_train_male, x_test_male, y_train_male, y_test_male = train_test_split( x_is_male, y_is_male, stratify=y_is_male, test_size=0.1, random_state=45 )
x_train_male, x_val_male, y_train_male, y_val_male = train_test_split( x_train_male, y_train_male, stratify=y_train_male, test_size=0.1, random_state=45 )

model = CatBoostClassifier(n_estimators=10000, learning_rate=0.017395, max_depth=8, thread_count=10, early_stopping_rounds=100, task_type="GPU")
#model = ExtraTreesClassifier(n_estimators=400, max_depth=40, n_jobs=10, verbose=1)
#model = CalibratedClassifierCV( CatBoostClassifier(n_estimators=1000, max_depth=8, thread_count=10, early_stopping_rounds=100, task_type="GPU"), cv=10 )
#model = LGBMClassifier( n_estimators=100, n_jobs=10 )
"""model = CatBoostBinarySelfOptimized(n_jobs=8,
                                    optimize_treshold=False, treshold_opt_iters=200,
                                    optimize_hyperparams=True, gbm_opt_iters=200, max_opt_time=2*60*60)"""
#model = CatBoostBinaryClassifierMixup()
#model = CatBoostCVAverager()
"""model = DANetClassifier(input_dim = len( x_train_male[0] ), 
                        num_classes = len( np.unique(y_train_male) ), 
                        #layer_num=48, base_outdim=96, k=8,
                        #layer_num=32, base_outdim=96, k=8,
                        layer_num=32, base_outdim=96, k=8,
                        virtual_batch_size=256, drop_rate=0.1,
                        device="cuda")"""

#x_train_male, y_train_male = SMOTE(random_state=45, k_neighbors=5, n_jobs=8).fit_resample(x_train_male, y_train_male)

#sample_weight = compute_sample_weight(class_weight="balanced", y=y_train_male)
#sample_weight = np.sqrt( sample_weight )
#######
# debug
#x_train_male = x_train_male[:3000]
#y_train_male = y_train_male[:3000]
#x_val_male = x_val_male[:300]
#y_val_male = y_val_male[:300]
#######
model.fit( x_train_male, y_train_male, eval_set=(x_val_male, y_val_male) )
######
#x_train_cv = np.vstack([x_train_male, x_val_male])
#y_train_cv = np.hstack([y_train_male, y_val_male])
#model.fit(x_train_cv, y_train_cv, cv=10, add_full_retrain=True, n_estimators=10000, learning_rate=0.017395, max_depth=8, 
#          thread_count=10, early_stopping_rounds=100, task_type="GPU")
#model.fit(x_train_cv, y_train_cv, cv=10)
######
#model.fit( x_train_male, y_train_male, x_val_male, y_val_male,
#            start_lr=0.008, end_lr=0.0001, batch_size=2048, epochs=100 )
######
save(model, Path(interim_dir, "is_male_model_partial_fit.pkl"))

model = load(Path(interim_dir, "is_male_model_partial_fit.pkl"))
#####
#model.optimize_mix_weights(x_test_male, y_test_male, opt_metric="gini", n_trials=200)
#####

gini_score = age_score(model, x_test_male, y_test_male)
print("is_male Gini: {}".format( round( gini_score, 4 ) ))

y_pred = model.predict( x_test_male )
classify_report = classification_report(y_test_male, y_pred)
print( classify_report )

#feature_importance = model.feature_importances_
#plot_feature_importance(feature_importance, feature_names)
#plt.show()
###############################
# full retrain
"""x_full = np.vstack([x_train_male, x_val_male, x_test_male])
y_full = np.hstack([y_train_male, y_val_male, y_test_male])
#model = CatBoostClassifier(n_estimators=7750, max_depth=8, learning_rate=0.017395, thread_count=10, task_type="GPU")
#model.fit( x_full, y_full )
model = CatBoostCVAverager()
model.fit(x_full, y_full, cv=10, add_full_retrain=True, n_estimators=10000, learning_rate=0.017395, max_depth=8, 
          thread_count=10, early_stopping_rounds=100, task_type="GPU")
save(model, Path(interim_dir, "is_male_model_full_fit.pkl"))

###############################

model = load(Path(interim_dir, "is_male_model_full_fit.pkl"))
submission_predicts = model.predict_proba( submission_features )[:, 1]
submission_ids = submission_ids.reshape((-1, 1))
submission_predicts = submission_predicts.reshape((-1, 1))
submission_data = np.hstack( [submission_ids, submission_predicts] )
my_submission_df = pd.DataFrame( data=submission_data, columns=["user_id", "is_male"] )
my_submission_df["user_id"] = my_submission_df["user_id"].astype(int)
my_submission_df.to_csv( Path(production_dir, "is_male_predicts.csv"), index=False )"""

print("done")