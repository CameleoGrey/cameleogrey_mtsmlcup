
import gc
import numpy as np
import pandas as pd
import bisect
import torch

from pathlib import Path
from tqdm import tqdm

from classes.paths_config import *
from classes.utils import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from classes.danet.DANetClassifier import DANetClassifier
from imblearn.over_sampling import SMOTE, ADASYN

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
    selected_feature_ids.append(i)
selected_feature_ids = np.array( selected_feature_ids )

for i in tqdm(range(len(selected_feature_ids)), desc="Scaling selected features"):
    current_feature_id = selected_feature_ids[i]
    
    # contaminated version (using features from unclassified samples to improve scaling)
    scaler  = StandardScaler()
    scaler.partial_fit( x_is_male[:, current_feature_id].reshape((-1, 1)) )
    scaler.partial_fit( submission_features[:, current_feature_id].reshape((-1, 1)) )
    x_is_male[:, current_feature_id] = scaler.transform( x_is_male[:, current_feature_id].reshape((-1, 1)) ).reshape(-1,)
    submission_features[:, current_feature_id] = scaler.transform( submission_features[:, current_feature_id].reshape((-1, 1)) ).reshape(-1,)
    
print()
#############

######
#debug
#x_is_male = x_is_male[:30000]
#y_is_male = y_is_male[:30000]
######

i = 0
cv = 10
val_scores = []
k_fold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=45)
for train_ids, val_ids in tqdm(k_fold.split(x_is_male, y_is_male), desc="Fitting cv classifiers"):
    x_t, y_t = x_is_male[ train_ids ], y_is_male[ train_ids ]
    x_v, y_v = x_is_male[ val_ids ], y_is_male[ val_ids ]
    
    model = DANetClassifier(input_dim = len( x_t[0] ), 
                            num_classes = len( np.unique(y_is_male) ), 
                            #layer_num=48, base_outdim=96, k=8,
                            #layer_num=32, base_outdim=96, k=8,
                            layer_num=32, base_outdim=64, k=5,
                            virtual_batch_size=256, drop_rate=0.1,
                            device="cuda")
    model.fit( x_t, y_t, x_v, y_v, start_lr=0.008, end_lr=0.0001, batch_size=2048, epochs=100 )
    save( model, Path(models_dir, "danet_is_male_cv_{}.pkl".format(i)) )
    
    val_score_i = age_score(model, x_v, y_v)
    val_scores.append( val_score_i )
    
    del model
    gc.collect()
    torch.cuda.empty_cache()
    i += 1
print(val_scores)
print("Mean val score: {}".format(np.mean(val_scores)))

probas = []
for i in tqdm(range(cv), desc="Predicting probas"):
    model = load( Path(models_dir, "danet_is_male_cv_{}.pkl".format(i)) )
    probas_i = model.predict_proba( submission_features )[:, 1]
    probas_i = probas_i.reshape((-1, 1))
    probas.append(probas_i)
    del model
    gc.collect()
    torch.cuda.empty_cache()
probas = np.hstack(probas)
mean_probas = np.mean( probas, axis=1 )

submission_ids = submission_ids.reshape((-1, 1))
submission_predicts = mean_probas.reshape((-1, 1))
submission_data = np.hstack( [submission_ids, submission_predicts] )
my_submission_df = pd.DataFrame( data=submission_data, columns=["user_id", "is_male"] )
my_submission_df["user_id"] = my_submission_df["user_id"].astype(int)
my_submission_df.to_csv( Path(production_dir, "is_male_predicts.csv"), index=False )

print("Submission builded and saved")
print("Mean val score (reminder): {}".format(np.mean(val_scores)))

print("done")