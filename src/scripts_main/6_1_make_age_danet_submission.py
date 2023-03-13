
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
from sklearn.metrics import roc_auc_score, classification_report, f1_score
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

train_df.drop( columns=["user_id", "age", "is_male"], inplace=True )
feature_names = np.array( train_df.columns )
x = train_df.values
del train_df


submission_df_path = Path(interim_dir, "submission_dataset_{}.feather".format(dataset_postfix))
submission_df = pd.read_feather( submission_df_path )
submission_ids = submission_df["user_id"].values
del submission_df["user_id"]
submission_features = submission_df.values

########################
# age classification
def age_bucket(x):
    #return bisect.bisect_left([18,25,35,45,55,65], x)
    return bisect.bisect_right([19,26,36,46,56,66], x) 
not_nan_mask = ~np.isnan(y_age)
x_age = x[ not_nan_mask ]
y_age = y_age[ not_nan_mask ]
y_age = y_age.astype(np.int64)
##########
##########
y_age = np.array(list(map(age_bucket, y_age)))
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
    scaler.partial_fit( x_age[:, current_feature_id].reshape((-1, 1)) )
    scaler.partial_fit( submission_features[:, current_feature_id].reshape((-1, 1)) )
    x_age[:, current_feature_id] = scaler.transform( x_age[:, current_feature_id].reshape((-1, 1)) ).reshape(-1,)
    submission_features[:, current_feature_id] = scaler.transform( submission_features[:, current_feature_id].reshape((-1, 1)) ).reshape(-1,)
print()
#############

########
# debug
#x_age = x_age[:3000]
#y_age = y_age[:3000]
#######

"""x_train, x_test, y_train, y_test = train_test_split( x_age, y_age, stratify=y_age, test_size=0.10, random_state=45 )
x_train, x_val, y_train, y_val = train_test_split( x_train, y_train, stratify=y_train, test_size=0.10, random_state=45 )

model = DANetClassifier(input_dim = len( x_train[0] ), 
                        num_classes = len( np.unique(y_train) ), 
                        #layer_num=48, base_outdim=96, k=8,
                        #layer_num=32, base_outdim=96, k=8,
                        layer_num=32, base_outdim=64, k=5,
                        virtual_batch_size=256, drop_rate=0.1,
                        device="cuda")

#x_train, y_train = SMOTE(random_state=45, k_neighbors=5, n_jobs=8).fit_resample(x_train, y_train)

model.fit( x_train, y_train, x_val, y_val,
            start_lr=0.008, end_lr=0.0001, batch_size=2048, epochs=200 )
######
save(model, Path(interim_dir, "age_model_partial_fit.pkl"))

model = load(Path(interim_dir, "age_model_partial_fit.pkl"))
#model.optimize_proba_multipliers(x_test, y_test, n_trials=400)
y_pred = model.predict( x_test )
f1 = f1_score(y_test, y_pred, average="weighted")
print("age f1_weighted by bucket score: {}".format( round( f1, 4 ) ))

y_pred = model.predict( x_test )
classify_report = classification_report(y_test, y_pred)
print( classify_report )"""

#feature_importance = model.feature_importances_
#plot_feature_importance(feature_importance, feature_names)
#plt.show()
#################################
###############################
# full retrain

######
#debug
#x_age = x_age[:30000]
#y_age = y_age[:30000]
######

i = 0
cv = 10
val_scores = []
k_fold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=45)
for train_ids, val_ids in tqdm(k_fold.split(x_age, y_age), desc="Fitting cv classifiers"):
    x_t, y_t = x_age[ train_ids ], y_age[ train_ids ]
    x_v, y_v = x_age[ val_ids ], y_age[ val_ids ]
    
    model = DANetClassifier(input_dim = len( x_t[0] ), 
                            num_classes = len( np.unique(y_age) ), 
                            #layer_num=48, base_outdim=96, k=8,
                            #layer_num=32, base_outdim=96, k=8,
                            layer_num=32, base_outdim=64, k=5,
                            virtual_batch_size=256, drop_rate=0.1,
                            device="cuda")
    model.fit( x_t, y_t, x_v, y_v, start_lr=0.008, end_lr=0.0001, batch_size=2048, epochs=100 )
    save( model, Path(models_dir, "danet_age_cv_{}.pkl".format(i)) )
    
    y_pred = model.predict(x_v)
    val_score_i = f1_score(y_v, y_pred, average="weighted")
    val_scores.append( val_score_i )
    
    del model
    gc.collect()
    torch.cuda.empty_cache()
    i += 1
print(val_scores)
print("Mean val score: {}".format(np.mean(val_scores)))

probas = []
for i in tqdm(range(cv), desc="Predicting probas"):
    model = load( Path(models_dir, "danet_age_cv_{}.pkl".format(i)) )
    probas_i = model.predict_proba( submission_features )
    probas.append(probas_i)
    del model
    gc.collect()
    torch.cuda.empty_cache()

y_unique = np.unique(y_age)
submission_predicts = []
for i in tqdm(range(len(submission_features)), desc="Building predicts"):
    probas_i = []
    for j in range(len(probas)):
        probas_j = probas[j][i]
        probas_j = probas_j.reshape((-1, 1))
        probas_i.append( probas_j )
    probas_i = np.hstack(probas_i)
    mean_probas_i = np.mean( probas_i, axis=1 )
    max_proba_id = np.argmax( mean_probas_i )
    predicted_label = y_unique[ max_proba_id ]
    submission_predicts.append( predicted_label )
submission_predicts = np.array( submission_predicts )

submission_ids = submission_ids.reshape((-1, 1))
submission_predicts = submission_predicts.reshape((-1, 1))
submission_data = np.hstack( [submission_ids, submission_predicts] )
my_submission_df = pd.DataFrame( data=submission_data, columns=["user_id", "age"] )
my_submission_df["user_id"] = my_submission_df["user_id"].astype(int)
my_submission_df.to_csv( Path(production_dir, "age_predicts.csv"), index=False )

print("Submission builded and saved")
print("Mean val score (reminder): {}".format(np.mean(val_scores)))

print("done")