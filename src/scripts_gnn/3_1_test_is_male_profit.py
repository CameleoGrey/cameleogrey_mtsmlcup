
import numpy as np
import pandas as pd
import bisect

from pathlib import Path
from tqdm import tqdm

from classes.paths_config import *
from classes.utils import *
from classes.GreyFeatureSelector import GreyFeatureSelector
from classes.danet.DANetClassifier import DANetClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler

from catboost import CatBoostClassifier

embedder_postfix = "w20_300_100_l40_c40"
dataset_postfix = "node2vec_{}".format(embedder_postfix)
train_df_path = Path(interim_dir, "train_dataset_{}.feather".format(dataset_postfix))
train_df = pd.read_feather( train_df_path )

user_ids = train_df["user_id"].values
y_age = train_df["age"].values
y_is_male = train_df["is_male"].values

train_df.drop( columns=["user_id", "age", "is_male"], inplace=True )
feature_names = np.array( train_df.columns )
x = train_df.values
del train_df

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
    # contaminated version (using features from unclassified samples to improve scaling)
    scaler  = StandardScaler()
    scaler.partial_fit( x_is_male[:, current_feature_id].reshape((-1, 1)) )
    x_is_male[:, current_feature_id] = scaler.transform( x_is_male[:, current_feature_id].reshape((-1, 1)) ).reshape(-1,)
#############

###################################
# finding best model hyper parameters
x_train, x_test, y_train, y_test = train_test_split( x_is_male, y_is_male, stratify=y_is_male, test_size=0.1, random_state=45 )
x_train, x_val, y_train, y_val = train_test_split( x_train, y_train, stratify=y_train, test_size=0.1, random_state=45 )

model = CatBoostClassifier(n_estimators=10000, learning_rate=0.017395, max_depth=8, thread_count=10, early_stopping_rounds=100, task_type="GPU", gpu_ram_part=0.4)
model.fit( x_train, y_train, eval_set=(x_val, y_val) )

def age_score(model, x, y):
    y_pred = model.predict_proba( x )
    if len(y_pred.shape) == 2:
        y_pred = y_pred[:, 1]
    gini_score = 2.0 * roc_auc_score(y, y_pred) - 1.0
    return gini_score
gini_score = age_score(model, x_test, y_test)
print("is_male Gini: {}".format( round( gini_score, 4 ) ))

y_pred = model.predict( x_test )
classify_report = classification_report(y_test, y_pred)
print( classify_report )
