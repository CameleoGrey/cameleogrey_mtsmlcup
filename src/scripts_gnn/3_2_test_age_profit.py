
import numpy as np
import pandas as pd
import bisect

from pathlib import Path
from tqdm import tqdm

from classes.paths_config import *
from classes.utils import *

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, f1_score

from catboost import CatBoostClassifier

embedder_postfix = "w20_300_100_l40_c40"
dataset_postfix = "node2vec_{}".format(embedder_postfix)
train_df_path = Path(interim_dir, "train_dataset_{}.feather".format(dataset_postfix))
train_df = pd.read_feather( train_df_path )

user_ids = train_df["user_id"].values
y_age = train_df["age"].values

train_df.drop( columns=["user_id", "age", "is_male"], inplace=True )
feature_names = np.array( train_df.columns )
x = train_df.values
del train_df

########################
# age classification
def age_bucket(x):
    #return bisect.bisect_left([18,25,35,45,55,65], x)
    return bisect.bisect_right([19,26,36,46,56,66], x) 
not_nan_mask = ~np.isnan(y_age)
x_age = x[ not_nan_mask ]
y_age = y_age[ not_nan_mask ]
y_age = y_age.astype(np.int64)
y_age = np.array(list(map(age_bucket, y_age)))
########################

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
    scaler  = StandardScaler()
    scaler.partial_fit( x_age[:, current_feature_id].reshape((-1, 1)) )
    x_age[:, current_feature_id] = scaler.transform( x_age[:, current_feature_id].reshape((-1, 1)) ).reshape(-1,)
print()
#############

x_train, x_test, y_train, y_test = train_test_split( x_age, y_age, stratify=y_age, test_size=0.10, random_state=45 )
x_train, x_val, y_train, y_val = train_test_split( x_train, y_train, stratify=y_train, test_size=0.10, random_state=45 )

model = CatBoostClassifier(n_estimators=10000, learning_rate=0.066245, max_depth=8, thread_count=10, early_stopping_rounds=100, task_type="GPU", gpu_ram_part=0.4)
model.fit( x_train, y_train, eval_set=(x_val, y_val) )
save(model, Path(interim_dir, "age_model_partial_fit.pkl"))

model = load(Path(interim_dir, "age_model_partial_fit.pkl"))
y_pred = model.predict( x_test )
f1 = f1_score(y_test, y_pred, average="weighted")
print("age f1_weighted by bucket score: {}".format( round( f1, 4 ) ))

y_pred = model.predict( x_test )
classify_report = classification_report(y_test, y_pred)
print( classify_report )

print("done")