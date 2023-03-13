
import numpy as np
import pandas as pd
import bisect

from pathlib import Path
from tqdm import tqdm

from classes.paths_config import *
from classes.utils import *

from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, f1_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.cluster import KMeans
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.utils import compute_sample_weight

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from classes.CatBoostBinarySelfOptimized import CatBoostBinarySelfOptimized
from classes.CatBoostMulticlassOptimal import CatBoostMulticlassOptimal
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
    #if "graph" in current_feature_name:
    #    continue
    selected_feature_ids.append(i)
selected_feature_ids = np.array( selected_feature_ids )

for i in tqdm(range(len(selected_feature_ids)), desc="Scaling selected features"):
    current_feature_id = selected_feature_ids[i]
    
    #power_transformer = PowerTransformer(method="yeo-johnson", standardize=False)
    #power_transformer.fit( x_train_age[:, current_feature_id].reshape((-1, 1)))
    #x_train_age[:, current_feature_id] = power_transformer.transform( x_train_age[:, current_feature_id].reshape((-1, 1)) ).reshape(-1,)
    #x_test_age[:, current_feature_id] = power_transformer.transform( x_test_age[:, current_feature_id].reshape((-1, 1)) ).reshape(-1,)
    #x_val_age[:, current_feature_id] = power_transformer.transform( x_val_age[:, current_feature_id].reshape((-1, 1)) ).reshape(-1,)
    
    # fair scaling (only on labeled data)
    """scaler  = StandardScaler()    
    scaler.fit( x_train_age[:, current_feature_id].reshape((-1, 1)))
    x_train_age[:, current_feature_id] = scaler.transform( x_train_age[:, current_feature_id].reshape((-1, 1)) ).reshape(-1,)
    x_test_age[:, current_feature_id] = scaler.transform( x_test_age[:, current_feature_id].reshape((-1, 1)) ).reshape(-1,)
    x_val_age[:, current_feature_id] = scaler.transform( x_val_age[:, current_feature_id].reshape((-1, 1)) ).reshape(-1,)"""
    
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

x_train, x_test, y_train, y_test = train_test_split( x_age, y_age, stratify=y_age, test_size=0.10, random_state=45 )
x_train, x_val, y_train, y_val = train_test_split( x_train, y_train, stratify=y_train, test_size=0.10, random_state=45 )

"""rare_class_mask = (y_train == 0) | (y_train == 6)
x_train = x_train[~rare_class_mask]
y_train = y_train[~rare_class_mask]

rare_class_mask = (y_val == 0) | (y_val == 6)
x_val = x_val[~rare_class_mask]
y_val = y_val[~rare_class_mask]"""

model = CatBoostClassifier(n_estimators=10000, learning_rate=0.066245, max_depth=8, thread_count=10, early_stopping_rounds=100, task_type="GPU")
#model = ExtraTreesClassifier(n_estimators=400, max_depth=40, n_jobs=10, verbose=1)
#model = CalibratedClassifierCV( CatBoostClassifier(n_estimators=1000, max_depth=8, thread_count=10, early_stopping_rounds=100, task_type="GPU"), cv=10 )
#model = LGBMClassifier( n_estimators=100, n_jobs=10 )
#model = CatBoostMulticlassOptimal()
#model = CatBoostCVAverager()
"""model = DANetClassifier(input_dim = len( x_train[0] ), 
                        num_classes = len( np.unique(y_train) ), 
                        #layer_num=48, base_outdim=96, k=8,
                        #layer_num=32, base_outdim=96, k=8,
                        layer_num=32, base_outdim=64, k=5,
                        virtual_batch_size=256, drop_rate=0.1,
                        device="cuda")"""

#x_train, y_train = SMOTE(random_state=45, k_neighbors=5, n_jobs=8).fit_resample(x_train, y_train)

#sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
#sample_weight = np.sqrt( sample_weight )
#model.fit( x_train, y_train, eval_set=(x_val, y_val) )
######
#x_train_cv = np.vstack([x_train, x_val])
#y_train_cv = np.hstack([y_train, y_val])
#model.fit(x_train_cv, y_train_cv, cv=10, add_full_retrain=True, n_estimators=10000, learning_rate=0.066245, max_depth=8, 
#          thread_count=10, early_stopping_rounds=100, task_type="GPU")
# DANet
#model.fit( x_train, y_train, x_val, y_val, start_lr=0.008, end_lr=0.0001, batch_size=2048, epochs=200 )
######
save(model, Path(interim_dir, "age_model_partial_fit.pkl"))

model = load(Path(interim_dir, "age_model_partial_fit.pkl"))
#CatBoostOptimal
#model.optimize_proba_multipliers(x_test, y_test, n_trials=400)
######
y_pred = model.predict( x_test )
f1 = f1_score(y_test, y_pred, average="weighted")
print("age f1_weighted by bucket score: {}".format( round( f1, 4 ) ))

y_pred = model.predict( x_test )
classify_report = classification_report(y_test, y_pred)
print( classify_report )

#feature_importance = model.feature_importances_
#plot_feature_importance(feature_importance, feature_names)
#plt.show()
#################################
"""x_full = np.vstack([x_train, x_val, x_test])
y_full = np.hstack([y_train, y_val, y_test])
#sample_weight = compute_sample_weight(class_weight="balanced", y=y_full)
#sample_weight = np.sqrt( sample_weight )
#model = CatBoostClassifier(n_estimators=1300, max_depth=8, learning_rate=0.066245, thread_count=10, task_type="GPU")
#model.fit( x_full, y_full, sample_weight=None )
model = CatBoostCVAverager()
model.fit(x_full, y_full, cv=10, add_full_retrain=True, n_estimators=10000, learning_rate=0.066245, max_depth=8, 
          thread_count=10, early_stopping_rounds=100, task_type="GPU")
save(model, Path(interim_dir, "age_model_full_fit.pkl"))
y_pred = model.predict( x_test )
classify_report = classification_report(y_test, y_pred)
print( classify_report )

#################################
model = load(Path(interim_dir, "age_model_full_fit.pkl"))
submission_predicts = model.predict( submission_features )
submission_ids = submission_ids.reshape((-1, 1))
submission_predicts = submission_predicts.reshape((-1, 1))
submission_data = np.hstack( [submission_ids, submission_predicts] )
my_submission_df = pd.DataFrame( data=submission_data, columns=["user_id", "age"] )
my_submission_df["user_id"] = my_submission_df["user_id"].astype(int)
my_submission_df.to_csv( Path(production_dir, "age_predicts.csv"), index=False )"""

print("done")