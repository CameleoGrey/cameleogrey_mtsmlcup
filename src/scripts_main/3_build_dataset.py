
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from classes.paths_config import *
from classes.utils import *

#from pandas_profiling import ProfileReport

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

user_price_agg_dict = load(Path(interim_dir, "user_price_agg_dict.pkl"))
count_agg_feats = load(Path(interim_dir, "count_agg_feats.pkl"))
cat_feats_aggregates = load(Path(interim_dir, "cat_feats_aggregates.pkl"))
user_factor_features_dict = load( Path(interim_dir, "user_factor_features_dict_{}.pkl".format( user_factor_dict_postfix )) )
user_urls_feat_dict = load(Path(interim_dir, "user_urls_feat_dict_{}.pkl".format( w2v_user_url_feats_postfix )))
graph_features_dict = load(Path(interim_dir, "user_graph_feat_dict_{}.pkl".format( graph_feats_dict_postfix )))
content_features_dict = load(Path(interim_dir, "user_url_content_vectors_dict_{}.pkl".format( content_dict_postfix )))

feat_dicts = [
    user_price_agg_dict,
    count_agg_feats,
    cat_feats_aggregates,
    user_factor_features_dict,
    user_urls_feat_dict,
    graph_features_dict,
    content_features_dict
]

feature_columns = []
for feat_dict in feat_dicts:
    feature_columns += feat_dict["feature_names"]

feature_vectors = []
user_ids = []
for user_id in tqdm(feat_dicts[0].keys(), desc="Building merged features"):
    if user_id == "feature_names":
        continue
    user_common_vector = []
    user_ids.append( user_id )
        
    for feat_dict in feat_dicts:
        user_common_vector.append( feat_dict[user_id] )
        
    user_common_vector = np.hstack( user_common_vector )
    feature_vectors.append( user_common_vector )
feature_vectors = np.array( feature_vectors )
print(feature_vectors.shape)
users_dataset = pd.DataFrame( data=feature_vectors, columns=feature_columns )
users_dataset["user_id"] = user_ids

target_train_df_path = Path( competition_data_dir, "target_train.feather" )
target_train_df = pd.read_feather( target_train_df_path )
users_train_dataset = users_dataset.merge(target_train_df, how="inner", on=["user_id"])
if len(users_train_dataset) != len(target_train_df):
    print("Warning! target_train_df size: {} | users_train_dataset size: {}".format(len(target_train_df), len(users_train_dataset)))
users_train_dataset.reset_index(drop=True, inplace=True)
users_train_dataset.to_feather(Path(interim_dir, "train_dataset_{}.feather".format(dataset_postfix)))
#profile_report = ProfileReport(users_train_dataset, minimal=True)
#profile_report.to_file(output_file=Path(plots_dir, "3_users_train_dataset_report.html"))

submission_df_path = Path( competition_data_dir, "submission.feather" )
submission_df = pd.read_feather( submission_df_path )
submission_dataset = users_dataset.merge(submission_df, how="inner", on=["user_id"])
submission_dataset.reset_index(drop=True, inplace=True)
if len(submission_dataset) != len(submission_df):
    print("Warning! Submission size: {} | Submission dataset size: {}".format(len(submission_df), len(submission_dataset)))
submission_dataset.to_feather(Path(interim_dir, "submission_dataset_{}.feather".format(dataset_postfix)))
#profile_report = ProfileReport(submission_dataset, minimal=True)
#profile_report.to_file(output_file=Path(plots_dir, "3_users_submission_dataset_report.html"))

print("done")