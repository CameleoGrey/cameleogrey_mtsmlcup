
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from classes.paths_config import *
from classes.utils import *

embedder_postfix = "w20_300_100_l40_c40"
dataset_postfix = "node2vec_{}".format(embedder_postfix)
user_graph_feat_dict = load(Path(interim_dir, "user_graph_feat_dict_{}.pkl".format(dataset_postfix)))

feature_columns = user_graph_feat_dict["feature_names"] 
feature_vectors = []
user_ids = []
for user_id in tqdm(user_graph_feat_dict.keys(), desc="Building merged features"):
    if user_id == "feature_names":
        continue
    
    user_ids.append( user_id )
    user_graph_feat = user_graph_feat_dict[ user_id ]
    user_common_vector = user_graph_feat
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