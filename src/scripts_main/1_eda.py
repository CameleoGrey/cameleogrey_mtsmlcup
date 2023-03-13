

import pandas as pd
from pandas_profiling import ProfileReport
from pathlib import Path
from classes.paths_config import *

"""train_df_path = Path( competition_data_dir, "target_train.feather" )
train_df = pd.read_feather( train_df_path )
print(train_df.head(10))
print(train_df.tail(10))
print(train_df.info())

submission_df_path = Path( competition_data_dir, "submission.feather" )
submission_df = pd.read_feather( submission_df_path )
print(submission_df.head(10))
print(submission_df.tail(10))
print(submission_df.info())


full_data_path = Path( competition_data_dir, "dataset_full.feather" )
full_data_df = pd.read_feather( full_data_path )
full_data_df.sort_values("user_id", inplace=True)
print(full_data_df.head(10))
print(full_data_df.tail(10))
print(full_data_df.info())"""

data_subsample_path = Path( interim_dir, "data_subsample.feather" )
data_subsample_df = pd.read_feather( data_subsample_path )
data_subsample_df.sort_values("user_id", inplace=True)
print(data_subsample_df.head(10))
print(data_subsample_df.tail(10))
print(data_subsample_df.info())

design_report = ProfileReport(data_subsample_df)
design_report.to_file(output_file=Path(plots_dir, "0_eda_raw_report.html"))


print("done")
