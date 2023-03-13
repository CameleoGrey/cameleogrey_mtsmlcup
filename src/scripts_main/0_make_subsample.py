
import numpy as np
import pandas as pd
from pathlib import Path
from classes.paths_config import *

# make data subsample for debug purposes

full_data_path = Path( competition_data_dir, "dataset_full.feather" )
full_data_df = pd.read_feather( full_data_path )
print(full_data_df.head(10))
print(full_data_df.tail(10))
print(full_data_df.info())

user_ids = full_data_df["user_id"].values
unique_user_ids = np.unique( user_ids )
np.random.seed(45)
np.random.shuffle( unique_user_ids )
ids_for_subsample = unique_user_ids[:1000]

ids_for_subsample = set(ids_for_subsample)
data_subsample = full_data_df[ full_data_df["user_id"].isin( ids_for_subsample ) ]
data_subsample.reset_index( drop=True, inplace=True )
data_subsample.to_feather( Path(interim_dir, "data_subsample.feather") )


print("done")
