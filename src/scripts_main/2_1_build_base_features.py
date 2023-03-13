

import gc
import pandas as pd
import pyarrow as pa
from pyarrow import feather
from pathlib import Path

from classes.paths_config import *
from classes.utils import *
from classes.DataCleaner import DataCleaner
from classes.FeatureBuilder import FeatureBuilder

#from pandas_profiling import ProfileReport

full_data_path = Path( competition_data_dir, "dataset_full.feather" )
full_data_df = pd.read_feather( full_data_path )
df = full_data_df

# debug
#data_subsample_path = Path( interim_dir, "data_subsample.feather" )
#data_subsample_df = pd.read_feather( data_subsample_path )
#df = data_subsample_df

data_cleaner = DataCleaner()
#df = data_cleaner.unite_cathegories( df )
df = data_cleaner.fill_missing_price( df )

feature_builder = FeatureBuilder()

df = feature_builder.log_price( df )
user_price_agg_dict = feature_builder.collect_price_aggregates( df )
save(user_price_agg_dict, Path(interim_dir, "user_price_agg_dict.pkl"))

df = feature_builder.add_day_names( df )
df = feature_builder.crop_part_of_day( df )
df = feature_builder.add_day_name_part_feature( df )

print("Caching stage 1")
df.to_feather( Path(interim_dir, "stage_1.feather") )
del df
gc.collect()
print("Stage 1 cached")

print("Reading stage 1")
df = pd.read_feather( Path(interim_dir, "stage_1.feather") )
print("Stage 1 is ready")

count_agg_feats = feature_builder.collect_count_aggregates( df )
save(count_agg_feats, Path(interim_dir, "count_agg_feats.pkl"))
del count_agg_feats
gc.collect()

base_cat_feature_names = ["region_name", "city_name", "cpe_manufacturer_name", 
                          "cpe_model_name", "cpe_type_cd", "cpe_model_os_type",
                          "part_of_day", "day_name", "day_name_part"]

feature_builder.fit_cat_encoders(df, 
                                 feature_names=base_cat_feature_names, 
                                 embedding_sizes=[1, 1, 1, 1, 1, 1, 1, 1, 1],
                                 #embedding_sizes=[5, 5, 5, 5, 1, 1, 1, 1, 5], 
                                 shuffle_counts=[1, 1, 1, 1, 1, 1, 1, 1, 1])
save(feature_builder, Path(interim_dir, "feature_builder_cat_fit.pkl"))
del feature_builder
gc.collect()
feature_builder = load( Path(interim_dir, "feature_builder_cat_fit.pkl") )

df = feature_builder.transform_cat_features( df )

print("Caching stage 2")
df.to_feather( Path(interim_dir, "stage_2.feather") )
del df
gc.collect()
print("Stage 2 cached")

df = pd.read_feather( Path(interim_dir, "stage_2.feather") )
cat_feats_aggregates = feature_builder.collect_encoded_cat_features_aggregates( df, base_cat_feature_names )
save(cat_feats_aggregates, Path(interim_dir, "cat_feats_aggregates.pkl"))
del df
gc.collect()

feature_builder = load( Path(interim_dir, "feature_builder_cat_fit.pkl") )
df = pd.read_feather( Path(interim_dir, "stage_2.feather"), columns=["url_host", "user_id", "request_cnt"] )
url_frequency_dict = feature_builder.build_url_frequency_dict( df )
save( url_frequency_dict, Path(interim_dir, "url_frequency_dict.pkl") )
del url_frequency_dict
gc.collect()


#profile_report = ProfileReport(df)
#profile_report.to_file(output_file=Path(plots_dir, "1_eda_preprocessed_features_report.html"))


print("done")
