
import pandas as pd
import numpy as np
import implicit
from pathlib import Path
from tqdm import tqdm
from scipy.sparse import coo_matrix, csr_matrix

from classes.utils import *
from classes.paths_config import *
from classes.UserFeatDictBuilder import UserFeatDictBuilder

url_frequency_dict = load( Path(interim_dir, "url_frequency_dict.pkl") )

#keys = list(url_frequency_dict.keys())[:100]
#subsample_dict = {}
#for key in keys:
#    subsample_dict[key] = url_frequency_dict[key]
#save( subsample_dict, Path(interim_dir, "subsample_dict.pkl") )

#url_frequency_dict = load( Path(interim_dir, "subsample_dict.pkl") )

user_ids = []
url_hosts = []
url_sum_counts = []
for url in tqdm(url_frequency_dict.keys(), desc="Building sparse matrix content"):
    url_user_dict = url_frequency_dict[url]
    for user_id in url_user_dict.keys():
        url_user_count = url_user_dict[user_id]
        
        url_hosts.append( url )
        user_ids.append( user_id )
        url_sum_counts.append( url_user_count )

uniq_user_ids = np.unique( user_ids )
user_id_dict = {}
for i in tqdm(range( len(uniq_user_ids) ), desc="Building user --> id mapping"):
    user_id_dict[ uniq_user_ids[i] ] = i
inverted_user_id_dict = {v: k for k, v in user_id_dict.items()}
save( inverted_user_id_dict, Path(interim_dir, "factor_inverted_user_id_dict.pkl") )

uniq_urls = list(sorted(list(set(url_hosts))))
url_id_dict = {}
for i in tqdm(range( len(uniq_urls) ), desc="Building url --> id mapping"):
    url_id_dict[ uniq_urls[i] ] = i
inverted_url_id_dict = {v: k for k, v in url_id_dict.items()}
save( inverted_url_id_dict, Path(interim_dir, "factor_inverted_url_id_dict.pkl") )

for i in tqdm( range(len(url_sum_counts)), desc="Mapping content for building sparse matrix"):
    user_ids[i] = user_id_dict[ user_ids[i] ]
    url_hosts[i] = url_id_dict[ url_hosts[i] ]
user_ids = np.array( user_ids )
url_hosts = np.array( url_hosts )
url_sum_counts = np.array( url_sum_counts )

user_url_counts_matrix = coo_matrix( (url_sum_counts, (user_ids, url_hosts)), shape=(np.max(user_ids) + 1, np.max(url_hosts) + 1) )
save( user_url_counts_matrix, Path(interim_dir, "user_url_counts_matrix.pkl") )

user_url_counts_matrix = load( Path(interim_dir, "user_url_counts_matrix.pkl") )
als = implicit.approximate_als.FaissAlternatingLeastSquares(factors = 50,
                                                            iterations = 100,
                                                            nlist=400,
                                                            nprobe=20,
                                                            use_gpu = False,
                                                            calculate_training_loss = False, 
                                                            regularization = 0.1,
                                                            random_state=45)
als.fit(user_url_counts_matrix)
save(als, Path(interim_dir, "als_model.pkl"))

als = load(Path(interim_dir, "als_model.pkl"))
user_factors = als.model.user_factors
url_factors = als.model.item_factors

factor_dict_postfix = "50_100_400_20_1"

################################
user_factor_features_dict = {}
user_factor_features_dict["feature_names"] = []
for i in range(len(user_factors[0])):
    user_factor_features_dict["feature_names"].append( "user_factor_{}".format(i) )
inverted_user_id_dict = load( Path(interim_dir, "factor_inverted_user_id_dict.pkl") )
for i in tqdm(range( len(user_factors) ), desc="Building user factor features dict"):
    user_id = inverted_user_id_dict[i]
    user_factor_features = user_factors[i]
    user_factor_features_dict[ user_id ] = user_factor_features
save( user_factor_features_dict, Path(interim_dir, "user_factor_features_dict_{}.pkl".format(factor_dict_postfix)) )

#################################
url_factor_features_dict = {}
url_factor_features_dict["feature_names"] = []
for i in range(len(url_factors[0])):
    url_factor_features_dict["feature_names"].append( "url_factor_{}".format(i) )
inverted_url_id_dict = load( Path(interim_dir, "factor_inverted_url_id_dict.pkl") )
for i in tqdm(range( len(url_factors) ), desc="Building url factor features dict"):
    url_id = inverted_url_id_dict[i]
    url_factor_features = url_factors[i]
    url_factor_features_dict[ url_id ] = url_factor_features
save( url_factor_features_dict, Path(interim_dir, "url_only_factor_features_dict_{}.pkl".format(factor_dict_postfix)) )

user_feat_dict_builder = UserFeatDictBuilder()
df = pd.read_feather( Path(interim_dir, "stage_2.feather"), columns=["url_host", "user_id"] )
user_urls_factor_feat_dict = user_feat_dict_builder.build_feat_dict(df, url_factor_features_dict)
save( user_urls_factor_feat_dict, Path(interim_dir, "user_urls_factor_feat_dict_{}.pkl".format( factor_dict_postfix )) )

print("done")