
import pandas as pd
from pathlib import Path
from classes.GreyUrlEncoder import GreyUrlEncoder
from classes.paths_config import *
from classes.utils import *
from classes.UserFeatDictBuilder import UserFeatDictBuilder

df = pd.read_feather( Path(interim_dir, "stage_2.feather"), columns=["url_host", "user_id"] )

window = 20
vec_size = 300
epochs = 100
url_feat_dict_postfix = "_".join( [str(window), str(vec_size), str(epochs)] ) 
url_encoder = GreyUrlEncoder()
url_feat_dict = url_encoder.build_url_dict(df, shuffle_count=5, 
                                           vec_size=vec_size,
                                           window=window, n_jobs=8,
                                           min_count=1, sample=0,
                                           epochs=epochs, sg=0, seed=45)
save( url_feat_dict, Path(interim_dir, "w2v_url_feat_dict_{}.pkl".format( url_feat_dict_postfix )) )

url_feat_dict = load( Path(interim_dir, "w2v_url_feat_dict_{}.pkl".format( url_feat_dict_postfix )) )
user_feat_dict_builder = UserFeatDictBuilder()
user_urls_feat_dict = user_feat_dict_builder.build_feat_dict(df, url_feat_dict)
save( user_urls_feat_dict, Path(interim_dir, "user_urls_feat_dict_{}.pkl".format( url_feat_dict_postfix )) )

print("done")