
import pandas as pd
from classes.utils import *
from classes.paths_config import *

from classes.scrapping.TfidfW2vVectorizer import TfidfW2vVectorizer
from classes.scrapping.BERTVectorizer import BERTVectorizer
from classes.UserFeatDictBuilder import UserFeatDictBuilder

url_content_dict = load( Path(interim_dir, "url_content_dict.pkl") )

##############
#debug
"""debug_dict = {}
keys = list(url_content_dict.keys())
for i in range(300):
    current_url = keys[i]
    debug_dict[current_url] = url_content_dict[current_url]
save( debug_dict, Path( interim_dir, "debug_dict.pkl" ))"""
#url_content_dict = load( Path( interim_dir, "debug_dict.pkl" ) )
##############

url_hosts = []
url_contents = []
for url in url_content_dict.keys():
    current_url_content = url_content_dict[ url ]
    url_hosts.append( url )
    url_contents.append( current_url_content )

#content_encoder_postfix = "w20_300_100"
content_encoder_postfix = "bert"

"""content_encoder = TfidfW2vVectorizer()
content_encoder.fit(url_contents, vector_size=300, window=20,
                    n_jobs=10, min_count=1, sample=0.0, epochs=300, sg=0, seed=45)
save( content_encoder, Path(interim_dir, "url_content_encoder_w20_300_300.pkl") )"""

#content_encoder = load( Path(interim_dir, "url_content_encoder_w20_300_300.pkl") )
#content_vectors = content_encoder.vectorize_docs(url_contents, use_tfidf=True, n_jobs=14)

content_encoder = BERTVectorizer()
content_vectors = content_encoder.vectorize_docs( url_contents )

url_content_vectors_dict = {}
for i in range(len(content_vectors)):
    url_content_vectors_dict[ url_hosts[i] ] = content_vectors[i]
url_content_vectors_dict["feature_names"] = []
for i in range(len(content_vectors[0])):
    url_content_vectors_dict["feature_names"].append( "content_{}".format(i) )
save( url_content_vectors_dict, Path(interim_dir, "url_content_vectors_dict_{}.pkl".format( content_encoder_postfix )) )

user_feat_dict_builder = UserFeatDictBuilder()
url_content_vectors_dict = load( Path(interim_dir, "url_content_vectors_dict_{}.pkl".format( content_encoder_postfix )) )
df = pd.read_feather( Path(interim_dir, "stage_2.feather"), columns=["url_host", "user_id"] )
user_url_content_vectors_dict = user_feat_dict_builder.build_feat_dict(df, url_content_vectors_dict)
save( user_url_content_vectors_dict, Path(interim_dir, "user_url_content_vectors_dict_{}.pkl".format( content_encoder_postfix )) )


print("done")