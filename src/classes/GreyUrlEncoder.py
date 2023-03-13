
from tqdm import tqdm
import numpy as np
import random
import gc
from gensim.models import Word2Vec
from copy import deepcopy
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')

def identity_tokenizer(text):
    return text

class GreyUrlEncoder():
    
    def __init__(self):

        pass

    def build_url_dict(self, url_user_id_df, 
                       shuffle_count=5, vec_size=50,
                       window=5, n_jobs=8,
                       min_count=1, sample=0,
                       epochs=100, sg=0, seed=45):
        
        user_ids = url_user_id_df["user_id"].values
        url_hosts = url_user_id_df["url_host"].values
        
        texts_for_fit = []
        uniq_user_ids = np.unique( user_ids )
        user_ids_backward_index = self.build_backward_index_( user_ids )
        for uniq_id in tqdm( uniq_user_ids, desc="Aggregating urls by user_id" ):
            id_group_mask = user_ids_backward_index[ uniq_id ]
            group_urls = url_hosts[ id_group_mask ]
            group_tokens = " ".join( group_urls )
            group_tokens = group_tokens.split(" ")
            texts_for_fit.append( group_tokens )
            
        shuffled_token_rows = []
        for i in tqdm(range(shuffle_count), desc="Building shuffles for fitting urls encoder"):
            shuffled_tokens = deepcopy(texts_for_fit)
            for j in range(len(shuffled_tokens)):
                np.random.shuffle(shuffled_tokens[j])
                shuffled_token_rows.append( shuffled_tokens[j] )
        random.shuffle( shuffled_token_rows )

        url_feat_dict = self.build_w2_dict_( shuffled_token_rows, size=vec_size,
                                                  window=window, n_jobs=n_jobs,
                                                  min_count=min_count, sample=sample,
                                                  epochs=epochs, sg=sg, seed=seed )
        
        encoded_feats_names = []
        url_key = next(iter(url_feat_dict))
        vec_size = len(url_feat_dict[url_key])
        for i in range(vec_size):
            encoded_feats_names.append( "w2v_url_host_{}".format(i) )
        url_feat_dict["feature_names"] = encoded_feats_names

        return url_feat_dict

    def build_w2_dict_(self, docs, size=128, window=5, n_jobs=8, min_count=1, sample=0, epochs=100, sg=0, seed=45):
        logging.root.setLevel(level=logging.INFO)
        w2v_model = Word2Vec(docs, vector_size=size, window=window, workers=n_jobs, min_count=min_count, sample=sample, epochs=epochs, sg=sg, seed=seed)
        w2v_dict = dict(zip(w2v_model.wv.index_to_key, w2v_model.wv.vectors))
        del docs
        gc.collect()

        return w2v_dict
    
    def build_backward_index_(self, x_array):
            
        backward_index = {}
        for i in tqdm(range(len(x_array)), desc="Building backward index"):
            current_x = x_array[i]

            if current_x not in backward_index.keys():
                backward_index[current_x] = []
            backward_index[current_x].append(i)
            
        for x in tqdm(backward_index.keys(), desc="Building backward index (final types converting)"):
            backward_index[x] = np.array( backward_index[x] )
            
        return backward_index