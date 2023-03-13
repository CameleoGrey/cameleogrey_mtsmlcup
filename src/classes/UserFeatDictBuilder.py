
import gc
import numpy as np
from tqdm import tqdm
import logging

class UserFeatDictBuilder():
    def __init__(self):
        pass
    
    def build_feat_dict(self, url_user_id_df, url_feat_dict):
        
        user_ids = url_user_id_df["user_id"].values
        url_hosts = url_user_id_df["url_host"].values
        
        texts_for_transform = []
        uniq_user_ids = np.unique( user_ids )
        user_ids_backward_index = self.build_backward_index_( user_ids )
        for i in tqdm( range(len(uniq_user_ids)), desc="Aggregating urls by user_id" ):
            uniq_id = uniq_user_ids[i]
            id_group_mask = user_ids_backward_index[ uniq_id ]
            group_urls = url_hosts[ id_group_mask ]
            group_tokens = " ".join( group_urls )
            group_tokens = group_tokens.split(" ")
            texts_for_transform.append( group_tokens )
        
        encoded_feats = self.encode_user_urls_( texts_for_transform, url_feat_dict, verbose=True )
        encoded_feats = np.array( encoded_feats )
        
        user_url_feats = {}
        user_url_feats["feature_names"] = url_feat_dict["feature_names"]
        
        for i in range(len(uniq_user_ids)):
            user_url_feats[uniq_user_ids[i]] = encoded_feats[i, :]
        
        return user_url_feats

    def encode_user_urls_(self, docs, url_feat_dict=None, verbose=True):

        doc_vectors = []
        if verbose:
            proc_range = tqdm(range(len(docs)), desc="Encoding unique user's urls")
        else:
            proc_range = range(len(docs))

        for i in proc_range:
            current_vector = []
            uniq_docs = set(docs[i])
            for current_url in uniq_docs:
                extracted_vector = url_feat_dict[current_url]
                
                current_vector.append( extracted_vector )
            
            current_vector = np.mean( current_vector, axis=0 )
            doc_vectors.append(current_vector)
        return doc_vectors
    
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