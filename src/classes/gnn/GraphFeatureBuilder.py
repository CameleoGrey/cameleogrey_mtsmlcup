
import gc
import numpy as np

from tqdm import tqdm
from datetime import datetime

class GraphFeatureBuilder():
    def __init__(self):
        
        pass
    
    def build_user_graph_features_dict(self, df, graph_embedder, concat_user_id_vector=False ):
        
        user_url_feats = self.transform_( df, graph_embedder = graph_embedder, concat_user_id_vector = concat_user_id_vector )
        gc.collect()
        
        return user_url_feats
    
    def transform_(self, df, graph_embedder, concat_user_id_vector):
        
        user_ids = df["user_id"].values
        url_hosts = df["url_host"].values
        
        encoded_feats_names = []
        vec_size = graph_embedder.embedding_size
        if concat_user_id_vector:
            vec_size = 2 * vec_size
        for i in range(vec_size):
            encoded_feats_names.append( "graph_{}".format(i) )
        
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
        
        
        encoded_feats = self.vectorize_docs_( texts_for_transform, uniq_user_ids, graph_embedder, concat_user_id_vector, verbose=True )
        encoded_feats = np.array( encoded_feats )
        
        user_url_feats = {}
        user_url_feats["feature_names"] = encoded_feats_names
        for i in range(len(uniq_user_ids)):
            user_url_feats[uniq_user_ids[i]] = encoded_feats[i, :]
        
        return user_url_feats
    
    def vectorize_docs_(self, docs, uniq_user_ids, graph_embedder, concat_user_id_vector=False, verbose=True):

        doc_vectors = []
        if verbose:
            proc_range = tqdm(range(len(docs)), desc="Vectorizing docs")
        else:
            proc_range = range(len(docs))
            
        for i in proc_range:
            current_vector = []
            uniq_docs = set(docs[i])
            for current_url in uniq_docs:
                extracted_vector = graph_embedder.get_url_embedding( current_url )
                current_vector.append( extracted_vector )
            
            current_vector = np.mean( current_vector, axis=0 )
            
            if concat_user_id_vector:
                user_id = uniq_user_ids[i]
                user_vector = graph_embedder.get_user_embedding( user_id )
                current_vector = np.hstack( [current_vector, user_vector] )
                
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
    
    
    