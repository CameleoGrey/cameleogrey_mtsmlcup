
from tqdm import tqdm
import numpy as np
import pandas as pd
import gc
from gensim.models import Word2Vec
from copy import deepcopy
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')

def identity_tokenizer(text):
    return text

class GreyCategoricalEncoder():
    
    def __init__(self,
                 shuffle_count=5, vec_size=1,
                 window=5, n_jobs=8,
                 min_count=1, sample=0,
                 epochs=100, sg=0, seed=45):

        self.feature_name = None
        self.w2v_dict = None
        self.shuffle_count = shuffle_count
        self.vec_size = vec_size
        self.window = window
        self.n_jobs = n_jobs
        self.min_count = min_count
        self.sample = sample
        self.epochs = epochs
        self.sg = sg
        self.seed = seed

        pass

    def fit(self, values_list, feature_name):
        
        self.feature_name = feature_name
        
        shuffled_token_rows = []
        for i in tqdm(range(self.shuffle_count), desc="Generating token shuffles for fitting cat encoder"):
            shuffled_tokens = deepcopy(values_list)
            np.random.shuffle(shuffled_tokens)
            shuffled_token_rows.append( shuffled_tokens )
        
        w2v_dict = self.fit_word2vec_( shuffled_token_rows, size=self.vec_size,
                                      window=self.window, n_jobs=self.n_jobs,
                                      min_count=self.min_count, sample=self.sample,
                                      epochs=self.epochs, sg=self.sg, seed=self.seed)
        self.w2v_dict = w2v_dict

        return self
    

    def transform(self, values_list):
        
        encoded_feats_names = []
        for i in range(self.vec_size):
            encoded_feats_names.append( str(self.feature_name) + "_{}".format(i) )
        
        encoded_feats = self.encode_features_( values_list, verbose=True )
        encoded_feats = np.array( encoded_feats )

        return encoded_feats, encoded_feats_names

    def fit_word2vec_(self, texts, size=128, window=5, n_jobs=8, min_count=1, sample=0, epochs=100, sg=0, seed=45):
        logging.root.setLevel(level=logging.INFO)
        w2v_model = Word2Vec(texts, vector_size=size, window=window, workers=n_jobs, min_count=min_count, sample=sample, epochs=epochs, sg=sg, seed=seed)
        w2v_dict = dict(zip(w2v_model.wv.index_to_key, w2v_model.wv.vectors))
        del texts
        gc.collect()

        return w2v_dict

    def encode_features_(self, texts, verbose=True):

        text_vectors = []
        if verbose:
            proc_range = tqdm(range(len(texts)), desc="Vectorizing texts")
        else:
            proc_range = range(len(texts))

        vec_size = len(self.w2v_dict[next(iter(self.w2v_dict.keys()))])
        for i in proc_range:
            current_vector = None
            current_vector = self.w2v_dict[texts[i]]
            if vec_size == 1:
                current_vector = current_vector[0]

            text_vectors.append(current_vector)
        return text_vectors