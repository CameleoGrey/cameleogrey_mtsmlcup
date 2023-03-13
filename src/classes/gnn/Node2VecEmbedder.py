
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
       tf.config.experimental.set_memory_growth(gpu,True)

import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')

from joblib import Parallel, delayed
from copy import deepcopy

from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import pairwise_distances
from sklearn import preprocessing

from gensim.models import Word2Vec

import warnings
import collections
import matplotlib.pyplot as plt


from stellargraph import StellarGraph
from classes.gnn.UniformRandomWalk import UniformRandomWalk
from classes.gnn.BiasedRandomWalk import BiasedRandomWalk

class Node2VecEmbedder():
    def __init__(self):
        
        self.w2v_dict = None
        self.embedding_size = None
                            
        pass
    
    # graph types: "url_url", "user_url", "common"
    def build_graph(self, url_user_frequency_dict):
        
        source_nodes = []
        target_nodes = []
        link_weights = []
        
        # build special node_name --> node_id dict
        url_hosts = list(url_user_frequency_dict.keys())
        for i in range(len(url_hosts)):
            url_hosts[i] = self.add_url_host_postfix( url_hosts[i] )
        user_ids = set()
        for url_host in tqdm( url_user_frequency_dict, desc="Building graph" ):
            for user_id in url_user_frequency_dict[url_host].keys():
                user_id = self.add_user_postfix( user_id )
                user_ids.add( user_id )
        user_ids = list( user_ids )
        raw_node_ids = url_hosts + user_ids
        node_id_dict = {}
        for i in range(len(raw_node_ids)):
            node_id_dict[ raw_node_ids[i] ] = np.float64(i)

        
        for url_host in tqdm( url_user_frequency_dict, desc="Building graph" ):
    
            url_sum_count = 0.0
            for user_id in url_user_frequency_dict[url_host].keys():
                url_sum_count += url_user_frequency_dict[url_host][user_id]
            
            for user_id in url_user_frequency_dict[url_host].keys():
                url_node_id = node_id_dict[ self.add_url_host_postfix( url_host ) ]
                user_node_id = node_id_dict[ self.add_user_postfix( user_id ) ]
                link_weight = url_user_frequency_dict[url_host][user_id] / url_sum_count
                source_nodes.append( url_node_id )
                target_nodes.append( user_node_id )
                link_weights.append( link_weight )
                
        source_nodes = np.array( source_nodes ).reshape((-1, 1))
        target_nodes = np.array( target_nodes ).reshape((-1, 1))
        link_weights = np.array( link_weights ).reshape((-1, 1))
        #ids = np.array([i for i in range(len(source_nodes))])
        #source_nodes = IndexedArray( source_nodes, index=ids)
        #target_nodes = IndexedArray( target_nodes, index=ids)
        
        nodes = np.hstack( [source_nodes, target_nodes, link_weights] )
        nodes = pd.DataFrame( nodes, columns=["source", "target", "weight"] )
                
        #graph = StellarGraph( edges={"source": source_nodes, "target": target_nodes} )
        graph = StellarGraph( edges=nodes )
            
        return graph, node_id_dict
    
    def fit(self, graph, node_id_dict, use_weights=False, rw_len=10, rw_count=20, 
            size=300, window=5, min_count=1, sample=0, iter=100, sg=0, 
            n_jobs=8, random_seed=45):
        
        inverted_node_id_dict = { v: k for k, v in node_id_dict.items() }
        
        """def make_random_walks_batch( nodes_batch, graph_copy ):
            random_runner = UniformRandomWalk( graph_copy )
            random_walks_batch = random_runner.run(
                nodes = nodes_batch,  # root nodes
                length = rw_len,  # maximum length of a random walk
                n = rw_count,  # number of random walks per root node
            )
            return  random_walks_batch
        
        nodes = graph.nodes()
        nodes_batches = np.array_split( nodes, n_jobs )
        graph_copies = [ deepcopy(graph) for i in range(n_jobs) ]
        random_walk_batches = Parallel( n_jobs )( delayed( make_random_walks_batch )( nb, gc ) for nb, gc in zip( nodes_batches, graph_copies ) )
        random_walks = []
        for i in tqdm(range(len(random_walk_batches))):
            for j in range(len(random_walk_batches[i])):
                random_walks.append( random_walk_batches[i][j] )
                random_walk_batches[i][j] = None"""
        #random_walks = np.hstack( random_walk_batches )
        
        nodes = graph.nodes()
        
        random_runner = UniformRandomWalk( graph )
        random_walks = random_runner.run( nodes = nodes, length = rw_len, n = rw_count, seed = random_seed )
        
        #random_runner = BiasedRandomWalk( graph )
        #random_walks = random_runner.run( nodes = nodes, length = rw_len, n = rw_count, 
        #                                  seed = random_seed, p=0.5, q=2.0, weighted=True )
        
        for i in tqdm( range(len(random_walks)), desc="Inverting node ids to node names" ):
            for j in range(len(random_walks[i])):
                random_walks[i][j] = inverted_node_id_dict[ random_walks[i][j] ]

        np.random.seed( random_seed )
        np.random.shuffle( random_walks )
        
        logging.root.setLevel(level=logging.INFO)
        w2v_model = Word2Vec(random_walks, vector_size=size, window=window, workers=n_jobs, 
                                  min_count=min_count, sample=sample, epochs=iter, sg=sg, seed=random_seed)
        self.w2v_dict = dict(zip(w2v_model.wv.index_to_key, w2v_model.wv.vectors))
        self.embedding_size = size
        
        return self
    
    def add_user_postfix(self, user_id):
        url_host = str(user_id) + "_" + "user_id"
        return url_host
    
    def add_url_host_postfix(self, url_host):
        user_id = str(url_host) + "_" + "url_host"
        return user_id
    
    def get_url_embedding(self, url_host):
        url_host = self.add_url_host_postfix( url_host )
        url_embedding = self.w2v_dict[ url_host ]
        return url_embedding
    
    def get_user_embedding(self, user_id):
        user_id = self.add_user_postfix( user_id )
        user_id_embedding = self.w2v_dict[ user_id ]
        return user_id_embedding
    
    
    
    
    
    
    
    