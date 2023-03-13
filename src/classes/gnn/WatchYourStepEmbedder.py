import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
       tf.config.experimental.set_memory_growth(gpu,True)
from tensorflow.keras import optimizers, Model, layers, regularizers


from stellargraph.core import StellarGraph
from stellargraph.mapper import AdjacencyPowerGenerator
from stellargraph.layer import WatchYourStep
from stellargraph.losses import graph_log_likelihood
from stellargraph import datasets
from stellargraph.utils import plot_history

from matplotlib import pyplot as plt
from sklearn import preprocessing, feature_extraction, model_selection

import networkx as nx
import random
import numpy as np
import pandas as pd
import os

class WatchYourStepEmbedder():
    def __init__(self):
        
        self.w2v_model = None
        self.w2v_dict = None
        self.embedding_size = None
        
        self.node_id_dict = {}
                            
        pass
    
    # graph types: "url_url", "user_url", "common"
    def build_graph(self, url_user_frequency_dict):
        
        node_id = 0
        source_nodes = []
        target_nodes = []
        link_weights = []
        for url_host in tqdm( url_user_frequency_dict, desc="Building graph" ):
            if url_host not in self.node_id_dict.keys():
                self.node_id_dict[ url_host ] = node_id
                node_id += 1
            
            url_sum_count = 0
            for user_id in url_user_frequency_dict[url_host].keys():
                url_sum_count += url_user_frequency_dict[url_host][user_id]
            
            
            for user_id in url_user_frequency_dict[url_host].keys():
                if user_id not in self.node_id_dict.keys():
                    self.node_id_dict[ user_id ] = node_id
                    node_id += 1
                
                url_node_id = self.node_id_dict[ url_host ]
                user_node_id = self.node_id_dict[ user_id ]
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
            
        return graph
    
    def fit(self, graph, use_weights=False, rw_len=10, rw_count=10, 
            size=128, window=5, min_count=1, sample=0, iter=100, sg=0, 
            n_jobs=8, random_seed=45):
        
        new_node_id_dict = {}
        for key in self.node_id_dict.keys():
            new_node_id_dict[str(key)] = self.node_id_dict[key]
        self.node_id_dict = new_node_id_dict
        inverted_node_id_dict = { np.float32(v): k for k, v in self.node_id_dict.items() }
        
        """random_runner = BiasedRandomWalk( graph )
        random_walks = random_runner.run(
            nodes = graph.nodes(),  # root nodes
            length = rw_len,  # maximum length of a random walk
            n = rw_count,  # number of random walks per root node
            p = 0.5,  # Defines (unormalised) probability, 1/p, of returning to source node
            q = 2.0,  # Defines (unormalised) probability, 1/q, for moving away from source node
            weighted = use_weights,  # for weighted random walks
            seed = random_seed,  # random seed fixed for reproducibility
        )"""
        
        
        def make_random_walks_batch( nodes_batch, graph_copy ):
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
        for i in range(len(random_walk_batches)):
            for j in range(len(random_walk_batches[i])):
                random_walks.append( random_walk_batches[i][j] )
                random_walk_batches[i][j] = None
        #random_walks = np.hstack( random_walk_batches )
        
        for i in tqdm( range(len(random_walks)), desc="Inverting node ids to node names" ):
            for j in range(len(random_walks[i])):
                random_walks[i][j] = inverted_node_id_dict[ random_walks[i][j] ]

        np.random.seed( random_seed )
        np.random.shuffle( random_walks )
        
        logging.root.setLevel(level=logging.INFO)
        self.w2v_model = Word2Vec(random_walks, vector_size=size, window=window, workers=n_jobs, min_count=min_count, sample=sample, epochs=iter, sg=sg, seed=random_seed)
        self.w2v_dict = dict(zip(self.w2v_model.wv.index_to_key, self.w2v_model.wv.vectors))
        self.embedding_size = size
        
        return self
    
    def get_embedding(self, key):
        
        key = str(key)
        embedding = self.w2v_dict[ key ]
        
        return