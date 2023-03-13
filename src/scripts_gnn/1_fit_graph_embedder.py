import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
       tf.config.experimental.set_memory_growth(gpu,True)

import gc
import pandas as pd
import pyarrow as pa
from pyarrow import feather
from pathlib import Path

from classes.paths_config import *
from classes.utils import *

from classes.gnn.Node2VecEmbedder import Node2VecEmbedder

if __name__ == "__main__":

    url_frequency_dict = load( Path(interim_dir, "url_frequency_dict.pkl") )
    
    #######
    # debug 
    """debug_frequency_dict = {}
    i = 0
    url_iter = iter(url_frequency_dict.keys())
    while i < 1000:
        current_url = next( url_iter )
        debug_frequency_dict[ current_url ] = url_frequency_dict[ current_url ]
        i += 1
    save( debug_frequency_dict, Path(interim_dir, "debug_frequency_dict.pkl") )"""
    #url_frequency_dict = load( Path(interim_dir, "debug_frequency_dict.pkl") )
    #######
    
    ######
    # Node2Vec
    graph_embedder = Node2VecEmbedder()
    embedder_postfix = "w20_300_100_l40_c40"
    
    graph, node_id_dict = graph_embedder.build_graph( url_frequency_dict )
    save( graph, Path( interim_dir, "graph.pkl" ) )
    save( node_id_dict, Path( interim_dir, "node_id_dict.pkl" ) )
    
    graph = load( Path( interim_dir, "graph.pkl" ) )
    node_id_dict = load( Path( interim_dir, "node_id_dict.pkl" ) )
    #print(graph.info())
    
    graph_embedder.fit(graph, node_id_dict, use_weights=False, rw_len=40, rw_count=40, 
                       size=300, window=20, min_count=1, sample=0, iter=100, sg=0, 
                       n_jobs=8, random_seed=45)
    save( graph_embedder, Path( interim_dir, "graph_embedder_node2vec_{}.pkl".format(embedder_postfix)) )
    ######
    
    print( "done" )