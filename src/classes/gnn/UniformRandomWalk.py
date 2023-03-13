
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
       tf.config.experimental.set_memory_growth(gpu,True)

import pandas as pd
import numpy as np
import warnings
from collections import defaultdict, deque
from scipy import stats
from scipy.special import softmax
from tqdm import tqdm

from datetime import datetime

from classes.gnn.RandomWalk import RandomWalk, _default_if_none

from joblib import Parallel, delayed

class UniformRandomWalk(RandomWalk):
    """
    Performs uniform random walks on the given graph

    .. seealso::

       Related functionality:

       - :class:`.UnsupervisedSampler` for transforming random walks into links for unsupervised training of link prediction models
       - Other random walks: :class:`.BiasedRandomWalk`, :class:`.UniformRandomMetaPathWalk`, :class:`.TemporalRandomWalk`.

    Args:
        graph (StellarGraph): Graph to traverse
        n (int, optional): Total number of random walks per root node
        length (int, optional): Maximum length of each random walk
        seed (int, optional): Random number generator seed

    """

    def __init__(self, graph, n=None, length=None, seed=None):
        super().__init__(graph, seed=seed)
        self.n = n
        self.length = length

    def run(self, nodes, *, n=None, length=None, seed=None):
        """
        Perform a random walk starting from the root nodes. Optional parameters default to using the
        values passed in during construction.

        Args:
            nodes (list): The root nodes as a list of node IDs
            n (int, optional): Total number of random walks per root node
            length (int, optional): Maximum length of each random walk
            seed (int, optional): Random number generator seed

        Returns:
            List of lists of nodes ids for each of the random walks

        """
        n = _default_if_none(n, self.n, "n")
        length = _default_if_none(length, self.length, "length")
        self._validate_walk_params(nodes, n, length)
        rs, _ = self._get_random_state(seed)

        nodes = self.graph.node_ids_to_ilocs(nodes)
        nodes_neighbors = {}
        for current_node in tqdm(nodes, desc="Building nodes neighbors dict"):
            current_neighbours = self.graph.neighbor_arrays(current_node, use_ilocs=True)
            nodes_neighbors[ current_node ] = current_neighbours
            
        # for each root node, do n walks
        random_walks = [self._walk(rs, node, length, nodes_neighbors) for node in tqdm(nodes, desc="Generating random walks") for _ in range(n)]
        
        #random_walks = Parallel( n_jobs=n_jobs, verbose=50 )( delayed(walk_multi_thread)(self.graph, rs, node, length) for node in nodes )
        
        return random_walks
        
        
    
    def _walk(self, rs, start_node, length, nodes_neighbors):
        walk = [start_node]
        current_node = start_node
        for _ in range(length - 1):
            #neighbours = self.graph.neighbor_arrays(current_node, use_ilocs=True)
            neighbors = nodes_neighbors[ current_node ]
            if len(neighbors) == 0:
                # dead end, so stop
                break
            else:
                # has neighbours, so pick one to walk to
                current_node = rs.choice(neighbors)
            walk.append(current_node)
        
        walk = list(self.graph.node_ilocs_to_ids(walk))

        return walk

    
    
    
    
    
    