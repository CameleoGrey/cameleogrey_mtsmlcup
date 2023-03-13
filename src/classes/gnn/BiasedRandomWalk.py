
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
       tf.config.experimental.set_memory_growth(gpu,True)

import pandas as pd
import numpy as np
import warnings
import bisect
from collections import defaultdict, deque
from scipy import stats
from scipy.special import softmax
from tqdm import tqdm

from datetime import datetime

from stellargraph.core.validation import  comma_sep

from classes.gnn.RandomWalk import RandomWalk, _default_if_none

def naive_weighted_choices(rs, weights, size=None):
    """
    Select indices at random, weighted by the iterator `weights` of
    arbitrary (non-negative) floats. That is, `x` will be returned
    with probability `weights[x]/sum(weights)`.

    For doing a single sample with arbitrary weights, this is much (5x
    or more) faster than numpy.random.choice, because the latter
    requires a lot of preprocessing (normalized probabilties), and
    does a lot of conversions/checks/preprocessing internally.
    """
    probs = np.cumsum(weights)
    total = probs[-1]
    if total == 0:
        # all weights were zero (probably), so we shouldn't choose anything
        return None

    thresholds = rs.random() if size is None else rs.random(size)
    idx = np.searchsorted(probs, thresholds * total, side="left")
    
    #idx = bisect.bisect(probs, rs.random() * total)

    return idx

class BiasedRandomWalk(RandomWalk):
    """
    Performs biased second order random walks (like those used in Node2Vec algorithm
    https://snap.stanford.edu/node2vec/) controlled by the values of two parameters p and q.

    .. seealso::

       Examples using this random walk:

       - unsupervised representation learning: `Node2Vec using Gensim Word2Vec <https://stellargraph.readthedocs.io/en/stable/demos/embeddings/node2vec-embeddings.html>`__, `Node2Vec using StellarGraph <https://stellargraph.readthedocs.io/en/stable/demos/embeddings/keras-node2vec-embeddings.html>`__
       - node classification: `Node2Vec using Gensim Word2Vec <https://stellargraph.readthedocs.io/en/stable/demos/node-classification/node2vec-node-classification.html>`__, `Node2Vec using StellarGraph <https://stellargraph.readthedocs.io/en/stable/demos/node-classification/keras-node2vec-node-classification.html>`__, `Node2Vec with edge weights <https://stellargraph.readthedocs.io/en/stable/demos/node-classification/node2vec-weighted-node-classification.html>`__
       - link prediction: `Node2Vec <https://stellargraph.readthedocs.io/en/stable/demos/link-prediction/node2vec-link-prediction.html>`__, `comparison to CTDNE (TemporalRandomWalk) <https://stellargraph.readthedocs.io/en/stable/demos/link-prediction/ctdne-link-prediction.html>`__, `comparison of algorithms <https://stellargraph.readthedocs.io/en/stable/demos/link-prediction/homogeneous-comparison-link-prediction.html>`__

       Related functionality:

       - :class:`.UnsupervisedSampler` for transforming random walks into links for unsupervised training of link prediction models
       - :class:`.Node2Vec`, :class:`.Node2VecNodeGenerator` and :class:`.Node2VecLinkGenerator` for training a Node2Vec using only StellarGraph
       - Other random walks: :class:`.UniformRandomWalk`, :class:`.UniformRandomMetaPathWalk`, :class:`.TemporalRandomWalk`.

    Args:
        graph (StellarGraph): Graph to traverse
        n (int, optional): Total number of random walks per root node
        length (int, optional): Maximum length of each random walk
        p (float, optional): Defines probability, 1/p, of returning to source node
        q (float, optional): Defines probability, 1/q, for moving to a node away from the source node
        weighted (bool, optional): Indicates whether the walk is unweighted or weighted
        seed (int, optional): Random number generator seed

    """

    def __init__(
        self, graph, n=None, length=None, p=1.0, q=1.0, weighted=False, seed=None,
    ):
        super().__init__(graph, seed=seed)
        self.n = n
        self.length = length
        self.p = p
        self.q = q
        self.weighted = weighted
        self._checked_weights = False

        if weighted:
            self._check_weights_valid()

    def _check_weights_valid(self):
        if self._checked_weights:
            # we only need to check the weights once, either in the constructor or in run, whichever
            # sets `weighted=True` first
            return

        # Check that all edge weights are greater than or equal to 0.
        source, target, _, weights = self.graph.edge_arrays(
            include_edge_weight=True, use_ilocs=True
        )
        (invalid,) = np.where((weights < 0) | ~np.isfinite(weights))
        if len(invalid) > 0:

            def format(idx):
                s = source[idx]
                t = target[idx]
                w = weights[idx]
                return f"{s!r} to {t!r} (weight = {w})"

            raise ValueError(
                f"graph: expected all edge weights to be non-negative and finite, found some negative or infinite: {comma_sep(invalid, stringify=format)}"
            )

        self._checked_weights = True

    def run(
        self, nodes, *, n=None, length=None, p=None, q=None, seed=None, weighted=None
    ):

        """
        Perform a random walk starting from the root nodes. Optional parameters default to using the
        values passed in during construction.

        Args:
            nodes (list): The root nodes as a list of node IDs
            n (int, optional): Total number of random walks per root node
            length (int, optional): Maximum length of each random walk
            p (float, optional): Defines probability, 1/p, of returning to source node
            q (float, optional): Defines probability, 1/q, for moving to a node away from the source node
            seed (int, optional): Random number generator seed; default is None
            weighted (bool, optional): Indicates whether the walk is unweighted or weighted

        Returns:
            List of lists of nodes ids for each of the random walks

        """
        n = _default_if_none(n, self.n, "n")
        length = _default_if_none(length, self.length, "length")
        p = _default_if_none(p, self.p, "p")
        q = _default_if_none(q, self.q, "q")
        weighted = _default_if_none(weighted, self.weighted, "weighted")
        self._validate_walk_params(nodes, n, length)
        self._check_weights(p, q, weighted)
        rs, _ = self._get_random_state(seed)
        
        nodes = self.graph.node_ids_to_ilocs(nodes)

        if weighted:
            self._check_weights_valid()

        weight_dtype = self.graph._edges.weights.dtype
        cast_func = np.cast[weight_dtype]
        ip = cast_func(1.0 / p)
        iq = cast_func(1.0 / q)
        
        if np.isinf(ip):
            raise ValueError(
                f"p: value ({p}) is too small. It must be possible to represent 1/p in {weight_dtype}, but this value overflows to infinity."
            )
        if np.isinf(iq):
            raise ValueError(
                f"q: value ({q}) is too small. It must be possible to represent 1/q in {weight_dtype}, but this value overflows to infinity."
            )
        
        
        nodes_neighbors = {}
        nodes = list(reversed(nodes))
        for current_node in tqdm(nodes, desc="Building nodes neighbors dict"):
            current_neighbours, current_weights = self.graph.neighbor_arrays( current_node, 
                                                                              include_edge_weight=True, 
                                                                              use_ilocs=True )
            
            sorted_vals_ids = np.argsort( current_weights )
            current_neighbours = current_neighbours[ sorted_vals_ids ]
            current_weights = current_weights[ sorted_vals_ids ]
            nodes_neighbors[ current_node ] = (current_neighbours, current_weights)
        
        walks = []
        for node in tqdm(nodes, desc="Random walks"):  # iterate over root nodes
            for walk_number in range(n):  # generate n walks per root node
                # the walk starts at the root
                walk = [node]

                previous_node = None
                previous_node_neighbours = []
                current_node = node

                for _ in range(length - 1):
                    # select one of the neighbours using the
                    # appropriate transition probabilities
                    
                    if weighted:
                        neighbours = nodes_neighbors[ current_node ][0]
                        weights = nodes_neighbors[ current_node ][1]
                    else:
                        neighbours = self.graph.neighbor_arrays(
                            current_node, use_ilocs=True
                        )
                        weights = np.ones(neighbours.shape, dtype=weight_dtype)
                    if len(neighbours) == 0:
                        break
            
                    mask = neighbours == previous_node
                    weights[mask] *= ip
                    mask |= np.isin(neighbours, previous_node_neighbours)
                    weights[~mask] *= iq
                    
                    choice = naive_weighted_choices(rs, weights)
                    if choice is None:
                        break

                    previous_node = current_node
                    previous_node_neighbours = neighbours
                    current_node = neighbours[choice]

                    walk.append(current_node)
                
                walks.append(list(self.graph.node_ilocs_to_ids(walk)))

        return walks

    def _check_weights(self, p, q, weighted):
        """
        Checks that the parameter values are valid or raises ValueError exceptions with a message indicating the
        parameter (the first one encountered in the checks) with invalid value.

        Args:
            p: <float> The backward walk 'penalty' factor.
            q: <float> The forward walk 'penalty' factor.
            weighted: <False or True> Indicates whether the walk is unweighted or weighted.
       """
        if p <= 0.0:
            raise ValueError(f"p: expected positive numeric value, found {p}")

        if q <= 0.0:
            raise ValueError(f"q: expected positive numeric value, found {q}")

        if type(weighted) != bool:
            raise ValueError(f"weighted: expected boolean value, found {weighted}")