from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import pairwise_distances
from sklearn import preprocessing

import numpy as np

from stellargraph.data import BiasedRandomWalk
from stellargraph import StellarGraph

from gensim.models import Word2Vec

import warnings
import collections
from stellargraph import datasets
import matplotlib.pyplot as plt

def jaccard_weights(graph, _subjects, edges):
    sources = graph.node_features(edges.source)
    targets = graph.node_features(edges.target)

    intersection = np.logical_and(sources, targets)
    union = np.logical_or(sources, targets)
    
    intersected_sum = intersection.sum(axis=1)
    union_sum = intersection.sum(axis=1)
    jw = intersected_sum / union_sum

    return jw

dataset = datasets.Cora()
G, subjects = dataset.load(
    largest_connected_component_only=True,
    edge_weights=jaccard_weights,
    str_node_ids=True,  # Word2Vec requires strings, not ints
)
print(G.info())

_, weights = G.edges(include_edge_weight=True)

wt, cnt = np.unique(weights, return_counts=True)

rw = BiasedRandomWalk(G)
weighted_walks = rw.run(
    nodes=G.nodes(),  # root nodes
    length=10,  # maximum length of a random walk
    n=10,  # number of random walks per root node
    p=0.5,  # Defines (unormalised) probability, 1/p, of returning to source node
    q=2.0,  # Defines (unormalised) probability, 1/q, for moving away from source node
    weighted=True,  # for weighted random walks
    seed=45,  # random seed fixed for reproducibility
)
print("Number of random walks: {}".format(len(weighted_walks)))


weighted_model = Word2Vec(
    weighted_walks, vector_size=128, window=5, min_count=0, sg=1, workers=1, iter=1
)

emb = weighted_model.wv["19231"]
emb.shape

node_ids = weighted_model.wv.index2word  # list of node IDs
weighted_node_embeddings_2d = (
    weighted_model.wv.vectors
)  # numpy.ndarray of size number of nodes times embeddings dimensionality
# the gensim ordering may not match the StellarGraph one, so rearrange
node_targets = subjects.loc[node_ids].astype("category")

alpha = 0.7

plt.figure(figsize=(10, 8))
plt.scatter(
    weighted_node_embeddings_2d[:, 0],
    weighted_node_embeddings_2d[:, 1],
    c=node_targets.cat.codes,
    cmap="jet",
    alpha=0.7,
)
plt.show()




print("done")