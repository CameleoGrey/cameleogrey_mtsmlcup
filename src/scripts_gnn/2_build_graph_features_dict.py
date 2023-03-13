

import pandas as pd
from pathlib import Path

from classes.paths_config import *
from classes.utils import *
from classes.gnn.GraphFeatureBuilder import GraphFeatureBuilder

embedder_postfix = "w20_300_100_l40_c40"
graph_features_postfix = "node2vec_{}".format(embedder_postfix)

feature_builder = GraphFeatureBuilder()
graph_embedder = load( Path(interim_dir, "graph_embedder_{}.pkl".format( graph_features_postfix )) )
df = pd.read_feather( Path(interim_dir, "stage_2.feather"), columns=["url_host", "user_id"] )

user_graph_feat_dict = feature_builder.build_user_graph_features_dict(df, graph_embedder, concat_user_id_vector=False)
save(user_graph_feat_dict, Path(interim_dir, "user_graph_feat_dict_{}.pkl".format( graph_features_postfix )))

print("done")
