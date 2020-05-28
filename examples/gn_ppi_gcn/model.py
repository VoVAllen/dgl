import argparse
import time
import math
import numpy as np
import networkx as nx
import tensorflow as tf
from dgl import DGLGraph
import dgl.function as fn
from dgl.data import register_data_args, load_data
from tensorflow.keras import layers

from graph_nets import utils_np
from graph_nets import utils_tf
from graph_nets import _base

class GCNLayer(_base.AbstractModule):
    """GraphNetwork with MLP edge, node, and global models."""

    def __init__(self, in_feats, out_feats, activation, dropout, norm_tensor, bias=True, name="GCNLayer"):
        super(GCNLayer, self).__init__(name=name)

        self.activation = activation
        self.pre_apply = blocks.NodeBlock(lambda: PreApply(
            norm_tensor, in_feats, out_feats, dropout), False, False, True, False, None, None)

        self.edge_block = blocks.EdgeBlock(
            edge_model_fn=lambda: Identity(),
            use_edges=False,
            use_receiver_nodes=False,
            use_sender_nodes=True,
            use_globals=False)

        # def node_model_fn():
        #     return AfterApply(out_feats, activation, bias)
        self._node_block = blocks.NodeBlock(
            node_model_fn=lambda: AfterApply(out_feats, activation, bias),
            use_received_edges=True,
            use_sent_edges=False,
            use_nodes=False,
            use_globals=False)

    def _build(self, inputs):
        return self._node_block(self.edge_block(self.pre_apply(inputs)))




class GCN(layers.Layer):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.layers = []

        # input layer
        self.layers.append(
            GCNLayer(g, in_feats, n_hidden, activation, dropout))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(
                GCNLayer(g, n_hidden, n_hidden, activation, dropout))
        # output layer
        self.layers.append(GCNLayer(g, n_hidden, n_classes, None, dropout))

    def call(self, features):
        h = features
        for layer in self.layers:
            h = layer(h)
        return h