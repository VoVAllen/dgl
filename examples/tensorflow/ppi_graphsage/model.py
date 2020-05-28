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
from dgl.nn.tensorflow.conv import SAGEConv


class GraphSAGE(layers.Layer):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type):
        super(GraphSAGE, self).__init__()
        print("n_classes: {}".format(n_classes))
        self.layers = []
        self.g = g

        # input layer
        self.layers.append(SAGEConv(
            in_feats, n_hidden, aggregator_type, feat_drop=dropout, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(SAGEConv(
                n_hidden, n_hidden, aggregator_type, feat_drop=dropout, activation=activation))
        # output layer
        self.layers.append(SAGEConv(n_hidden, n_classes, aggregator_type,
                                    feat_drop=dropout, activation=None))  # activation None

    def call(self, features):
        h = features
        for layer in self.layers:
            h = layer(self.g, h)
        return h
