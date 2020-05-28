from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from graph_nets import blocks
import time


from sklearn.metrics import f1_score
from graph_nets import utils_np
from graph_nets import utils_tf
from graph_nets import _base
# from graph_nets import
from graph_nets.demos_tf2 import models
# import matplotlib.pyplot as plt
import numpy as np
import argparse

import sonnet as snt
import tensorflow as tf

import dgl
from dgl.data import PPIDataset

import networkx as nx

from dgl import DGLGraph
from dgl.data import register_data_args, load_data

SEED = 1
np.random.seed(SEED)
tf.random.set_seed(SEED)

IS_TRAINING = True

# NUM_LAYERS = 2  # Hard-code number of layers in the edge/node/global models.
# LATENT_SIZE = 16  # Hard-code latent layer sizes for demos.


class Identity(_base.AbstractModule):
    def __init__(self, name="Identity"):
        super(Identity, self).__init__(name=name)

    def _build(self, inputs):
        return inputs


class PreApply(_base.AbstractModule):
    def __init__(self, norm_tensor, in_feats, out_feats, dropout, name='PreApply'):
        super(PreApply, self).__init__(name=name)
        self.norm_tensor = norm_tensor
        if dropout:
            self.dropout = snt.Dropout(dropout)
        else:
            self.dropout = 0.
        w_init = tf.keras.initializers.VarianceScaling(
            scale=1.0, mode="fan_out", distribution="uniform")
        self.weight = tf.Variable(initial_value=w_init(shape=(in_feats, out_feats),
                                                       dtype='float32'),
                                  trainable=True)

    def _build(self, h):
        if self.dropout:
            h = self.dropout(h, IS_TRAINING)
        h = tf.matmul(h, self.weight)
        norm_h = h * self.norm_tensor
        return norm_h


class AfterApply(_base.AbstractModule):
    def __init__(self, out_feats, activation, bias, name='AfterApply'):
        super(AfterApply, self).__init__(name=name)
        self.out_feats = out_feats
        self.activation = activation

    def _build(self, h):
        h = self.fc_self(h) + self.fc_neigh(h)
        if self.activation:
            h = self.activation(h)
        return h


class SAGELayer(_base.AbstractModule):
    """GraphNetwork with MLP edge, node, and global models."""

    def __init__(self, in_feats, out_feats, activation, dropout, norm_tensor, bias=True, name="GCNLayer"):
        super(GCNLayer, self).__init__(name=name)

        self.activation = activation
        # self.snt_pre_apply = PreApply(
        #     norm_tensor, in_feats, out_feats, dropout)
        # self.pre_apply = blocks.NodeBlock(lambda: self.snt_pre_apply, False, False, True, False, None, None)

        self.edge_block = blocks.EdgeBlock(
            edge_model_fn=lambda: Identity(),
            use_edges=False,
            use_receiver_nodes=False,
            use_sender_nodes=True,
            use_globals=False)

        # def node_model_fn():
        #     return AfterApply(out_feats, activation, bias)
        self._node_block = blocks.NodeBlock(
            node_model_fn=lambda: Identity(),
            use_received_edges=True,
            use_sent_edges=False,
            use_nodes=False,
            use_globals=False,
            received_edges_reducer=tf.math.unsorted_segment_mean)
        

        self.fc_neigh = layers.Dense(out_feats, use_bias=bias)
        self.fc_self = layers.Dense(out_feats, use_bias=bias)

    def _build(self, inputs):
        h_pre = inputs.nodes
        h = self._node_block(self.edge_block(inputs))

        h = self.fc_self(h_pre) + self.fc_neigh(h)

        if self.activation:
            h = self.activation(h)
        return h
    
    def set_norm(self, norm):
        self.snt_pre_apply.norm_tensor = norm


class GraphSAGE(_base.AbstractModule):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout, norm_tensor,
                 name="GCN"):
        super(GraphSAGE, self).__init__(name=name)

        self.layers = []

        # input layer
        self.layers.append(
            SAGELayer(in_feats, n_hidden, activation, dropout, norm_tensor))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(
                SAGELayer(n_hidden, n_hidden, activation, dropout, norm_tensor))
        # output layer
        self.layers.append(SAGELayer(n_hidden, n_classes,
                                    activation, dropout, norm_tensor))

    def _build(self, input):
        h = input
        for layer in self.layers:
            h = layer(h)
        return h
