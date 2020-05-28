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


class GCNLayer(layers.Layer):
    def __init__(self,
                 g,
                 in_feats,
                 out_feats,
                 activation,
                 dropout,
                 bias=True):
        super(GCNLayer, self).__init__()
        self.g = g

        w_init = tf.keras.initializers.VarianceScaling(
            scale=1.0, mode="fan_out", distribution="uniform")
        self.weight = tf.Variable(initial_value=w_init(shape=(in_feats, out_feats),
                                                       dtype='float32'),
                                  trainable=True)
        if dropout:
            self.dropout = layers.Dropout(rate=dropout)
        else:
            self.dropout = 0.
        if bias:
            b_init = tf.zeros_initializer()
            self.bias = tf.Variable(initial_value=b_init(shape=(out_feats,),
                                                         dtype='float32'),
                                    trainable=True)
        else:
            self.bias = None
        self.activation = activation

    def call(self, h):
        if self.dropout:
            h = self.dropout(h)
        self.g.ndata['h'] = tf.matmul(h, self.weight)
        self.g.ndata['norm_h'] = self.g.ndata['h'] * self.g.ndata['norm']
        self.g.update_all(fn.copy_src('norm_h', 'm'),
                          fn.sum('m', 'h'))
        h = self.g.ndata['h']
        if self.bias is not None:
            h = h + self.bias
        if self.activation:
            h = self.activation(h)
        return h


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