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
        if bias:
            b_init = tf.zeros_initializer()
            self.bias = tf.Variable(initial_value=b_init(shape=(out_feats,),
                                                         dtype='float32'),
                                    trainable=True)
        else:
            self.bias = None

    def _build(self, h):
        if self.bias is not None:
            h = h + self.bias
        if self.activation:
            h = self.activation(h)
        return h


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


class GCN(_base.AbstractModule):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout, norm_tensor,
                 name="GCN"):
        super(GCN, self).__init__(name=name)

        self.layers = []

        # input layer
        self.layers.append(
            GCNLayer(in_feats, n_hidden, activation, dropout, norm_tensor))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(
                GCNLayer(n_hidden, n_hidden, activation, dropout, norm_tensor))
        # output layer
        self.layers.append(GCNLayer(n_hidden, n_classes,
                                    activation, dropout, norm_tensor))

    def _build(self, input):
        h = input
        for layer in self.layers:
            h = layer(h)
        return h


def evaluate(model, graph_tuples, labels):
    IS_TRAINING = False
    logits = model(graph_tuples).nodes

    predict = np.where(logits.numpy() >= 0., 1, 0)
    score = f1_score(labels,
                     predict, average='micro')
    return score


ds = PPIDataset("train")


def load_g(args, i):
    # ds = PPIDataset("train")
    g, labels = ds[i]
    n_classes = ds[0][1].shape[1]
    in_feats = 50
    features = g.ndata['feat']
    # g.remove_edges_from(nx.selfloop_edges(g))
    # g = DGLGraph(g)
    # # add self loop
    # g.add_edges(g.nodes(), g.nodes())
    n_edges = g.number_of_edges()

    degs = tf.cast(tf.identity(g.in_degrees()), dtype=tf.float32)
    norm = tf.math.pow(degs, -0.5)
    norm = tf.where(tf.math.is_inf(norm), tf.zeros_like(norm), norm)
    norm = tf.expand_dims(norm, -1)

    src, dst = g.edges()
    data_dict_1 = {
        "nodes": tf.cast(features, tf.float32),
        "senders": tf.cast(src, tf.int32),
        "receivers": tf.cast(dst, tf.int32)
    }
    graphs_tuple = utils_tf.data_dicts_to_graphs_tuple([data_dict_1])

    return graphs_tuple, n_classes, in_feats, labels, norm, n_edges


def main(args):
    # load and preprocess dataset
    # data = load_data(args)

    if args.gpu < 0:
        device = "/cpu:0"
    else:
        device = "/gpu:{}".format(args.gpu)

    graphs_tuple, n_classes, in_feats, labels, norm, n_edges = load_g(
        args, 0)

    model = GCN(in_feats,
                args.n_hidden,
                n_classes,
                args.n_layers,
                tf.nn.relu,
                args.dropout, norm)


    for epoch in range(args.n_epochs):


        for i in range(20):
            # total_time = 0
            graphs_tuple, n_classes, in_feats, labels, norm, n_edges = load_g(
                args, i)

            graphs_tuple = utils_tf.set_zero_edge_features(graphs_tuple, 0)
            graphs_tuple = utils_tf.set_zero_global_features(graphs_tuple, 0)

            # Get the input signature for that function by obtaining the specs
            input_signature = [
                utils_tf.specs_from_graphs_tuple(graphs_tuple)
            ]

            for layer in model.layers:
                layer.norm_tensor = norm
            # graphs_tuple = graphs_tuple.replace(edges=blocks.broadcast_sender_nodes_to_edges(graphs_tuple))

            # create GCN model

            optimizer = tf.keras.optimizers.Adam(
                learning_rate=args.lr)

            loss_fcn = tf.keras.losses.BinaryCrossentropy(
                from_logits=True)

            def update_step(graphs_tuple):
                with tf.GradientTape() as tape:
                    tape.watch(model.trainable_variables)
                    IS_TRAINING = True
                    outgraphs = model(graphs_tuple)
                    output_graph = utils_tf.get_graph(outgraphs, 0)
                    logits = output_graph.nodes
                    loss_value = loss_fcn(labels, logits)
                    # Manually Weight Decay
                    # We found Tensorflow has a different implementation on weight decay
                    # of Adam(W) optimizer with PyTorch. And this results in worse results.
                    # Manually adding weights to the loss to do weight decay solves this problem.
                    for weight in model.trainable_variables:
                        loss_value = loss_value + \
                            args.weight_decay*tf.nn.l2_loss(weight)

                    grads = tape.gradient(loss_value, model.trainable_variables)
                    optimizer.apply_gradients(
                        zip(grads, model.trainable_variables))
                return loss_value

            for i in range(20):
                graphs_tuple, n_classes, in_feats, labels, norm, n_edges = load_g(
                    args, i)

                model = GCN(in_feats,
                            args.n_hidden,
                            n_classes,
                            args.n_layers,
                            tf.nn.relu,
                            args.dropout, norm)

                graphs_tuple = utils_tf.set_zero_edge_features(graphs_tuple, 0)
                graphs_tuple = utils_tf.set_zero_global_features(
                    graphs_tuple, 0)

                # Get the input signature for that function by obtaining the specs
                input_signature = [
                    utils_tf.specs_from_graphs_tuple(graphs_tuple)
                ]

                compiled_update_step = tf.function(
                    update_step, input_signature=input_signature)

                if epoch >= 3:
                    t0 = time.time()

                with tf.device(device):
                    loss_value = update_step(graphs_tuple)
                # forward
                if epoch >= 3:
                    dur.append(time.time() - t0)
                acc = evaluate(model, graphs_tuple, labels)
                f1_scores.append(acc)
            total_dur.append(np.mean(dur) * 20)

            print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
                  "ETputs(KTEPS) {:.2f}". format(epoch, np.mean(total_dur), loss_value.numpy().item(),
                                                 np.mean(f1_scores), n_edges / np.mean(dur) / 1000))

        # acc = evaluate(model, graphs_tuple, labels)
        # print("Test Accuracy {:.4f}".format(acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    # register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=16,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    args = parser.parse_args()
    print(args)

    main(args)
