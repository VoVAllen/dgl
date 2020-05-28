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
from ppi_gat_model import AttentionGNN

SEED = 1
np.random.seed(SEED)
tf.random.set_seed(SEED)

IS_TRAINING = True


def evaluate(model, graph_tuples, labels):
    # IS_TRAINING = False
    # logits = model(graph_tuples).nodes

    model(graph_tuples)
    # output_graph = utils_tf.get_graph(outgraphs, 0)
    logits =model.outputs
    predict = np.where(logits.numpy() >= 0., 1, 0)
    score = f1_score(labels,
                     predict, average='micro')
    return score

def load_g(args, ds, i):
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

    ds = PPIDataset("train")

    graphs_tuple, n_classes, in_feats, labels, norm, n_edges = load_g(
        args, ds, 0)

    # num_nodes =
    model = AttentionGNN(n_classes, args.num_hidden, args.num_heads, args.num_layers)

    total_dur = []
    for epoch in range(args.n_epochs):
        dur = []
        scores = []
        loss_list = []
        for i in range(len(ds)):
            # total_time = 0
            graphs_tuple, n_classes, in_feats, labels, norm, n_edges = load_g(
                args, ds, i)

            graphs_tuple = utils_tf.set_zero_edge_features(graphs_tuple, 0)
            graphs_tuple = utils_tf.set_zero_global_features(graphs_tuple, 0)

            # Get the input signature for that function by obtaining the specs
            # input_signature = [
            #     utils_tf.specs_from_graphs_tuple(graphs_tuple)
            # ]

            # replace norm tensor
            # for layer in model.layers:
                # layer.set_norm(norm)

            # create GCN model

            optimizer = tf.keras.optimizers.Adam(
                learning_rate=args.lr)

            loss_fcn = tf.keras.losses.BinaryCrossentropy(
                from_logits=True)

            def update_step(graphs_tuple):
                with tf.GradientTape() as tape:
                    IS_TRAINING = True
                    model(graphs_tuple)
                    # output_graph = utils_tf.get_graph(outgraphs, 0)
                    logits =model.outputs
                    loss_value = loss_fcn(labels, logits)
                    # Manually Weight Decay
                    # We found Tensorflow has a different implementation on weight decay
                    # of Adam(W) optimizer with PyTorch. And this results in worse results.
                    # Manually adding weights to the loss to do weight decay solves this problem.
                    # for weight in model.trainable_variables:
                    #     loss_value = loss_value + \
                    #         args.weight_decay*tf.nn.l2_loss(weight)

                    grads = tape.gradient(
                        loss_value, model.trainable_variables)
                    optimizer.apply_gradients(
                        zip(grads, model.trainable_variables))
                return loss_value

            if epoch >= 3:
                t0 = time.time()

            with tf.device(device):
                loss_value = update_step(graphs_tuple)
            # forward
            if epoch >= 3:
                dur.append(time.time() - t0)

            f1_score = evaluate(model, graphs_tuple, labels)
            scores.append(f1_score)
            loss_list.append(loss_value.numpy().mean())
        # print(dur)
        if len(dur) != 0:
            total_dur.append(np.mean(dur))
            # print(np.sum(dur))
            # print(np.mean(total_dur))

        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
              "ETputs(KTEPS) {:.2f}". format(epoch, np.mean(total_dur), np.mean(loss_list),
                                             np.mean(scores), n_edges / np.mean(dur) / 1000))

        # acc = evaluate(model, graphs_tuple, labels)
        # print("Test Accuracy {:.4f}".format(acc))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GAT')
    parser.add_argument("--gpu", type=int, default=-1,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--n-epochs", type=int, default=400,
                        help="number of training epochs")
    parser.add_argument("--num-heads", type=int, default=4,
                        help="number of hidden attention heads")
    parser.add_argument("--num-out-heads", type=int, default=6,
                        help="number of output attention heads")
    parser.add_argument("--num-layers", type=int, default=2,
                        help="number of hidden layers")
    parser.add_argument("--num-hidden", type=int, default=256,
                        help="number of hidden units")
    parser.add_argument("--residual", action="store_true", default=True,
                        help="use residual connection")
    parser.add_argument("--in-drop", type=float, default=0,
                        help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=0,
                        help="attention dropout")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=0,
                        help="weight decay")
    parser.add_argument('--alpha', type=float, default=0.2,
                        help="the negative slop of leaky relu")
    parser.add_argument('--batch-size', type=int, default=2,
                        help="batch size used for training, validation and test")
    parser.add_argument('--patience', type=int, default=10,
                        help="used for early stop")
    args = parser.parse_args()
    print(args)

    main(args)
