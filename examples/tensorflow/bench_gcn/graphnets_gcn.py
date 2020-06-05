import argparse
import numpy as np
import random
import time
import tensorflow as tf
from pyinstrument import Profiler
from tensorflow.keras import layers
import dgl
import dgl.function as fn
from graph_nets import blocks

from graph_nets import utils_tf
from graph_nets import _base
from scipy import sparse as spsp

# from tensorflow.python.context import context
from tensorflow.python.eager import context

def load_random_graph(args):
    n_nodes = args.n_nodes
    n_edges = n_nodes * 1

    # row = np.random.RandomState(6657).choice(n_nodes, n_edges)
    # col = np.random.RandomState(6657).choice(n_nodes, n_edges)
    row = np.arange(n_nodes)
    col = np.arange(n_nodes)
    spm = spsp.coo_matrix((np.ones(len(row)), (row, col)), shape=(n_nodes, n_nodes))
    g = dgl.graph(spm)

    # load and preprocess dataset
    features = tf.ones((n_nodes, args.n_feats))
    labels = tf.constant(np.random.choice(args.n_classes, n_nodes),dtype=tf.int64)
    train_mask = np.ones(shape=(n_nodes))
    train_mask = tf.constant(train_mask, dtype=tf.bool)
    print("""----Data statistics------'
      #Edges %d
      #Train samples %d""" % (n_edges, train_mask.numpy().sum().item()))
    

    src, dst = g.edges()
    data_dict_1 = {
        "nodes": features,
        'n_node': n_nodes,
        'n_edge': n_edges,
        "senders": tf.cast(src, tf.int32),
        "receivers": tf.cast(dst, tf.int32)
    }
    graphs_tuple = utils_tf.data_dicts_to_graphs_tuple([data_dict_1])
    
    return graphs_tuple, features, labels, train_mask

class GraphConv(_base.AbstractModule):
    def __init__(self,
                 in_feats,
                 out_feats):
        super(GraphConv, self).__init__()

        self._in_feats = in_feats
        self._out_feats = out_feats        
        w_init = tf.keras.initializers.VarianceScaling(
            scale=1.0, mode="fan_out", distribution="uniform")
        self.weight = tf.Variable(initial_value=w_init(shape=(in_feats, out_feats),
                                                       dtype='float32'),
                                  trainable=True)

        b_init = tf.zeros_initializer()
        self.bias = tf.Variable(initial_value=b_init(shape=(out_feats,),
                                                        dtype='float32'),
                                trainable=True)

        self.received_edges_aggregator = blocks.ReceivedEdgesToNodesAggregator(
                reducer=tf.math.unsorted_segment_sum)

    def _build(self, graph, feat):
        if self._in_feats > self._out_feats:
            # mult W first to reduce the feature size for aggregation.
            feat = tf.matmul(feat, self.weight)
            
            graph = graph.replace(nodes=feat)
            sender_values = blocks.broadcast_sender_nodes_to_edges(
                graph)
            # Summing all of the attended values from each node.
            # [total_num_nodes, num_heads, embedding_size]
            aggregated_attended_values = self.received_edges_aggregator(graph.replace(edges=sender_values))
            # print(tf.reduce_sum(aggregated_attended_values - feat))
            # aggregated_attended_values=feat
            graph = graph.replace(nodes=aggregated_attended_values)
            rst = graph.nodes

        else:
            # aggregated_attended_values = feat
            # aggregate first then mult W
            graph = graph.replace(nodes=feat)
            sender_values = blocks.broadcast_sender_nodes_to_edges(
                graph)
            # Summing all of the attended values from each node.
            # [total_num_nodes, num_heads, embedding_size]
            aggregated_attended_values = self.received_edges_aggregator(graph.replace(edges=sender_values))
            
            # print(tf.reduce_sum(aggregated_attended_values - feat))
            graph = graph.replace(nodes=aggregated_attended_values)
            rst = graph.nodes
            rst = tf.matmul(rst, self.weight)

        rst = rst + self.bias
        return rst

class GCN(_base.AbstractModule):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.g = g
        self.layers = []
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes))
        self.activation = activation
        self.dropout = layers.Dropout(dropout)

    def _build(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            h = layer(self.g, h)
            if i != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h, training=True)
        return h

def main(args):
    if args.gpu >= 0:
        device = f'gpu:{args.gpu}'
    else:
        device = 'cpu'
    
    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # if args.gpu >= 0 and torch.cuda.is_available():
    #     torch.cuda.manual_seed(args.seed)

    g, features, labels, train_mask = load_random_graph(args)
    # g.nodes = features
    with tf.device(device):

        # create GCN model
        model = GCN(g=g,
                    in_feats=args.n_feats,
                    n_hidden=args.n_hidden,
                    n_classes=args.n_classes,
                    n_layers=args.n_layers,
                    activation=tf.nn.relu,
                    dropout=args.dropout)

        loss_fcn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True)
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=args.lr)

        # initialize graph
        dur = []
        profiler = Profiler()
        # profiler.start()
        for epoch in range(args.n_epochs):
            with tf.GradientTape() as tape:                
                context.async_wait()
                if epoch >= 3:
                    t0 = time.time()
                # forward
                logits = model(features)

                loss = loss_fcn(labels[train_mask], logits[train_mask])

                for weight in model.trainable_variables:
                    loss = loss + \
                        args.weight_decay*tf.nn.l2_loss(weight)
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                # Manual Sync
                context.async_wait()
                if epoch >= 3:
                    dur.append(time.time() - t0)
                    print('Training time: {:.4f}'.format(np.mean(dur)))
        # profiler.stop()
        # print(profiler.output_text(unicode=True, color=True))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument("--seed", type=int, default=0,
                        help='Random seed')
    parser.add_argument("--n-nodes", type=int, default=1000000,
                        help="Number of nodes in the random graph")
    parser.add_argument("--n-feats", type=int, default=100,
                        help="Number of input node features")
    parser.add_argument("--n-classes", type=int, default=10,
                        help="Number of node classes")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=0,
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