import graph_nets as gn
import sonnet as snt
import tensorflow as tf
import numpy as np

from graph_nets import _base


class Head(_base.AbstractModule):
    def __init__(self, num_heads, hidden, name="head"):
        super(Head, self).__init__(name=name)
        self.num_heads = num_heads
        self.hidden = hidden
        
        self.linear_val = snt.Linear(
            self.hidden, name="linear_val", with_bias=False)
        self.linear_key = snt.Linear(
            self.hidden, name="linear_key", with_bias=False)
        self.linear_query = snt.Linear(
            self.hidden, name="linear_query", with_bias=False)

    def _build(self, inputs):

        # inputs - [nodes, emb_dim]
        # tf.split(...) - [nodes, num_heads, emb_dim/num_heads]
        vals = tf.transpose(tf.split(self.linear_val(inputs),
                                     self.num_heads, axis=1), [1, 0, 2])
        keys = tf.transpose(tf.split(self.linear_key(inputs),
                                     self.num_heads, axis=1), [1, 0, 2])
        queries = tf.transpose(tf.split(self.linear_query(
            inputs), self.num_heads, axis=1), [1, 0, 2])
        queries /= np.sqrt(self.hidden / self.num_heads)
        return vals, keys, queries


class GraphAttention(_base.AbstractModule):
    def __init__(self, hidden, num_heads, name="graph_attention"):
        super(GraphAttention, self).__init__(name=name)
        self.hidden = hidden
        self.num_heads = num_heads
        
        self.head = Head(self.num_heads, self.hidden)
        self.attention = gn.modules.SelfAttention()
        self.output_proj = snt.Linear(self.hidden, with_bias=False)
        self.mlp = snt.nets.MLP([self.hidden, self.hidden])

    def _build(self, inputs, input_graph):
        # norm1 = snt.BatchNorm()
        # norm2 = snt.BatchNorm()

        vals, keys, queries = self.head(inputs)
        attended_nodes = self.attention(vals, keys, queries, input_graph).nodes
        output_projected = self.output_proj(tf.reduce_sum(attended_nodes, axis=1))

        # This skip-connection is non-orthodox
        output_projected = output_projected + \
            tf.reshape(vals, [-1, self.hidden])

        normalized = self.mlp(output_projected) + output_projected
        return normalized


class AttentionGNN(_base.AbstractModule):
    def __init__(self, num_classes, hidden, num_heads, num_iters, name="attention_gnn"):
        super(AttentionGNN, self).__init__(name=name)
        self.num_classes = num_classes
        # self.num_nodes = num_nodes
        # self.emb_size = emb_size
        self.hidden = hidden
        self.num_heads = num_heads
        self.num_iters = num_iters
        # self.ids = tf.constant(list(range(num_nodes)), dtype=tf.int32)
        self.linear = snt.Linear(self.hidden)
        self.mlp = snt.nets.MLP([self.hidden, self.hidden], activate_final=False)
        self.gnn_attentions = []
        self.final = snt.Linear(self.num_classes)

    def _build(self, input_graph):
        # embed = snt.Embed(self.num_nodes, self.emb_size)
        # norm = snt.BatchNorm()
        for _ in range(self.num_iters):
            gnn_attention = GraphAttention(self.hidden, self.num_heads)
            self.gnn_attentions.append(gnn_attention)
        

        node_features = input_graph.nodes
        if node_features[0] is not None:
            embs = tf.constant(node_features, dtype=tf.float32)
            embs = tf.nn.relu(self.linear(embs))
            embs =self.mlp(embs) + embs
        # else:
        #     embs = embed(self.ids)

        for i in range(self.num_iters):
            embs = self.gnn_attentions[i](embs, input_graph)

        logits = self.final(embs)
        self.outputs = logits

    def predict(self):
        return tf.nn.softmax(self.outputs)
