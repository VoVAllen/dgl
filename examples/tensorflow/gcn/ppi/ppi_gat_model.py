import graph_nets as gn
import sonnet as snt
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from graph_nets import _base

from graph_nets import blocks



def _unsorted_segment_softmax(data,
                              segment_ids,
                              num_segments,
                              name="unsorted_segment_softmax"):
  """Performs an elementwise softmax operation along segments of a tensor.

  The input parameters are analogous to `tf.math.unsorted_segment_sum`. It
  produces an output of the same shape as the input data, after performing an
  elementwise sofmax operation between all of the rows with common segment id.

  Args:
    data: A tensor with at least one dimension.
    segment_ids: A tensor of indices segmenting `data` across the first
      dimension.
    num_segments: A scalar tensor indicating the number of segments. It should
      be at least `max(segment_ids) + 1`.
    name: A name for the operation (optional).

  Returns:
    A tensor with the same shape as `data` after applying the softmax operation.

  """
  with tf.name_scope(name):
    segment_maxes = tf.math.unsorted_segment_max(data, segment_ids,
                                                 num_segments)
    maxes = tf.gather(segment_maxes, segment_ids)
    # Possibly refactor to `tf.stop_gradient(maxes)` for better performance.
    data -= maxes
    exp_data = tf.exp(data)
    segment_sum_exp_data = tf.math.unsorted_segment_sum(exp_data, segment_ids,
                                                        num_segments)
    sum_exp_data = tf.gather(segment_sum_exp_data, segment_ids)
    return exp_data / sum_exp_data


def _received_edges_normalizer(graph,
                               normalizer,
                               name="received_edges_normalizer"):
  """Performs elementwise normalization for all received edges by a given node.

  Args:
    graph: A graph containing edge information.
    normalizer: A normalizer function following the signature of
      `modules._unsorted_segment_softmax`.
    name: A name for the operation (optional).

  Returns:
    A tensor with the resulting normalized edges.

  """
  with tf.name_scope(name):
    return normalizer(
        data=graph.edges,
        segment_ids=graph.receivers,
        num_segments=tf.reduce_sum(graph.n_node))


class SelfAttention(_base.AbstractModule):
  """Multi-head self-attention module.

  The module is based on the following three papers:
   * A simple neural network module for relational reasoning (RNs):
       https://arxiv.org/abs/1706.01427
   * Non-local Neural Networks: https://arxiv.org/abs/1711.07971.
   * Attention Is All You Need (AIAYN): https://arxiv.org/abs/1706.03762.

  The input to the modules consists of a graph containing values for each node
  and connectivity between them, a tensor containing keys for each node
  and a tensor containing queries for each node.

  The self-attention step consist of updating the node values, with each new
  node value computed in a two step process:
  - Computing the attention weights between each node and all of its senders
   nodes, by calculating sum(sender_key*receiver_query) and using the softmax
   operation on all attention weights for each node.
  - For each receiver node, compute the new node value as the weighted average
   of the values of the sender nodes, according to the attention weights.
  - Nodes with no received edges, get an updated value of 0.

  Values, keys and queries contain a "head" axis to compute independent
  self-attention for each of the heads.

  """

  def __init__(self, 
                 out_feats,
                 num_heads,
                 name="self_attention"):
    """Inits the module.

    Args:
      name: The module name.
    """
    super(SelfAttention, self).__init__(name=name)
    self._normalizer = _unsorted_segment_softmax

    xinit = tf.keras.initializers.VarianceScaling(scale=np.sqrt(
        2), mode="fan_avg", distribution="untruncated_normal")
    self.fc = layers.Dense(
        out_feats * num_heads, use_bias=False, kernel_initializer=xinit)
    
    self.attn_l = tf.Variable(initial_value=xinit(
        shape=(1, num_heads, out_feats), dtype='float32'), trainable=True)
    self.attn_r = tf.Variable(initial_value=xinit(
        shape=(1, num_heads, out_feats), dtype='float32'), trainable=True)
    self._num_heads = num_heads
    self._out_feats = out_feats

  def _build(self, inputs, attention_graph):
    """Connects the multi-head self-attention module.

    The self-attention is only computed according to the connectivity of the
    input graphs, with receiver nodes attending to sender nodes.

    Args:
      node_values: Tensor containing the values associated to each of the nodes.
        The expected shape is [total_num_nodes, num_heads, key_size].
      node_keys: Tensor containing the key associated to each of the nodes. The
        expected shape is [total_num_nodes, num_heads, key_size].
      node_queries: Tensor containing the query associated to each of the nodes.
        The expected shape is [total_num_nodes, num_heads, query_size]. The
        query size must be equal to the key size.
      attention_graph: Graph containing connectivity information between nodes
        via the senders and receivers fields. Node A will only attempt to attend
        to Node B if `attention_graph` contains an edge sent by Node A and
        received by Node B.

    Returns:
      An output `graphs.GraphsTuple` with updated nodes containing the
      aggregated attended value for each of the nodes with shape
      [total_num_nodes, num_heads, value_size].

    Raises:
      ValueError: if the input graph does not have edges.
    """

    # Sender nodes put their keys and values in the edges.
    # [total_num_edges, num_heads, query_size]
    h_src= inputs
    feat_src = feat_dst = tf.reshape(
                    self.fc(h_src), (-1, self._num_heads, self._out_feats))

    el = tf.reduce_sum(feat_src * self.attn_l, axis=-1, keepdims=True)
    er = tf.reduce_sum(feat_dst * self.attn_r, axis=-1, keepdims=True)

    sender_vals = blocks.broadcast_sender_nodes_to_edges(
        attention_graph.replace(nodes=el))
    receiver_vals = blocks.broadcast_sender_nodes_to_edges(
        attention_graph.replace(nodes=er))
    
    # [total_num_edges, num_heads, value_size]

    # Attention weight for each edge.
    # [total_num_edges, num_heads]
    attention_weights_logits = sender_vals + receiver_vals
    normalized_attention_weights = _received_edges_normalizer(
        attention_graph.replace(edges=attention_weights_logits),
        normalizer=self._normalizer)

    # Attending to sender values according to the weights.
    # [total_num_edges, num_heads, embedding_size]
    sender_values = blocks.broadcast_sender_nodes_to_edges(
        attention_graph.replace(nodes=feat_src))
    attented_edges = sender_values * normalized_attention_weights

    # Summing all of the attended values from each node.
    # [total_num_nodes, num_heads, embedding_size]
    received_edges_aggregator = blocks.ReceivedEdgesToNodesAggregator(
        reducer=tf.math.unsorted_segment_sum)
    aggregated_attended_values = received_edges_aggregator(
        attention_graph.replace(edges=attented_edges))

    return attention_graph.replace(nodes=aggregated_attended_values)



class GraphAttention(_base.AbstractModule):
    def __init__(self, hidden, num_heads, name="graph_attention"):
        super(GraphAttention, self).__init__(name=name)
        self.hidden = hidden
        self.num_heads = num_heads
        
        self.attention = SelfAttention(hidden, num_heads)
        # self.output_proj = snt.Linear(self.hidden, with_bias=False)
        # self.mlp = snt.nets.MLP([self.hidden, self.hidden])

    def _build(self, inputs, input_graph):
        # norm1 = snt.BatchNorm()
        # norm2 = snt.BatchNorm()

        # vals, keys, queries = self.head(inputs)
        attended_nodes = self.attention(inputs, input_graph).nodes
        # output_projected = tf.nn.elu(attended_nodes)
        # output_projected = self.output_proj(tf.reduce_sum(attended_nodes, axis=1))

        # This skip-connection is non-orthodox
        # output_projected = output_projected + \
        #     tf.reshape(vals, [-1, self.hidden])

        # normalized = self.mlp(output_projected) + output_projected
        output_projected = tf.nn.elu(attended_nodes)
        return output_projected


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
        # self.mlp = snt.nets.MLP([self.hidden, self.hidden], activate_final=False)
        self.gnn_attentions = []
        # self.final = snt.Linear(self.num_classes)

        for _ in range(self.num_iters):
            gnn_attention = GraphAttention(self.hidden, 4)
            self.gnn_attentions.append(gnn_attention)


        gnn_attention = GraphAttention(num_classes, 4)
        self.gnn_attentions.append(gnn_attention)

    def _build(self, input_graph):
        # embed = snt.Embed(self.num_nodes, self.emb_size)
        # norm = snt.BatchNorm()
        

        node_features = input_graph.nodes
        if node_features[0] is not None:
            embs = tf.constant(node_features, dtype=tf.float32)
            # embs = tf.nn.relu(self.linear(embs))
            # embs =self.mlp(embs) + embs
        # else:
        #     embs = embed(self.ids)

        for i in range(self.num_iters):
            embs = self.gnn_attentions[i](embs, input_graph)
            embs = tf.reshape(embs, (embs.shape[0], -1))

        embs = self.gnn_attentions[-1](embs, input_graph)
        print(embs.shape)
        logits = tf.reduce_mean(embs, 1)
        # logits = self.final(embs)
        self.outputs = logits

    def predict(self):
        return tf.nn.softmax(self.outputs)
