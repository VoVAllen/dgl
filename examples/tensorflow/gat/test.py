import tensorflow as tf
from tensorflow.keras import layers
import dgl.function as fn
from dgl.nn.tensorflow import edge_softmax, GATConv
import dgl

model = GATConv(3, 10, 8)

features = tf.ones([6, 3])
g = dgl.DGLGraph()
g.add_nodes(6)
g.add_edges([0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 0, 5])
optimizer = tf.keras.optimizers.Adam()

with tf.GradientTape() as tape:
    tape.watch(model.trainable_weights)
    logits = model(g, features)
    loss_value = tf.reduce_sum(logits)

    grads = tape.gradient(loss_value, model.trainable_weights)
    print(grads)
    # optimizer.apply_gradients(zip(grads, model.trainable_weights))