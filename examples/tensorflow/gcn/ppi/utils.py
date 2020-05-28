import numpy as np
import tensorflow as tf

def initializer(dim):
    m = 1 / np.sqrt(dim)
    return tf.initializers.random_uniform(-m, m)