from __future__ import absolute_import

from distutils.version import LooseVersion

import tensorflow as tf
from tensorflow.python.eager import context
import builtins
import tfdlpack
import numpy as np
from tfdlpack import to_dlpack, from_dlpack

from ..._ffi.object import ObjectBase
from ... import ndarray as nd
from ... import kernel as K
# import dgl.graph_index
# from ... import GraphIndex
# from dgl.graph_index import GraphIndex
from ...function.base import TargetCode

TF_VERSION = LooseVersion(tf.__version__)


def data_type_dict():
    return {'float16': tf.float16,
            'float32': tf.float32,
            'float64': tf.float64,
            'uint8': tf.uint8,
            'int8': tf.int8,
            'int16': tf.int16,
            'int32': tf.int32,
            'int64': tf.int64}


def cpu():
    return "/cpu:0"
    # return tf.DeviceSpec.from_string('/job:localhost/replica:0/task:0/device:CPU:0')


def tensor(data, dtype=None):
    return tf.convert_to_tensor(data, dtype=dtype)


def as_scalar(data):
    return data.numpy().asscalar()


def get_preferred_sparse_format():
    """Get the preferred sparse matrix format supported by the backend.

    Different backends have their preferred backend. This info is useful when
    constructing a sparse matrix.
    """
    return "coo"


def sparse_matrix(data, index, shape, force_format=False):
    fmt = index[0]
    if fmt != 'coo':
        raise TypeError(
            'Tensorflow backend only supports COO format. But got %s.' % fmt)
    spmat = tf.SparseTensor(indices=tf.transpose(index[1], (1, 0)), values=data, dense_shape=shape)
    return spmat, None


def sparse_matrix_indices(spmat):
    return ('coo', spmat.indices)


def is_tensor(obj):
    return isinstance(obj, tf.Tensor)


def shape(input):
    return input.shape


def dtype(input):
    return input.dtype


def ndim(input):
    return input.ndim


def context(input):
    return input.device
    # return tf.DeviceSpec.from_string(input.device)


def device_type(ctx):
    return tf.DeviceSpec.from_string(ctx).device_type.lower()


def device_id(ctx):
    return tf.DeviceSpec.from_string(ctx).device_index


def astype(input, ty):
    return tf.cast(input, dtype=ty)


def asnumpy(input):
    if isinstance(input, tf.SparseTensor):
        # tf.sparse.to_dense assume sorted indices, need to turn off validate_indices in our cases
        return tf.sparse.to_dense(input, validate_indices=False).numpy() 
    else:
        return input.numpy()


def copy_to(input, ctx):
    with tf.device(ctx):
        new_tensor = tf.identity(input)
    return new_tensor


def sum(input, dim, keepdims=False):
    return tf.reduce_sum(input, axis=dim, keepdims=keepdims)


def reduce_sum(input):
    return tf.reduce_sum(input)


def mean(input, dim):
    return th.reduce_mean(input, axis=dim)


def reduce_mean(input):
    return th.reduce_mean(input)


def max(input, dim):
    return tf.reduce_max(input, axis=dim)


def reduce_max(input):
    return tf.reduce_max(input)


def min(input, dim):
    return tf.reduce_min(input, axis=dim)


def reduce_min(input):
    return tf.reduce_min(input)


def argsort(input, dim, descending):
    if descending:
        return tf.argsort(input, axis=dim, descending="DESCENDING")
    else:
        return tf.argsort(input, axis=dim, descending="ASCENDING")


def topk(input, k, dim, descending=True):
    if dim != -1:
        raise NotImplementedError("Only support last dimension")
    if not descending:
        raise NotImplementedError("Only support descending")

    return tf.math.top_k(input, k=k).values


def argtopk(input, k, dim, descending=True):
    if dim != -1:
        raise NotImplementedError("Only support last dimension")
    if not descending:
        raise NotImplementedError("Only support descending")

    return tf.math.top_k(input, k=k).indices


def exp(input):
    return tf.exp(input)


def softmax(input, dim=-1):
    return tf.math.softmax(input, axis=dim)


def cat(seq, dim):
    return tf.concat(seq, axis=dim)


def stack(seq, dim):
    return tf.stack(seq, axis=dim)


def split(input, sizes_or_sections, dim):
    return tf.split(input, sizes_or_sections, axis=dim)


def repeat(input, repeats, dim):
    tile_shape = np.ones(input.ndim, dtype=np.int32)
    tile_shape[dim] = repeats
    return tf.tile(input, tf.constant(tile_shape))


def gather_row(data, row_index):
    # Weird implementation, because tf.gather require one dimension to be batch_dim
    return tf.gather(data, row_index)


def slice_axis(data, axis, begin, end):
    assert axis == 0
    return tf.slice(data, begin=begin, size=end-begin)


def take(data, indices, dim):
    return tf.gather_nd(data, indices, dim)


def narrow_row(x, start, stop):
    return x[start:stop]


def scatter_row(data, row_index, value):
    # ndim = data.ndim
    # assert ndim == 2
    # for i in range(ndim - 1):
    row_index = tf.expand_dims(row_index, 1)
    return tf.tensor_scatter_nd_update(data, row_index, value)


def scatter_row_inplace(data, row_index, value):
    raise NotImplementedError("Tensorflow doesn't support inplace update")


def squeeze(input, dim):
    return tf.squeeze(input, axis=dim)


def unsqueeze(input, dim):
    return tf.expand_dims(input, axis=dim)


def reshape(input, shape):
    return tf.reshape(input, shape)


def swapaxes(input, axis1, axis2):
    return tf.transpose(input, perm=[axis1, axis2])


def zeros(shape, dtype, ctx):
    with tf.device(ctx):
        t = tf.zeros(shape, dtype=dtype)
    return t


def zeros_like(input):
    return tf.zeros_like(input)


def ones(shape, dtype, ctx):
    with tf.device(ctx):
        t = tf.ones(shape, dtype=dtype)
    return t


def uniform(shape, dtype, ctx, low, high):
    with tf.device(ctx):
        t = tf.random.uniform(shape, dtype=dtype, minval=low, maxval=high)
    return t


def pad_packed_tensor(input, lengths, value, l_min=None):
    raise NotImplementedError


def pack_padded_tensor(input, lengths):
    raise NotImplementedError


def unsorted_1d_segment_sum(input, seg_id, n_segs, dim):
    assert dim == 0  # Why we need dim for 1d?
    return tf.math.unsorted_segment_sum(input, seg_id, n_seg)


def unsorted_1d_segment_mean(input, seg_id, n_segs, dim):
    assert dim == 0  # Why we need dim for 1d?
    return tf.math.unsorted_segment_mean(input, seg_id, n_seg)


def boolean_mask(input, mask):
    return tf.boolean_mask(input, mask)


def equal(x, y):
    return x == y


def logical_not(input):
    return ~input


def unique(input):
    return tf.unique(input).y


def full_1d(length, fill_value, dtype, ctx):
    with tf.device(ctx):
        t = tf.fill([length], value=fill_value)
        t = tf.cast(t, dtype=dtype)
    return t


def nonzero_1d(input):
    nonzero_bool = (input != False)
    return tf.squeeze(tf.where(nonzero_bool))


def sort_1d(input):
    return tf.sort(input), tf.cast(tf.argsort(input), dtype=tf.int64)


def arange(start, stop):
    return tf.range(start, stop, dtype=tf.int64)


def rand_shuffle(arr):
    return tf.random.shuffle(arr)


def zerocopy_to_dlpack(input):
    return tfdlpack.to_dlpack(input)


def zerocopy_from_dlpack(dlpack_tensor):
    # Not zero copy
    return tfdlpack.from_dlpack(dlpack_tensor)


def zerocopy_to_numpy(input):
    # NOTE: not zerocopy
    return np.array(memoryview(input))


def zerocopy_from_numpy(np_array):
    # NOTE: not zerocopy
    return tf.convert_to_tensor(np_array)


def zerocopy_to_dgl_ndarray(input):
    return nd.from_dlpack(zerocopy_to_dlpack(input))


def zerocopy_from_dgl_ndarray(input):
    return zerocopy_from_dlpack(input.to_dlpack())


def one_hot(t, num_classes=-1):
    raise NotImplementedError
    # return th.nn.functional.one_hot(t, num_classes)


# Weird tf convert, still investigating
def convert_dglobject_to_tftensor(obj, dtype=None, name=None, as_ref=False):
    """This only store the handle and erase the type information"""
    return tf.constant([obj.handle.value], dtype=tf.int64, name=name)


def patch_None_tuple(obj, dtype=None, name=None, as_ref=False):
    if obj == (None, None):
        return tf.constant([48966, 0], dtype=tf.int64)
    else:
        return tf.constant(obj, dtype=dtype, name=name)


def convert_back_to_tuple(tf_tensor):
    print("TTTTTTTTTTTTTT")
    if tf_tensor.numpy()[0].item() == 48966:
        return (None, None)
    else:
        return tuple(tf_tensor.numpy())


def dummy_convert(obj, dtype=None, name=None, as_ref=False):
    return tf.constant([0], dtype=tf.int64)


tf.register_tensor_conversion_function(
    ObjectBase, dummy_convert)

tf.register_tensor_conversion_function(
    tuple, patch_None_tuple,  priority=90)

# tf.register_tensor_conversion_function(
#     ObjectBase, convert_dglobject_to_tftensor)
# tf.register_tensor_conversion_function(tuple, patch_None_tuple, priority=90)


def convert_tftensor_to_gindex(tf_tensor):
    from dgl.graph_index import GraphIndex
    handle = tf_tensor.numpy().item()
    cls = GraphIndex
    obj = cls.__new__(cls)
    obj.handle = handle
    return obj


@tf.custom_gradient
def binary_reduce(reducer, binary_op, graph, lhs, rhs, lhs_data, rhs_data,
                  out_size, lhs_map, rhs_map, out_map):
    lhs_data_nd = zerocopy_to_dgl_ndarray(lhs_data)
    rhs_data_nd = zerocopy_to_dgl_ndarray(rhs_data)
    feat_shape = K.infer_binary_feature_shape(
        binary_op, lhs_data_nd, rhs_data_nd)
    out_shape = feat_shape
    if binary_op == 'dot':
        out_shape = feat_shape[:-1]
    # out_data = lhs_data.new_empty((out_size,) + out_shape)
    out_data = tf.zeros((out_size,) + out_shape, dtype=lhs_data.dtype)
    out_data_nd = zerocopy_to_dgl_ndarray(out_data)
    K.binary_op_reduce(
        reducer if reducer != 'mean' else 'sum',
        binary_op, graph, lhs, rhs, lhs_data_nd, rhs_data_nd,
        out_data_nd, lhs_map[0], rhs_map[0], out_map[0])
    # normalize if mean reducer
    # NOTE(zihao): this is a temporary hack and we should have better solution in the future.
    if reducer == 'mean':
        degs = lhs_data.new_empty((out_data.shape[0],))
        degs_nd = zerocopy_to_dgl_ndarray(degs)
        if lhs != TargetCode.DST:  # src or edge
            target = lhs
            n = lhs_data.shape[0]
            in_map = lhs_map[0]
        else:  # rhs != TargetCode.DST
            target = rhs
            n = rhs_data.shape[0]
            in_map = rhs_map[0]
        in_ones = lhs_data.new_ones((n,))
        in_ones_nd = zerocopy_to_dgl_ndarray(in_ones)
        K.copy_reduce(
            'sum', graph, target, in_ones_nd, degs_nd, in_map, out_map[0])
        # reshape
        degs = tf.reshape(degs,
                          (out_data.shape[0],) + (1,) * (out_data.dim() - 1))
        degs = tf.clip_by_value(degs, clip_value_min=1,
                                clip_value_max=np.inf)  # ???
        out_data = out_data / degs
    else:
        degs = None

    def grad(grad_out):
        grad_lhs = None
        grad_rhs = None
        if reducer == 'mean':
            grad_out = grad_out / degs
        grad_out_nd = zerocopy_to_dgl_ndarray(grad_out)
        if True:
            # grad_lhs = grad_out.new_empty((lhs_data_nd.shape[0],) + feat_shape)
            grad_lhs = tf.zeros((lhs_data_nd.shape[0],) + feat_shape)
            K.backward_lhs_binary_op_reduce(
                reducer if reducer != 'mean' else 'sum',
                binary_op, graph, lhs, rhs, lhs_data_nd, rhs_data_nd,
                out_data_nd, grad_out_nd, zerocopy_to_dgl_ndarray(grad_lhs),
                lhs_map[1], rhs_map[1], out_map[1])
            grad_lhs = _reduce_grad(grad_lhs, lhs_data_nd.shape)
        if True:
            # grad_rhs = grad_out.new_empty((rhs_data_nd.shape[0],) + feat_shape)
            grad_rhs = tf.zeros((rhs_data_nd.shape[0],) + feat_shape)
            K.backward_rhs_binary_op_reduce(
                reducer if reducer != 'mean' else 'sum',
                binary_op, graph, lhs, rhs, lhs_data_nd, rhs_data_nd,
                out_data_nd, grad_out_nd, zerocopy_to_dgl_ndarray(grad_rhs),
                lhs_map[1], rhs_map[1], out_map[1])
            grad_rhs = _reduce_grad(grad_rhs, rhs_data_nd.shape)

        return None, None, None, None, None, grad_lhs, grad_rhs, None, None, \
            None, None
    return out_data, grad


@tf.custom_gradient
def copy_reduce(reducer, graph, target, in_data, out_size, in_map,
                out_map):
    out_data = tf.zeros(
        (out_size,) + tuple(in_data.shape[1:]), dtype=in_data.dtype)
    in_data_nd = zerocopy_to_dgl_ndarray(in_data)
    out_data_nd = zerocopy_to_dgl_ndarray(out_data)
    K.copy_reduce(
        reducer if reducer != 'mean' else 'sum',
        graph, target, in_data_nd, out_data_nd, in_map[0], out_map[0])
    # normalize if mean reducer
    # NOTE(zihao): this is a temporary hack and we should have better solution in the future.
    if reducer == 'mean':
        # in_ones = in_data.new_ones((in_data.shape[0],))
        in_ones = tf.ones(in_data.shape[0], dtype=in_data.dtype)
        # degs = in_data.new_empty((out_data.shape[0],))
        degs = tf.zeros(out_data.shape[0], dtype=in_data.dtype)
        in_ones_nd = zerocopy_to_dgl_ndarray(in_ones)
        degs_nd = zerocopy_to_dgl_ndarray(degs)
        K.copy_reduce(
            'sum', graph, target, in_ones_nd, degs_nd, in_map[0], out_map[0])
        # reshape
        degs = tf.reshape(degs,
                          (out_data.shape[0],) + (1,) * (out_data.dim() - 1))
        degs = tf.clip_by_value(degs, clip_value_min=1,
                                clip_value_max=np.inf)  # TODO: ???
        out_data = out_data / degs
    else:
        degs = None
    # save_for_backward can only save variables

    def grad(grad_out):
        if reducer == 'mean':
            grad_out = grad_out / degs
        grad_out_nd = zerocopy_to_dgl_ndarray(grad_out)
        # if ctx.needs_input_grad[3]:
        if True:
            # grad_in = grad_out.new_empty(in_data_nd.shape)
            grad_in = tf.zeros(in_data_nd.shape)
            K.backward_copy_reduce(
                reducer if reducer != 'mean' else 'sum',
                graph, target, in_data_nd, out_data_nd, grad_out_nd,
                zerocopy_to_dgl_ndarray(grad_in), in_map[1], out_map[1])
        return None, None, None, grad_in, None, None, None
    return out_data, grad


def _reduce_grad(grad, shape):
    """Reduce gradient on the broadcast dimension

    If there is broadcast in forward pass, gradients need to be reduced on
    broadcast dimension. This function checks the input tensor shape and
    gradient shape and perform the reduction.

    Parameters
    ----------
    grad: Tensor
        Gradient tensor
    shape: tuple
        Shape of input tensor

    Returns
    -------
    Tensor
    """
    grad_shape = grad.shape[1:]
    in_shape = shape[1:]
    if in_shape == grad_shape:
        # no need to reduce
        return grad
    num_to_squeeze = len(grad_shape) - len(in_shape)
    # pad inshape
    in_shape = (1,) * num_to_squeeze + in_shape
    reduce_idx = np.nonzero(np.array(grad_shape) - np.array(in_shape))
    reduce_idx += 1  # skip batch dim
    grad = tf.reduce_sum(grad, axis=tuple(reduce_idx), keepdims=True)
    return tf.reshape(grad, shape)


def sync():
    context = context().context()
    context.async_wait()
