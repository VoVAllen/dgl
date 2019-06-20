from __future__ import absolute_import

import ctypes
import numpy as np
import networkx as nx
import scipy

from ._ffi.base import c_array
from ._ffi.function import _init_api
from .base import DGLError
from . import backend as F
from . import utils
from .ndarray import from_dlpack, NDArrayBase
# from ._ffi._ctypes.function import _make_array

_init_api("dgl.graph_serialize")

GraphIndexHandle = ctypes.c_void_p


# def saveDGLGraph():
from .graph import DGLGraph
g = DGLGraph()
g.add_nodes(10)
g.add_edges(1, 2)
g.add_edges(3, 2)
g.add_edges(3, 3)

import torch as th

g.edata['e1'] = th.ones(3, 5)
g.ndata['n1'] = th.ones(10, 2)

graph_list = [g._graph]
inputs = c_array(GraphIndexHandle, [gr._handle for gr in graph_list])
inputs = ctypes.cast(inputs, ctypes.c_void_p)

def wrap_tensor_list(l):
    # nl = []
    # for item in l:
    #     nl.append(from_dlpack(item))
    inputs = c_array(NDArrayBase, l)
    inputs = ctypes.cast(inputs, NDArrayBase)
    return inputs

def wrap_str_list(l):
    inputs = c_array(ctypes.c_char_p, l)
    inputs = ctypes.cast(inputs, ctypes.c_char_p)
    return inputs


_CAPI_DGLSaveGraphs(inputs, len(graph_list),
                    wrap_str_list(["e1"]), wrap_tensor_list([F.zerocopy_to_dgl_ndarray(th.ones(3, 5))]), 1,
                    wrap_str_list(["n1"]), wrap_tensor_list([F.zerocopy_to_dgl_ndarray(th.ones(10, 2))]), 1,
                    "/tmp/test.bin")
