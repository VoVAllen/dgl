"""Define distributed tensor."""

import os

from .dist_context import is_initialized
from .kvstore import get_kvstore
from .role import get_role
from .. import utils
from .. import backend as F
from ..cache import Cache

def _default_init_data(shape, dtype):
    return F.zeros(shape, dtype, F.cpu())

# These Ids can identify the anonymous distributed tensors.
DIST_TENSOR_ID = 0

class Future:
    '''Future wrapper that combines data from KVStore future and cache future.

    Parameters
    ----------
    kv_future : Future
        The KVStore future
    ids : 1D tensor
        The Ids of data referenced by the cache future.
    cache_future : Future
        The cache future.
    '''
    def __init__(self, kv_future, ids, cache_future):
        self._kv_future = kv_future
        self._ids = ids
        self._cache_future = cache_future

    def get_kv_future(self):
        '''Get the kvstore future.
        '''
        return self._kv_future

    def __call__(self):
        kv_data = self._kv_future()
        assert kv_data is not None
        if self._cache_future is None:
            return kv_data
        else:
            data = self._cache_future()
            data[self._ids < 0] = kv_data.to(data.device)
            return data


class DistTensor:
    ''' Distributed tensor.

    ``DistTensor`` references to a distributed tensor sharded and stored in a cluster of machines.
    It has the same interface as Pytorch Tensor to access its metadata (e.g., shape and data type).
    To access data in a distributed tensor, it supports slicing rows and writing data to rows.
    It does not support any operators of a deep learning framework, such as addition and
    multiplication.

    Currently, distributed tensors are designed to store node data and edge data of a distributed
    graph. Therefore, their first dimensions have to be the number of nodes or edges in the graph.
    The tensors are sharded in the first dimension based on the partition policy of nodes
    or edges. When a distributed tensor is created, the partition policy is automatically
    determined based on the first dimension if the partition policy is not provided: if the first
    dimension matches the number of nodes, ``DistTensor`` will use the node partition policy;
    if the first dimension matches the number of edges, ``DistTensor`` wll use the edge partition
    policy. To determine the partition policy automatically, a DistGraph object has to be created.
    Users can overwrite the rule by providing a partition policy directly.

    A distributed tensor can be ether named or anonymous.
    When a distributed tensor has a name, the tensor can be persistent if ``persistent=True``.
    Normally, DGL destroys the distributed tensor in the system when the ``DistTensor`` object
    goes away. However, a persistent tensor lives in the system even if
    the ``DistTenor`` object disappears in the trainer process. The persistent tensor has
    the same life span as the DGL servers. DGL does not allow an anonymous tensor to be persistent.

    When a ``DistTensor`` object is created, it may reference to an existing distributed tensor or
    create a new one. A distributed tensor is identified by the name passed to the constructor.
    If the name exists, ``DistTensor`` will reference the existing one.
    In this case, the shape and the data type must match the existing tensor.
    If the name doesn't exist, a new tensor will be created in the kvstore.

    When a distributed tensor is created, its values are initialized to zero. Users
    can define an initialization function to control how the values are initialized.
    The init function has two input arguments: shape and data type and returns a tensor.
    Below shows an example of an init function:

    .. highlight:: python
    .. code-block:: python

        def init_func(shape, dtype):
            return torch.ones(shape=shape, dtype=dtype)

    Parameters
    ----------
    shape : tuple
        The shape of the tensor. The first dimension has to be the number of nodes or
        the number of edges of a distributed graph.
    dtype : dtype
        The dtype of the tensor. The data type has to be the one in the deep learning framework.
    name : string, optional
        The name of the embeddings. The name can uniquely identify embeddings in a system
        so that another ``DistTensor`` object can referent to the distributed tensor.
    init_func : callable, optional
        The function to initialize data in the tensor. If the init function is not provided,
        the values of the embeddings are initialized to zero.
    part_policy : PartitionPolicy, optional
        The partition policy of the rows of the tensor to different machines in the cluster.
        Currently, it only supports node partition policy or edge partition policy.
        The system determines the right partition policy automatically.
    persistent : bool
        Whether the created tensor lives after the ``DistTensor`` object is destroyed.

    Examples
    --------
    >>> init = lambda shape, dtype: th.ones(shape, dtype=dtype)
    >>> arr = dgl.distributed.DistTensor((g.number_of_nodes(), 2), th.int32, init_func=init)
    >>> print(arr[0:3])
    tensor([[1, 1],
            [1, 1],
            [1, 1]], dtype=torch.int32)
    >>> arr[0:3] = th.ones((3, 2), dtype=th.int32) * 2
    >>> print(arr[0:3])
    tensor([[2, 2],
            [2, 2],
            [2, 2]], dtype=torch.int32)

    Note
    ----
    The creation of ``DistTensor`` is a synchronized operation. When a trainer process tries to
    create a ``DistTensor`` object, the creation succeeds only when all trainer processes
    do the same.
    '''
    def __init__(self, shape, dtype, name=None, init_func=None, part_policy=None,
                 persistent=False):
        self.kvstore = get_kvstore()
        assert self.kvstore is not None, \
                'Distributed module is not initialized. Please call dgl.distributed.initialize.'
        self._shape = shape
        self._dtype = dtype

        part_policies = self.kvstore.all_possible_part_policy
        # If a user doesn't provide a partition policy, we should find one based on
        # the input shape.
        if part_policy is None:
            for policy_name in part_policies:
                policy = part_policies[policy_name]
                if policy.get_size() == shape[0]:
                    # If multiple partition policies match the input shape, we cannot
                    # decide which is the right one automatically. We should ask users
                    # to provide one.
                    assert part_policy is None, \
                            'Multiple partition policies match the input shape. ' \
                            + 'Please provide a partition policy explicitly.'
                    part_policy = policy
            assert part_policy is not None, \
                    'Cannot find a right partition policy. It is either because ' \
                    + 'its first dimension does not match the number of nodes or edges ' \
                    + 'of a distributed graph or there does not exist a distributed graph.'

        self._part_policy = part_policy
        assert part_policy.get_size() == shape[0], \
                'The partition policy does not match the input shape.'

        if init_func is None:
            init_func = _default_init_data
        exist_names = self.kvstore.data_name_list()
        # If a user doesn't provide a name, we generate a name ourselves.
        # We need to generate the name in a deterministic way.
        if name is None:
            assert not persistent, 'We cannot generate anonymous persistent distributed tensors'
            global DIST_TENSOR_ID
            # All processes of the same role should create DistTensor synchronously.
            # Thus, all of them should have the same Ids.
            name = 'anonymous-' + get_role() + '-' + str(DIST_TENSOR_ID)
            DIST_TENSOR_ID += 1
        assert isinstance(name, str), 'name {} is type {}'.format(name, type(name))
        data_name = part_policy.get_data_name(name)
        self._name = str(data_name)
        self._persistent = persistent
        if self._name not in exist_names:
            self.kvstore.init_data(self._name, shape, dtype, part_policy, init_func)
            self._owner = True
        else:
            self._owner = False
            dtype1, shape1, _ = self.kvstore.get_data_meta(self._name)
            assert dtype == dtype1, 'The dtype does not match with the existing tensor'
            assert shape == shape1, 'The shape does not match with the existing tensor'

        self._gpu_cache = None

    def __del__(self):
        initialized = os.environ.get('DGL_DIST_MODE', 'standalone') == 'standalone' \
                or is_initialized()
        if not self._persistent and self._owner and initialized:
            self.kvstore.delete_data(self._name)

    def __getitem__(self, idx):
        idx = utils.toindex(idx)
        idx = idx.tousertensor()
        if self._gpu_cache is None:
            return self.kvstore.pull(name=self._name, id_tensor=idx)
        else:
            # Read data from GPU cache.
            out_idx, out_data = self._gpu_cache.lookup(idx, filtered=False, lazy=False)
            # Pull remaining data from KVStore.
            kv_data = self.kvstore.pull(name=self._name, id_tensor=idx[out_idx < 0])
            # Merge the data from the GPU cache and from KVStore.
            out_data[out_idx < 0] = kv_data.to(out_data.device)
            return out_data

    def __setitem__(self, idx, val):
        # TODO(zhengda) setting data doesn't work with cache.
        assert self._gpu_cache is None
        idx = utils.toindex(idx)
        idx = idx.tousertensor()
        # TODO(zhengda) how do we want to support broadcast (e.g., G.ndata['h'][idx] = 1).
        self.kvstore.push(name=self._name, id_tensor=idx, data_tensor=val)

    def __len__(self):
        return self._shape[0]

    def init_cache(self, cache_size, cache_nodes, cache_priority, device):
        '''This initializes GPU cache for the tensor.
        '''
        cache_nodes = utils.toindex(cache_nodes)
        cache_nodes = cache_nodes.tousertensor()
        self._gpu_cache = Cache(cache_size, self.shape, self.dtype, device, 'GPU cache')
        cache_data = self.__getitem__(cache_nodes)
        idx = F.sort_1d(-cache_priority)[1]
        cache_nodes = cache_nodes[idx]
        cache_data = cache_data[idx]
        self._gpu_cache.add_data(cache_nodes, cache_data)
        occ_per = self._gpu_cache.get_num_occupied_entries() / self._gpu_cache.get_cache_size()
        print('GPU cache size: {}, {:.3f} % is used'.format(self._gpu_cache.get_cache_size(),
                                                            occ_per * 100))

    @property
    def part_policy(self):
        '''Return the partition policy

        Returns
        -------
        PartitionPolicy
            The partition policy of the distributed tensor.
        '''
        return self._part_policy

    @property
    def shape(self):
        '''Return the shape of the distributed tensor.

        Returns
        -------
        tuple
            The shape of the distributed tensor.
        '''
        return self._shape

    @property
    def dtype(self):
        '''Return the data type of the distributed tensor.

        Returns
        ------
        dtype
            The data type of the tensor.
        '''
        return self._dtype

    @property
    def name(self):
        '''Return the name of the distributed tensor

        Returns
        -------
        str
            The name of the tensor.
        '''
        return self._name

    def prefetch(self, idx):
        """Prefect data from distrubuted tensor

        Parameters
        ----------
        idx : tensor
            data index

        Returns
        -------
        Future
            A future object that can be waited on
        """
        if self._gpu_cache is not None:
            out_idx, out_data = self._gpu_cache.lookup(idx, filtered=False, lazy=True)
            # We only need to fetch data that don't exist in the GPU cache.
            idx = idx[out_idx < 0]
        else:
            out_idx = None
            out_data = None
        fut = self.kvstore.async_pull([self.name], [idx])
        return Future(fut[0], out_idx, out_data)

    def print_cache_stats(self):
        '''Print cache statistics.
        '''
        if self._gpu_cache is not None:
            self._gpu_cache.print_stats()
