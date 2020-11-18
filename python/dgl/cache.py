"""Module for data caching."""

from ._ffi.function import _init_api
from ._ffi.object import register_object, ObjectBase

from . import backend as F
from . import utils

@register_object('cache.SACache')
class CacheIndex(ObjectBase):
    '''This is a Python interface for the cache index implemented in C++.

    The data stored in the cache are identified by Ids. The cache index provides
    a mapping between the Ids and the locations where the data should be stored.
    The location is the offset in a tensor that acts as the cache buffer.
    '''
    def add_data(self, ids):
        '''Add new Ids to the cache.

        When a new Id is added to the cache index successfully, the cache index
        will assign a new location for storing data associated to the Id. However,
        it is possible that an Id is failed to be added to the cache index. In this case,
        the corresponding location will be set to -1.

        Parameters
        ----------
        ids : 1D tensor
            The Ids of data.

        Returns
        -------
            1D tensor that stores the locations for the data associated to the Ids.
        '''
        locs = F.empty(F.shape(ids), F.int64, F.cpu())
        ids = utils.toindex(ids)
        _CAPI_DGLCacheAddData(self, ids.todgltensor(), F.zerocopy_to_dgl_ndarray(locs))
        return locs

    def lookup(self, ids):
        '''Find locations for the Ids.

        It searches the index and return locations for the Ids if the Ids exist
        in the cache.

        This function returns two 1D tensor for locations and data Ids of the same length
        as the input Id tensor. If an Id does not exist, the location is pointed to the end
        of the cache and the corrsponding Id in the output Id tensor will be set to -1.
        This is useful for us to assemble data from the cache efficiently.

        Parameters
        ----------
        ids : 1D tensor
            The Ids of data

        Returns
        -------
            A tuple of 1D tensor for locations and data Ids.
        '''
        locs = F.empty(F.shape(ids), F.int64, F.cpu())
        out_ids = F.empty(F.shape(ids), F.dtype(ids), F.cpu())
        ids = utils.toindex(ids)
        _CAPI_DGLCacheLookup(self, ids.todgltensor(), F.zerocopy_to_dgl_ndarray(locs),
                             F.zerocopy_to_dgl_ndarray(out_ids))
        return locs, out_ids

    def get_cache_size(self):
        '''The cache size.

        The cache size is the number of entries that can be stored in the cache.
        '''
        return _CAPI_DGLGetCacheSize(self)

    def get_num_occupied_entries(self):
        '''The number of entries are occupied.
        '''
        return _CAPI_DGLGetNumOccupied(self)

class Cache:
    '''Cache in the fast memory.

    This cache is designed for only caching data from a single tensor.
    TODO(zhengda) how are we going to deal with heterogeneous graphs?

    This can be used to cache data in GPU to reduce data copy between CPUs and GPUs
    as well as cache data in CPU to reduce data copy between machines.

    To get benefit of caching data in GPUs, the cache needs to have very efficient
    implementation for cache lookup and data assembling in GPUs. The current implementation
    performs cache lookup in CPU, but its implementation is sufficiently fast to benefit
    from reducing data copy between CPUs and GPUs.

    Parameters
    ----------
    cache_size : int
        The number of entries the cache can store.
    shape : a tuple or a list of int
        The shape of the tensor that the cache runs on.
    dtype : data type
        The data type of the tensor that the cache runs on
    device : the framework device type
        The device where data is cached.
    '''
    def __init__(self, cache_size, shape, dtype, device, name):
        self._idx = _CAPI_DGLCacheCreate(cache_size)
        shape = list(shape)
        shape[0] = self._idx.get_cache_size() + 1
        self._shape = tuple(shape)
        self._dtype = dtype
        self._cache_buf = F.zeros(shape, dtype, device)
        self._num_lookups = 0
        self._num_hits = 0
        self._name = name

    def get_cache_size(self):
        '''The cache size.

        The cache size is the number of entries that can be stored in the cache.
        '''
        return self._idx.get_cache_size()

    def get_num_occupied_entries(self):
        '''The number of entries are occupied.
        '''
        return self._idx.get_num_occupied_entries()

    def lookup(self, ids, filtered=True, lazy=False):
        '''Look up the data from the cache.

        It runs in two modes. If filtered=True, the returned data tensor contains only
        the data that can be found in the cache. In other words, the number of rows in
        the data tensor is the number of Ids that can be found in the cache.
        If filtered=False, the number of rows in the returned data tensor will be
        the same as the number of input Ids. If an input Id does not exist in the cache,
        the corresponding row will store arbitrary values and the corresponding location
        in the returned Id tensor will store -1.

        Parameters
        ----------
        ids : 1D tensor
            The input Ids to look up in the cache.
        filtered : bool
            whether we filter the output tensors.
        lazy : bool
            whether or not to return data from the cache lazily.

        Returns
        -------
            A tuple of Id tensor and data tensor.
        '''
        ids = utils.toindex(ids).tousertensor()
        locs, out_ids = self._idx.lookup(ids)
        cache_size = self._idx.get_cache_size()
        if filtered:
            self._num_lookups += len(ids)
            ids = ids[locs < cache_size]
            locs = locs[locs < cache_size]
            if lazy:
                data = lambda: self._cache_buf[locs]
            else:
                data = self._cache_buf[locs]
            self._num_hits += len(ids)
            return ids, data
        else:
            if lazy:
                data = lambda: self._cache_buf[locs]
            else:
                data = self._cache_buf[locs]
            self._num_lookups += len(ids)
            self._num_hits += F.sum(out_ids >= 0, 0)
            return out_ids, data

    def print_stats(self):
        '''Print the statistics of the cache.
        '''
        print('{} has {} lookups and {} hits. hit ratio: {:.3f}'.format(
            self._name, self._num_lookups, self._num_hits,
            int(self._num_hits) / int(self._num_lookups)))

    def add_data(self, ids, data):
        '''Add data to the cache.

        This is used to add data to an empty cache. Some of the data may not be added.
        When adding data to the cache, each data is associated with an Id for identifying the data.

        Returns
        -------
            The Ids of the data that have been successfully added.
        '''
        ids = utils.toindex(ids).tousertensor()
        assert len(ids) == len(data)
        assert F.shape(data)[1:] == self._shape[1:]
        assert F.dtype(data) == self._dtype
        locs = self._idx.add_data(ids)
        self._cache_buf[locs[locs >= 0]] = data[locs >= 0].to(self._cache_buf.device)
        return ids[locs >= 0]

_init_api("dgl.cache")
