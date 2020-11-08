import unittest, pytest
import numpy as np
import dgl
import numpy as np
import backend as F

def gather_row(data, cache, ids):
    found_ids, found_data = cache.lookup(ids, filtered=False)
    print('cache hits:', float(F.sum(found_ids >= 0, 0)) / len(ids))
    idx = F.nonzero_1d(found_ids < 0)
    found_data[idx] = data[ids[idx]]
    return found_data

def test_cache():
    id_space_size = 100000
    cache = dgl.Cache(10000, (0, 10), F.float32, F.cpu(), 'test')
    data = F.uniform((id_space_size, 10), F.float32, F.cpu(), 0, 1)
    ids = np.random.choice(id_space_size, size=1000)
    added_ids = cache.add_data(ids, data[ids])
    print('try to add {} entries and {} are added'.format(len(ids), len(added_ids)))

    added_data = data[added_ids]
    found_ids, found_data = cache.lookup(added_ids)
    assert np.all(F.asnumpy(found_ids) == F.asnumpy(added_ids))
    assert np.all(F.asnumpy(found_data) == F.asnumpy(added_data))

    read_ids = np.random.choice(id_space_size, size=1000)
    read_data = gather_row(data, cache, read_ids)
    assert np.all(read_data.numpy() == data[read_ids].numpy())

if __name__ == '__main__':
    test_cache()
