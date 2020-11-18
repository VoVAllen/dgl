import numpy as np
import time
import dgl
import torch as th
import numpy as np
from pyinstrument import Profiler

def gather_row(data, cache, ids):
    found_ids, found_data = cache.lookup(ids, filtered=False)
    bool_idx = found_ids < 0
    found_data[bool_idx] = data[ids[bool_idx]].to(found_data.device)
    return found_data

def _test_cache(device):
    id_space_size = 1000 * 1000 * 20
    cache_size = 1000 * 1000 * 15
    cache = dgl.Cache(cache_size, (0, 100), th.float32, device)
    data = th.empty((id_space_size, 100), dtype=th.float32).uniform_(0, 1)

    cache_ids = np.random.choice(id_space_size, size=cache_size, replace=False)
    cached_ids = cache.add_data(cache_ids, data[cache_ids])
    print('try to add {} entries and {} are added'.format(len(cache_ids), len(cached_ids)))

    read_ids = []
    for _ in range(10):
        read_ids.append(np.random.choice(id_space_size, size=2000 * 1000))

    found_ids, _ = cache.lookup(read_ids[0], filtered=False)
    print('cache hits:', float(th.sum(found_ids >= 0)) / len(read_ids[0]))

    profiler = Profiler()
    profiler.start()

    start = time.time()
    for ids in read_ids:
        read_data = gather_row(data, cache, ids)
    print('read data with cache: {:.3f} seconds'.format(time.time() - start))

    start = time.time()
    for ids in read_ids:
        read_data = data[ids].to(device)
    print('read data directly: {:.3f} seconds'.format(time.time() - start))

    profiler.stop()
    print(profiler.output_text(unicode=True, color=True))

def test_cache():
    device = th.device('cuda')
    _test_cache(device)

if __name__ == '__main__':
    test_cache()

