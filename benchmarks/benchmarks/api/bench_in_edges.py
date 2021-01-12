import time
import dgl
import torch
import numpy as np

from .. import utils


@utils.benchmark('time', timeout=1200)
@utils.parametrize_cpu('graph_name', ['cora', 'livejournal', 'friendster'])
@utils.parametrize_gpu('graph_name', ['cora', 'livejournal'])
# in_edges on coo is not supported on cuda
@utils.parametrize_cpu('format', ['coo', 'csc'])
@utils.parametrize_gpu('format', ['csc'])
@utils.parametrize('fraction', [0.01, 0.1])
def track_time(graph_name, format, fraction):
    device = utils.get_bench_device()
    graph = utils.get_graph(graph_name, format)
    graph = graph.to(device)
    nids = np.random.RandomState(666).choice(
        np.arange(graph.num_nodes(), dtype=np.int64), int(graph.num_nodes()*fraction))
    nids = torch.tensor(nids, device=device, dtype=torch.int64)

    # dry run
    for i in range(10):
        out = graph.in_edges(i)

    # timing
    t0 = time.time()
    for i in range(10):
        edges = graph.in_edges(nids)
    t1 = time.time()

    return (t1 - t0) / 10
