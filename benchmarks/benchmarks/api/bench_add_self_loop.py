import time
import dgl
import torch
import numpy as np
import dgl.function as fn


from .. import utils


@utils.benchmark('time')
@utils.parametrize('graph_name', ['cora', 'livejournal'])
@utils.parametrize('format', ['coo', 'csc'])
def track_time(graph_name, format):
    device = utils.get_bench_device()
    graph = utils.get_graph(graph_name, format)
    graph = graph.to(device)

    # dry run
    for i in range(3):
        g = graph.add_self_loop()

    # timing
    t0 = time.time()
    for i in range(3):
        edges = graph.add_self_loop()
    t1 = time.time()

    return (t1 - t0) / 3
