import dgl
import numpy as np
from data import MovieLens

ds = MovieLens('ml-100k', 'cpu', use_one_hot_fea=True)

g = ds.train_dec_graph


def sample_graph(g, eid, h=1, sample_ratio=1.0, max_nodes_per_hop=None):
    subgraph_eids = [eid]
    u, v = g.find_edges(eid)
    u, v = int(u), int(v)
    u_nodes = [u]
    v_nodes = [v]
    u_visited = [u]
    v_visited = [v]

    for dist in range(1, h+1):
        all_v_neighbor = g.in_edges(v, 'all')
        all_u_neighbor = g.in_edges(u, 'all')
        u_fringe = np.setdiff1d(
            all_v_neighbor[0].numpy(), np.array(u_visited))
        v_fringe = np.setdiff1d(
            all_u_neighbor[0].numpy(), np.array(v_visited))
        u_visited = np.union1d(u_fringe, u_visited)
        v_visited = np.union1d(v_fringe, v_visited)

        if sample_ratio < 1.0:
            u_fringe = np.random.choice(
                u_fringe, int(sample_ratio*len(u_fringe)))
            v_fringe = np.random.choice(
                v_fringe, int(sample_ratio*len(v_fringe)))
        if max_nodes_per_hop is not None:
            if max_nodes_per_hop < len(u_fringe):
                u_fringe = np.random.choice(u_fringe, max_ndoes_per_hop)
            
                if max_nodes_per_hop < len(v_fringe):
                    v_fringe = np.random.choice(v_fringe, max_ndoes_per_hop)
            
            u_nodes = u_nodes + list(u_fringe)
            v_nodes = v_nodes + list(v_fringe)

    subgraph_eids =g.edge_ids(u_nodes, v_nodes)
    return g.edge_subgraph({'rate': subgraph_eids})

