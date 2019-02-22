from __future__ import absolute_import
import numpy as np
import dgl
import os 
from utils import download, extract_archive, get_download_dir, _get_dgl_url

class TUDataset(object):
    
    _url= r"https://ls11-www.cs.uni-dortmund.de/people/morris/graphkerneldatasets/{}.zip"
    
    def __init__(self, name, use_node_attr=False):
        
        self.name = name
        self.extract_dir = self._download()
        DS_edge_list = self._idx_from_zero(np.loadtxt(self._file_path("A"), delimiter=",", dtype=int))
        DS_indicator = self._idx_from_zero(np.loadtxt(self._file_path("graph_indicator"), dtype=int))
        DS_graph_labels = self._idx_from_zero(np.loadtxt(self._file_path("graph_labels"), dtype=int))
        DS_node_labels = self._idx_from_zero(np.loadtxt(self._file_path("node_labels"), dtype=int))

        g = dgl.DGLGraph()
        g.add_nodes(DS_edge_list.max() + 1)
        g.add_edges(DS_edge_list[:, 0], DS_edge_list[:, 1])
        
        node_idx_list = []
        for idx in range(np.max(DS_indicator)):
            node_idx = np.where(DS_indicator == idx)
            node_idx_list.append(node_idx[0])
        self.graph_lists = g.subgraphs(node_idx_list)
        self.graph_labels = DS_graph_labels
        
        if use_node_attr:
            DS_node_attr = np.loadtxt(self._file_path("node_attributes"), delimiter=",")
            for idxs, g in zip(node_idx_list, self.graph_lists):
                g.ndata['feat'] = DS_node_attr[idxs, :]
    
    def __getitem__(self, idx):
        return self.graph_lists[idx], self.graph_labels[idx]
    
    def _download(self):
        download_dir = get_download_dir()
        zip_file_path = os.path.join(download_dir, "tu_{}.zip".format(self.name))
        download(self._url.format(self.name), path=zip_file_path)
        extract_dir = os.path.join(download_dir, "tu_{}".format(self.name))
        extract_archive(zip_file_path, extract_dir)
        return extract_dir
    
    def _file_path(self, category):
        return os.path.join(self.extract_dir, self.name, "{}_{}.txt".format(self.name, category))
    
    @staticmethod
    def _idx_from_zero(idx_tensor):
        return idx_tensor - np.min(idx_tensor)
        
