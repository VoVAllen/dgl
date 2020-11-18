"""OAG dataset.

"""
from __future__ import absolute_import

import numpy as np
import scipy.sparse as sp
import os, sys
from os import listdir

from .utils import save_graphs, load_graphs, save_info, load_info, makedirs, _get_dgl_url
from .utils import generate_mask_tensor
from .utils import deprecate_property, deprecate_function
from .dgl_dataset import DGLBuiltinDataset
from .. import convert
from .. import batch
from .. import backend as F
from ..convert import graph as dgl_graph
from ..convert import heterograph
from ..convert import from_networkx, to_networkx

backend = os.environ.get('DGLBACKEND', 'pytorch')

class OAGDataset(DGLBuiltinDataset):
    r"""The open academic graph dataset. It has three versions: OAG CS, OAG Med and OAG full.

    Parameters
    -----------
    name: str
      name can be 'cs', 'med' or 'pull'.
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Default: ~/.dgl/
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose: bool
      Whether to print out progress information. Default: True.
    """
    _urls = {
        'cs' : 'dataset/OAG/OAG_CS2.zip',
        'med' : 'dataset/OAG/OAG_med2.zip',
    }
    _internal_dirs = {
        'cs' : 'OAG_CS',
        'med' : 'OAG_med',
    }

    def __init__(self, name, raw_dir=None, force_reload=False, verbose=True):
        assert name.lower() in ['cs', 'med']

        url = _get_dgl_url(self._urls[name])
        super(OAGDataset, self).__init__(name, url=url,
                                         raw_dir=raw_dir,
                                         force_reload=force_reload,
                                         verbose=verbose)

    def process(self):
        """Loads input data from data directory

        """
        root = self.raw_path

        dir_path = root + '/' + self._internal_dirs[self.name]
        spms = {}
        for file_name in listdir(dir_path):
            if '.npz' not in file_name:
                continue
            spm = sp.load_npz(dir_path + '/' + file_name).tocoo().transpose()
            file_name = file_name.replace('.npz', '')
            head, rel, tail = file_name.split('-')
            spms[(head, rel, tail)] = (spm.row, spm.col)
        g = heterograph(spms)
        for file_name in listdir(dir_path):
            if '_emb.npy' not in file_name:
                continue
            feat = np.load(dir_path + '/' + file_name)
            ntype = file_name.replace('_emb.npy', '')
            g.nodes[ntype].data['emb'] = F.tensor(feat)

        g.nodes['field'].data['level'] = F.tensor(np.load(dir_path + '/field_level.npy'))
        #TODO(zhengda) let's not load time first.
        #g.nodes['paper'].data['time'] = F.tensor(np.load(dir_path + '/paper_time.npy'))

        self._g = g
        self._graph = g

        if self.verbose:
            print('Finished data loading and preprocessing.')
            for ntype in g.ntypes:
                print('  NumNodes of {}: {}'.format(ntype, g.number_of_nodes(ntype)))
            for etype in g.canonical_etypes:
                print('  NumEdges of {}: {}'.format(etype, g.number_of_edges(etype)))
            for ntype in g.ntypes:
                for name in g.nodes[ntype].data:
                    shape = g.nodes[ntype].data[name].shape
                    ndim = shape[1] if len(shape) > 1 else 1
                    print('  NumFeats {} of node {}: {}'.format(name, ntype, ndim))


    def has_cache(self):
        return False

    def save(self):
        """save the graph list and the labels"""
        pass

    def load(self):
        self.process()

    def __getitem__(self, idx):
        assert idx == 0, "This dataset has only one graph"
        return self._g

    def __len__(self):
        return 1

    @property
    def save_name(self):
        return self.name + '_dgl_graph'

    @property
    def graph(self):
        deprecate_property('dataset.graph', 'dataset.g')
        return self._graph
