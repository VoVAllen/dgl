#!/usr/bin/env python
# coding: utf-8
# pylint: disable=C0103, C0111, E1101, W0612
"""Implementation of MPNN model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import dgl.function as fn
import dgl.nn.pytorch as dgl_nn
from dgl.nn.pytorch import NNConv


class MPNNModel(nn.Module):
    """
    MPNN model from:
        Gilmer, Justin, et al.
        Neural message passing for quantum chemistry.
    """

    def __init__(self,
                 node_input_dim=15,
                 edge_input_dim=5,
                 output_dim=12,
                 node_hidden_dim=64,
                 edge_hidden_dim=128,
                 num_step_message_passing=6,
                 num_step_set2set=6,
                 num_layer_set2set=3):
        """model parameters setting

        Args:
            node_input_dim: dimension of input node feature
            edge_input_dim: dimension of input edge feature
            output_dim: dimension of prediction
            node_hidden_dim: dimension of node feature in hidden layers
            edge_hidden_dim: dimension of edge feature in hidden layers
            num_step_message_passing: number of message passing steps
            num_step_set2set: number of set2set steps
            num_layer_ste2set: number of set2set layers
        """

        super().__init__()
        self.name = "MPNN"
        self.num_step_message_passing = num_step_message_passing
        self.lin0 = nn.Linear(node_input_dim, node_hidden_dim)
        edge_network = nn.Sequential(
            nn.Linear(edge_input_dim, edge_hidden_dim), nn.ReLU(),
            nn.Linear(edge_hidden_dim, node_hidden_dim * node_hidden_dim))
        self.conv = NNConv(in_feats=node_hidden_dim,
                           out_feats=node_hidden_dim,
                           edge_func=edge_network,
                           aggregator_type="sum",
                           residual=False,
                           bias=True)
        self.gru = nn.GRU(node_hidden_dim, node_hidden_dim)

        self.set2set = dgl_nn.glob.Set2Set(node_hidden_dim, num_step_set2set,
                                           num_layer_set2set)
        self.lin1 = nn.Linear(2 * node_hidden_dim, node_hidden_dim)
        self.lin2 = nn.Linear(node_hidden_dim, output_dim)

    def forward(self, g):
        h = g.ndata['n_feat']
        out = F.relu(self.lin0(h))
        h = out.unsqueeze(0)

        for i in range(self.num_step_message_passing):
            m = F.relu(self.conv(g, out, g.edata['e_feat']))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        out = self.set2set(g, out)
        out = F.relu(self.lin1(out))
        out = self.lin2(out)
        return out
