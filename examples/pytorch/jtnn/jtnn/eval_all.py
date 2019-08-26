import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

import math
import random
import sys
from optparse import OptionParser
from collections import deque
import rdkit
from pathlib import Path

from jtnn import *
d = Path("./results2/")


parser = OptionParser()
parser.add_option("-t", "--train", dest="train",
                  default='test', help='Training file name')
parser.add_option("-v", "--vocab", dest="vocab",
                  default='vocab', help='Vocab file name')
parser.add_option("-s", "--save_dir", dest="save_path")
parser.add_option("-m", "--model", dest="model_path",
                  default='/home/ubuntu/playground/dgl/examples/pytorch/jtnn/remote/results3/model.iter-2-4500')
parser.add_option("-b", "--batch", dest="batch_size", default=40)
parser.add_option("-w", "--hidden", dest="hidden_size", default=450)
parser.add_option("-l", "--latent", dest="latent_size", default=56)
parser.add_option("-d", "--depth", dest="depth", default=3)
parser.add_option("-z", "--beta", dest="beta", default=1.0)
parser.add_option("-q", "--lr", dest="lr", default=1e-3)
parser.add_option("-T", "--test", dest="test", action="store_true")
opts, args = parser.parse_args()

dataset = JTNNDataset(data=opts.train, vocab=opts.vocab, training=False)
vocab = dataset.vocab

batch_size = int(opts.batch_size)
hidden_size = int(opts.hidden_size)
latent_size = int(opts.latent_size)
depth = int(opts.depth)
beta = float(opts.beta)
lr = float(opts.lr)

model = DGLJTNNVAE(vocab, hidden_size, latent_size, depth)


def worker_init_fn(id_):
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)


worker_init_fn(None)


def reconstruct():
    dataset.training = False
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        collate_fn=JTNNCollator(vocab, False),
        drop_last=True,
        worker_init_fn=worker_init_fn)

    # Just an example of molecule decoding; in reality you may want to sample
    # tree and molecule vectors.
    acc = 0.0
    tot = 0
    print(len(dataset))
    for it, batch in enumerate(dataloader):
        gt_smiles = batch['mol_trees'][0].smiles
        print(gt_smiles)
        model.move_to_cuda(batch)
        _, tree_vec, mol_vec = model.encode(batch)

        tree_mean = model.T_mean(tree_vec)
        # Following Mueller et al.
        tree_log_var = -torch.abs(model.T_var(tree_vec))
        mol_mean = model.G_mean(mol_vec)
        # Following Mueller et al.
        mol_log_var = -torch.abs(model.G_var(mol_vec))

        epsilon = create_var(torch.randn(
            1, model.latent_size // 2), False).cuda()
        tree_vec = tree_mean + torch.exp(tree_log_var // 2) * epsilon
        epsilon = create_var(torch.randn(
            1, model.latent_size // 2), False).cuda()
        mol_vec = mol_mean + torch.exp(mol_log_var // 2) * epsilon
        dec_smiles = model.decode(tree_vec, mol_vec)
        print("Dec smiles")
        print(dec_smiles)

        if dec_smiles == gt_smiles:
            acc += 1
        tot += 1
        print(acc / tot)
    return acc/tot


record = {}
for f in d.glob("*"):
    print(f)
    filename = str(f)
    if int(filename.split('-')[1]) <= 5:
        continue
    model.load_state_dict(torch.load(str(f)))
    model = cuda(model)
    model.eval()
    print("Model #Params: %dK" %
          (sum([x.nelement() for x in model.parameters()]) / 1000,))
    record[str(f)] = reconstruct()
    with open("./eval.pkl", "wb") as f:
        pickle.dump(record, f)
