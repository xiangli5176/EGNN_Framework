from __future__ import print_function, division

from time import perf_counter
import itertools
import numpy as np

import torch

from ..models.model_app import drop_feature



def train_step_multi_view(config, optimizer, model_train, x, cand_p):
    """ Combinations of transformed views freely for constrastive loss"""
    start = perf_counter()

    model_train.train()
    optimizer.zero_grad()

    z_gen = (model_train(drop_feature(x, drop_p)) for drop_p in cand_p)

    loss = torch.mean(torch.stack([model_train.loss(z1, z2) for z1, z2 in itertools.combinations(z_gen, 2)]), axis = 0)
    loss.backward()
    optimizer.step()
    
    train_time = perf_counter() - start
    return train_time, loss.item()


def train_step_multi_view_anchor_x(config, optimizer, model_train, x, cand_p):
    """Use the major features as an anchor to be compared with each of the transformed views"""
    start = perf_counter()

    model_train.train()
    optimizer.zero_grad()

    z0 = model_train(x)

    z_gen = (model_train(drop_feature(x, drop_p)) for drop_p in cand_p)

    loss = torch.mean(torch.stack([model_train.loss(z0, z) for z in z_gen]), axis = 0)
    loss.backward()
    optimizer.step()
    
    train_time = perf_counter() - start
    return train_time, loss.item()


