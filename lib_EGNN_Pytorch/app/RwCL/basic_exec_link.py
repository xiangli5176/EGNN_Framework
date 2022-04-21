from __future__ import print_function, division

from collections import defaultdict
from time import perf_counter
import logging
import itertools
import numpy as np

import scipy.sparse as sp
import sklearn.preprocessing as skp

import torch
from torch.utils.data import Dataset, DataLoader

from ...evaluation import get_roc_score
from ...dataloader import LoadDataset_feat
from ...training_algo.contrastive import train_step_multi_view



def train(model_train, config, inputs, 
        device = "cpu", checkpoint_file_path = None, 
        metric_list = ["auc_score", "ap_score"],
        ):
    """ Main training process in mini-batches

    Args:
        model_train ([type]): [description]
        config (dict): config of settings
        inputs (list(numpy.array)): feature, labels_full, adj_raw
        device (str, optional): which device to use. Defaults to "cpu".
        checkpoint_file_path ([type], optional): Model state save file. Defaults to None.

    Returns:
        tuple(dict): training statistics
    """
    feature, adj_raw, val_edges, val_edges_false, test_edges, test_edges_false = inputs

    feature = torch.Tensor(feature).to(device)
    model_train = model_train.to(device)
    
    metric_summary = defaultdict(dict)

    optimizer = torch.optim.AdamW(model_train.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    train_loader = DataLoader(LoadDataset_feat(feature, device = device), batch_size=config["batch_size_train"], shuffle=True)

    if config["view_num"] == 1:
        cand_p = [config['drop_feature_rate_1']] * config["view_num"]
    else:
        if 'drop_feature_rate_2' in config:
            drop_feature_rate_start, drop_feature_rate_end = config['drop_feature_rate_1'], config['drop_feature_rate_2']
            if drop_feature_rate_start == drop_feature_rate_end:
                cand_p = [drop_feature_rate_start] * config["view_num"]
            else:
                if  drop_feature_rate_start > drop_feature_rate_end:
                    drop_feature_rate_start, drop_feature_rate_end = drop_feature_rate_end, drop_feature_rate_start
                cand_p = np.linspace(drop_feature_rate_start, drop_feature_rate_end, config["view_num"])
        else:
             cand_p = [config['drop_feature_rate_1']] * config["view_num"]
    
    ### ============= if using multi-view, without anchor view, need to ensure there are more than 1 view for comparison
    if len(cand_p) < 2:
        cand_p = cand_p * 2

    train_time = 0

    epoch = 1
    while epoch <= config["num_epochs"]:

        for batch_idx, (_, batch_x) in enumerate(train_loader, 1):
            # train_ep, loss_value = train_step(config, optimizer, model_train, batch_x)
            train_ep, loss_value = train_step_multi_view(config, optimizer, model_train, batch_x, cand_p)
            # train_ep, loss_value = train_step_multi_view_anchor_x(config, optimizer, model_train, batch_x, cand_p)
            train_time += train_ep

            if (epoch) % config["eval_display"] == 0:
                with torch.no_grad():
                    model_train.eval()
                    emb = model_train(feature, train=False).cpu().data.numpy()

                u, s, v = sp.linalg.svds(emb, k=16, which='LM')
                u = skp.normalize(emb.dot(v.T))

                # make prediction and calculate the distance:

                metric_summary["loss_hist"][(train_time, epoch)] = loss_value
                metric_res = get_roc_score(emb, adj_raw, val_edges, val_edges_false)
                for metric_name in metric_list:
                    metric_summary[f'{metric_name.lower()}_hist'][(train_time, epoch)] = metric_res[metric_name]

                logging.info(f"Epoch {epoch:4d} | total train loss: {loss_value:.3f} | Trained local batch number: {batch_idx} | train time: {train_time:.3f}s")

            epoch += 1
            if epoch > config["num_epochs"]:
                break

    if checkpoint_file_path is not None:
        torch.save(model_train.state_dict(), checkpoint_file_path)
    
    return train_time, metric_summary


def test(model_template, config, inputs, 
        device = "cpu", checkpoint_file_path = None):
    """
    Evaluate the trained model on the test data
    Args:
        model_template : blank dmon model
        inputs: [feature, y, adj], 
                feature: numpy.array, node attributes
                y: numpy.array , golden label for classes
                adj : pytorch sparse tensor
        checkpoint_file_path : the path for the file of the saved model checkpoint
    """
    feature, adj_raw, val_edges, val_edges_false, test_edges, test_edges_false = inputs
    feature = torch.Tensor(feature).to(device)

    model_template.load_state_dict(torch.load(checkpoint_file_path, map_location='cpu')) 
    
    start = perf_counter()
    with torch.no_grad():
        model_template.eval()
        emb = model_template(feature, train=False).cpu().data.numpy()

    u, s, v = sp.linalg.svds(emb, k=16, which='LM')
    u = skp.normalize(emb.dot(v.T))

    test_metric = get_roc_score(emb, adj_raw, test_edges, test_edges_false)
    test_time = perf_counter() - start
    
    
    metric_display_info = " | ".join([f"{key} : {val}" for key, val in test_metric.items()])
    logging.info(f"Test metrics: | {metric_display_info} | test time: {test_time:.3f}s")
    return test_time, test_metric    


