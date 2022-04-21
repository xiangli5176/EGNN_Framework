from __future__ import print_function, division

from collections import defaultdict
import logging
import os
import time
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from ...models.model_app import LogReg
from ...evaluation import eva_node_classification
from ...dataloader import LoadDataset_feat
from ...training_algo.contrastive import train_step_multi_view



def train(model_emb, config, inputs, device = "cpu", checkpoint_file_path = None):
    """ 
    Main procedure to perform train process: 
        1) pre-computation/training for embedding
        2) regression train for supervised classification task
    Args:
        model_train ([type]): [description]
        config (dict): [description]
        inputs (list(numpy.array)): features, labels_full, idx_train, idx_val, idx_test
        device (str, optional): [description]. Defaults to "cpu".
        checkpoint_file_path ([type], optional): [description]. Defaults to None.
    """

    features, labels_full, idx_train, idx_val, idx_test = inputs
    
    # this features is actually the precomputed embedding
    emb, emb_precompute_time = RwCL_emb_precompute(model_emb, config, features, device = device) 
    reg_inputs = [emb, emb_precompute_time, labels_full, idx_train, idx_val, idx_test]

    return train_classification(config, reg_inputs, device = device, checkpoint_file_path = checkpoint_file_path, 
            metric_list = ["f1_micro", "accuracy", "f1_macro"])


def RwCL_emb_precompute(model_train, config, feature,  device = "cpu"):
    """ Main training process in mini-batches

    Args:
        model_train ([type]): DNN module with contrastive objective
        config (dict): config of settings
        feature(list(numpy.array)): smoothed features torch.Tensor(torch.float32s)
        device (str, optional): which device to use. Defaults to "cpu".

    Returns:
        tuple(dict): training statistics
    """
    feature = torch.tensor(feature, dtype = torch.float32).to(device)
    model_train = model_train.to(device)
    

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

            epoch += 1
            if epoch > config["num_epochs"]:
                break
            
    with torch.no_grad():
        model_train.eval()
        emb = model_train(feature, train=False)
        # emb = model_train(feature, train=False).cpu().data.numpy()

    # u, s, v = sp.linalg.svds(emb, k=16, which='LM')
    # u = skp.normalize(emb.dot(v.T))  # u is the embedding after PCA to perform clustering

    return emb, train_time   # make sure the output embedding is still a torch.tensor on GPU


def train_classification(config, inputs, 
            device = "cpu", checkpoint_file_path = None, 
            metric_list = ["f1_micro", "accuracy", "f1_macro"],
        ):
    """ Perform the training process
    
    """
    emb, precompute_time, labels_full, idx_train, idx_val, idx_test = inputs
    labels_full = torch.tensor(labels_full, dtype = torch.int64).to(device)
    
    metric_summary = defaultdict(dict)

    model_train = LogReg(emb.shape[1], config['n_class'])
    model_train = model_train.to(device)
    optimizer = torch.optim.Adam(model_train.parameters(), lr=config['reg_lr'], weight_decay = config['reg_weight_decay'])

    train_time = precompute_time

    best_f1_micro_val = 0.0
    # best_f1_micro_val = torch.zeros((1))
    # best_loss_val = 100.
    best_model = None
    for epoch in range(config["reg_num_epochs"]):
        
        tmp_time = time.time()
        model_train.train()
        optimizer.zero_grad()
        output = model_train(emb[idx_train])
        loss_train = F.cross_entropy(output, labels_full[idx_train])
        loss_train.backward()
        optimizer.step()
        train_time += time.time() - tmp_time

        if epoch % config["reg_eval_step"] == 0:
            with torch.no_grad():
                model_train.eval()
                output = model_train(emb[idx_val])

                metric_summary["loss_hist"][(train_time, epoch)] = loss_train.item()
                metric_res = eva_node_classification(output.cpu().numpy(), labels_full[idx_val].cpu().numpy())
                for metric_name in metric_list:
                    metric_summary[f'{metric_name.lower()}_hist'][(train_time, epoch)] = metric_res[metric_name]
                
                f1_micro_val = metric_res["f1_micro"]

                # loss_val = F.cross_entropy(output, labels_full[idx_val])
                if config["reg_early_stop"] and best_f1_micro_val < f1_micro_val:
                    best_f1_micro_val = f1_micro_val
                    best_model = model_train
                # if best_loss_val > loss_val:
                #     best_loss_val = loss_val
                #     best_model = model_train(emb)

    if checkpoint_file_path is not None:
        emb_path = os.path.join(os.path.dirname(checkpoint_file_path), "emb.npy")
        np.save(emb_path, emb.cpu().numpy())
        if best_model is None:
            torch.save(model_train.state_dict(), checkpoint_file_path)
        else:
            torch.save(best_model.state_dict(), checkpoint_file_path)
    return train_time, metric_summary    


def test(config, inputs, 
        device = "cpu", checkpoint_file_path = None):
    """
    Evaluate the trained model on the test data
    Args:
        model_template : blank dmon model
        inputs: [y, adj_raw], 
                y: numpy.array , golden label for classes
                adj_raw: sp.sparse_csr.csr_matrix, raw adjacency matrix of the graph
        checkpoint_file_path : the path for the file of the saved model checkpoint
    """
    labels_full, idx_test = inputs
    emb_path = os.path.join(os.path.dirname(checkpoint_file_path), "emb.npy")
    emb = np.load(emb_path)
    emb = torch.Tensor(emb).to(device)

    model_template = LogReg(emb.shape[1], config['n_class'])
    model_template.load_state_dict(torch.load(checkpoint_file_path, map_location = device)) 
    model_template.eval()

    start = time.time()
    with torch.no_grad():
        output = model_template(emb[idx_test])
    test_metric = eva_node_classification(output.detach().numpy(), labels_full[idx_test])

    test_time = time.time() - start
    
    metric_display_info = " | ".join([f"{key} : {val}" for key, val in test_metric.items()])
    logging.info(f"Test metrics: | {metric_display_info} | test time: {test_time:.3f}s")
    return test_time, test_metric 



