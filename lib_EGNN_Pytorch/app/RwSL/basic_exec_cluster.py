from __future__ import print_function, division

from collections import defaultdict
import logging
import os
import csv
import time
from sklearn.cluster import KMeans

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from ...evaluation import eva
from ...dataloader import LoadDataset_feat, LoadDataset_feat_prob
from ... import utils



def train(model_train, config, inputs, 
            device = "cpu", checkpoint_file_path = None):
    """
    Args:
        model_train: established model to be trained
        config: configuration dict for the hyper-parameter settings
        inputs: [feature, y, adj], 
                feature: numpy.array, node attributes
                y: numpy.array , golden label for classes
                adj : pytorch sparse tensor
        checkpoint_file_path : keep snapshot of the model with the best pairwise f1 score for clustering
    """
    loss_hist = {}
    f1_macro_hist = defaultdict(dict); f1_micro_hist = defaultdict(dict); acc_hist = defaultdict(dict)
    ari_hist = defaultdict(dict); nmi_hist = defaultdict(dict)
    conductance_hist = defaultdict(dict); modularity_hist = defaultdict(dict)
    dmon_f1_score_hist = defaultdict(dict)
    
    def record_metric(label, pred, adj,
                        epoch, train_time, 
                      name = 'Q', display = False):
        metric = eva(label, pred, adj)
        f1_micro_hist[name][(train_time, epoch)] = metric["f1_micro"]
        f1_macro_hist[name][(train_time, epoch)] = metric["f1_macro"]
        acc_hist[name][(train_time, epoch)] = metric["Accuracy"]
        nmi_hist[name][(train_time, epoch)] = metric["NMI"]
        ari_hist[name][(train_time, epoch)] = metric["ARI"]

        conductance_hist[name][(train_time, epoch)] = metric["conductance"]
        modularity_hist[name][(train_time, epoch)] = metric["modularity"]
        dmon_f1_score_hist[name][(train_time, epoch)] = metric["dmon_f1_score"]

        if display:
            logging.info(f"{name} distribtion: Epoch {epoch:4d} | " + ' | '.join(f"{key} : {val:.4f}" for key, val in metric.items()))
        

        return metric["Accuracy"] + metric["NMI"] + metric["ARI"] + metric["modularity"] - metric["conductance"]

    # KNN Graph
    feature, y, adj = inputs
    
    # cluster parameter initiate
    n_nodes, n_features = feature.shape
    data = torch.Tensor(feature)

    with torch.no_grad():
        _, z, _ = model_train.ae(data)

    kmeans = KMeans(n_clusters=config["n_clusters"], n_init=20)
    # n_init: Number of time the k-means algorithm will be run with different centroid seeds.
    
    # z is of shape (N, n_z)
    y_pred = kmeans.fit_predict(z.data.numpy())
    # y_pred of shape (n_samples)
    # model_train.cluster_layer param is of shape: (n_clusters, n_z)
    
    # data = data.to(device)
    model_train = model_train.to(device)
    model_train.train()
    model_train.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    # cluster_centers_ndarray of shape (n_clusters, n_features)
    # Coordinates of clusteva(y, y_pred)er centers.

    optimizer = AdamW(model_train.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    # optimizer = Adam(model_train.parameters(), lr=config["lr"])

    p_update_loader = DataLoader(LoadDataset_feat(data, device = device), batch_size=config["batch_size_update_p"], shuffle=False)
    train_time = 0

    for epoch in range(1, config["n_epochs"]+1):
        # must start from 0: since we need an initial p at the very begining
        if (epoch-1) % config["update_p"] == 0:
        # update_interval
            update_p_start = time.time()

            # suppose the data is of large dimension, dense features, preform this by batches
            tmp_q =  torch.zeros([n_nodes, config["n_clusters"]], dtype=torch.float32).to(device)
            pred =  torch.zeros([n_nodes, config["n_clusters"]], dtype=torch.float32).to(device)

            for batch_idx, (batch_node_ids, batch_x) in enumerate(p_update_loader):
                _, batch_q, batch_pred, _ = model_train(batch_x)
                tmp_q[batch_node_ids] += batch_q
                pred[batch_node_ids] += batch_pred

            tmp_q = tmp_q.data
            p = target_distribution(tmp_q)
            train_time += time.time() - update_p_start
        
        calc_loss_start = time.time()
        # both q and pred are of shape: (n_sample, n_cluster)
        # possible to make these below in a mini-batch way:
        train_loader = DataLoader(LoadDataset_feat_prob(data, p, device = device), batch_size=config["batch_size_train"], shuffle=True)
        for batch_idx, (batch_node_ids, batch_x, batch_p) in enumerate(train_loader):
            x_bar_batch, q_batch, pred_batch, _ = model_train(batch_x)

            kl_loss = F.kl_div(q_batch.log(), batch_p, reduction='batchmean')
            ce_loss = F.kl_div(pred_batch.log(), batch_p, reduction='batchmean')
            re_loss = F.mse_loss(x_bar_batch, batch_x)

            loss = config["a"] * kl_loss + config["b"] * ce_loss + re_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_time += time.time() - calc_loss_start
        
        # evaluate the metrics on the CPU side, only need two matrix: tmp_q and pred
        if epoch % config["eval_step"] == 0:
            q_res = tmp_q.data.cpu().numpy().argmax(1)       #Q
            z_res = pred.data.cpu().numpy().argmax(1)   #Z
            p_res = p.data.cpu().numpy().argmax(1)      #P

            loss_hist[(train_time, epoch)] = loss.item()
            record_metric(y, q_res, adj, epoch, train_time, name = 'Q')
            record_metric(y, p_res, adj, epoch, train_time, name = 'P')
            test_metric_val = record_metric(y, z_res, adj, epoch, train_time, name = 'Z')

        if epoch % config["display_eval_step"] == 0:
            logging.info(f"Epoch {epoch:4d} | total train loss: {loss:.3f} | train time: {train_time:.3f}s")
    
    if checkpoint_file_path is not None:
        torch.save(model_train.state_dict(), checkpoint_file_path)

        # keep a record of the best epoch for test
        target_file = os.path.join(os.path.dirname(checkpoint_file_path), "test_epoch.csv")
        with open(target_file, 'w', newline='\n') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            header = ["best_epoch_num", epoch, train_time]
            wr.writerow(header)
    
    return train_time, loss_hist, f1_micro_hist, f1_macro_hist, acc_hist, ari_hist, nmi_hist, \
                conductance_hist, modularity_hist, dmon_f1_score_hist


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
    model_template.load_state_dict(torch.load(checkpoint_file_path, map_location='cpu')) 
    model_template.eval()
    feature, y, adj = inputs
    data = torch.Tensor(feature)
    
    start = time.time()
    
    with torch.no_grad():
        _, q, pred, _ = model_template(data)
    z_res = pred.data.cpu().numpy().argmax(1)   #Z
    test_time = time.time() - start
    
    test_metric = eva(y, z_res, adj)  # only care about Z 
    
    metric_display_info = " | ".join([f"{key} : {val}" for key, val in test_metric.items()])
    logging.info(f"Test metrics: | {metric_display_info} | test time: {test_time:.3f}s")
    return test_time, test_metric    




def pretrain_ae(model_pretrain, config, feature,
                device = "cpu", pretrain_save_path = None):
    """
    Args:
        model_pretrain: established AE model to be trained
        config: configuration dict for the hyper-parameter settings
        dataset: 
                feature : is the raw feature
        pretrain_save_path : Save the pre-trained AE model parameter to this Path
    """
    
    loss_hist = {}

    feature = torch.from_numpy(feature).float()
    
    pre_train_loader = DataLoader(LoadDataset_feat(feature, device = device), batch_size=config["batch_size_pre_train"], shuffle=True)
#     print(model_pretrain)

    model_pretrain = model_pretrain.to(device)
    model_pretrain.train()
    # optimizer = Adam(model_pretrain.parameters(), lr=config["pretrain_lr"])
    optimizer = AdamW(model_pretrain.parameters(), lr=config["lr"], weight_decay=config["pretrain_weight_decay"])
    
    train_time = 0
    for epoch in range(1, config["pretrain_n_epochs"]+1):
        calc_loss_start = time.time()
        for batch_idx, (_, x) in enumerate(pre_train_loader):
            x_bar, _, _ = model_pretrain(x)
            loss = F.mse_loss(x_bar, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_time += time.time() - calc_loss_start

        with torch.no_grad():
            x = feature.to(device)
            x_bar, _, _ = model_pretrain(x)
            loss = F.mse_loss(x_bar, x)
            print('{} loss: {}'.format(epoch, loss))
            loss_hist[(train_time, epoch)] = loss.item()
            
    if pretrain_save_path:
        target_folder = os.path.dirname(pretrain_save_path)
        # os.makedirs(target_folder, exist_ok=True)
        utils.save_info_pickle(data_path = os.path.join(target_folder, "pretrain_loss.pkl"), 
                               target_data = loss_hist, exist_ok = False)
        torch.save(model_pretrain.state_dict(), pretrain_save_path)
        
    return train_time, loss_hist



def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()