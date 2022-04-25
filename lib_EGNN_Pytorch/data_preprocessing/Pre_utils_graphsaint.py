
import numpy as np
import os
import yaml
import scipy.sparse as sp
import json

import numpy as np
from sklearn.preprocessing import StandardScaler
import metis

import torch




def train_setting(dataname, datapath, train_config_file):
    """
        The function yaml.load converts a YAML document to a Python object.
    """
    with open(train_config_file) as f_train_config:
        train_config = yaml.load(f_train_config)
        
    arch_gcn = {'dim':-1,
                'aggr':'concat',
                'loss':'softmax',
                'arch':'1',
                'act':'I',
                'bias':'norm'}
    # check the loss:  default to be softmax, multi-class problem, each node can only belong to just one class at last
    arch_gcn.update(train_config['network'][0])   # train_config['network'] is a list of dict
    
    
    train_params = {'lr' : 0.01, 'weight_decay' : 0., 'norm_loss':True, 'norm_aggr':True, 'q_threshold' : 50, 'q_offset':0}
    train_params.update(train_config['params'][0])
    train_phases = train_config['phase']
    for ph in train_phases:
        assert 'end' in ph
        assert 'sampler' in ph
    print("Loading training data..")
    temp_data = load_data_graphsaint(dataname, datapath = datapath)
    train_data = process_graph_data(*temp_data)
    print("Done loading training data..")
    
    # train_data is a tuple: adj_full, adj_train, feats, class_arr, role
    return train_params, train_phases, train_data, arch_gcn


"""Collections of partitioning functions."""
def partition_graph(adj_full, target_nodes, num_clusters):
    """partition a graph by METIS into smaller mini-clusters
        Later these mini-clusters will be re-orginaized/combined into larger batches
    Input:
        adj_full (sp.csr_matrix): full adjacent matrix of the whole graph 
        target_nodes (np.array): to-be partitioned target nodes, usually the train_nodes
    """
    num_nodes = len(target_nodes)     # just the to-be partitioned target nodes
    num_all_nodes = adj_full.shape[0]   # all nodes in the graph
    
    neighbor_intervals = []
    neighbors = []
    edge_cnt = 0
    neighbor_intervals.append(0)
    train_adj_lil = adj_full[target_nodes, :][:, target_nodes].tolil()
    train_ord_map = dict()
    train_adj_lists = [[] for _ in range(num_nodes)]
    for i in range(num_nodes):
        rows = train_adj_lil[i].rows[0]
        # self-edge needs to be removed for valid format of METIS
        if i in rows:
            rows.remove(i)
        train_adj_lists[i] = rows
        neighbors += rows
        edge_cnt += len(rows)
        neighbor_intervals.append(edge_cnt)
        train_ord_map[target_nodes[i]] = i
        
    if num_clusters > 1:
        _, groups = metis.part_graph(train_adj_lists, num_clusters, seed=1)
    else:
        groups = [0] * num_nodes
        
    part_row = []
    part_col = []
    part_data = []
    parts = [[] for _ in range(num_clusters)]
    for nd_idx in range(num_nodes):
        gp_idx = groups[nd_idx]
        nd_orig_idx = target_nodes[nd_idx]
        # add nodes to each group inside the parts
        parts[gp_idx].append(nd_orig_idx)
    
        for nb_orig_idx in adj_full[nd_orig_idx].indices:
            nb_idx = train_ord_map[nb_orig_idx]
            if groups[nb_idx] == gp_idx:
                part_data.append(1)
                part_row.append(nd_orig_idx)
                part_col.append(nb_orig_idx)
    part_data.append(0)
    part_row.append(num_all_nodes - 1)
    part_col.append(num_all_nodes - 1)
    part_adj = sp.coo_matrix((part_data, (part_row, part_col))).tocsr()
    
    # parts: is the divided groups of nodes
    # part_adj: is the csr_matrix for the adjacency matrix of the whole graph
    # here this part_adj is different since some of the inter-cluster edges are lost
    return part_adj, parts


def load_data_graphsaint(prefix, datapath = './', normalize=True):
    """
        prefix: should be the dataname: flickr, PPI_small, Reddit, Yelp, PPI_large
        datapath: location for all dataset files
    """
    # Load a sparse matrix from a file using .npz format. Return csc_matrix, csr_matrix, bsr_matrix, dia_matrix or coo_matrix
    adj_full = sp.load_npz(datapath + '{}/raw/adj_full.npz'.format(prefix)).astype(np.bool)
    adj_train = sp.load_npz(datapath + '{}/raw/adj_train.npz'.format(prefix)).astype(np.bool)
    
    role = json.load(open(datapath + '{}/raw/role.json'.format(prefix)))
    
    """
        .npy:  the standard binary file format in NumPy for persisting a single arbitrary NumPy array on disk.
        .npz:  simple way to combine multiple arrays into a single file, one can use ZipFile to contain multiple “.npy” files

        .npz is just a ZipFile containing multiple “.npy” files. 
        And this ZipFile can be either compressed (by using np.savez_compressed) or uncompressed (by using np.savez)
    """
    # Load arrays or pickled objects from .npy, .npz or pickled files.
    feats = np.load(datapath + '{}/raw/feats.npy'.format(prefix))
    """
        json.load() method (without “s” in “load”) used to read JSON encoded data from a file and convert it into Python dictionary.
        json.loads() method, which is used for parse valid JSON String into Python dictionary
    """
    class_map = json.load(open(datapath + '{}/raw/class_map.json'.format(prefix)))
    class_map = {int(k):v for k, v in class_map.items()}
    assert len(class_map) == feats.shape[0]
    # ---- normalize feats ----
    # scipy.sparse.csr_matrix.nonzero:  Returns a tuple of arrays (row,col) containing the indices of the non-zero elements of the matrix.
    train_nodes = np.array(list(set(adj_train.nonzero()[0])))
    train_feats = feats[train_nodes]
    scaler = StandardScaler()
    scaler.fit(train_feats)
    # transform the whole feature by fitting the train features 
    feats = scaler.transform(feats)
    # -------------------------
    # adj_full, adj_train: csr_matrix ; feats: np.array;  class_map, role : python dict
    
    return adj_full, adj_train, feats, class_map, role

def process_graph_data(adj_full, adj_train, feats, class_map, role):
    """
    setup vertex property map for output classes, train/val/test masks, and feats
    Input:
        adj_full, adj_train: csr_matrix
        feats: np.array
        class_map  : python dict;  value can be a list (multi-label task), or can be a value: (multi-class task)
        role : python dict
    Output:
        Return: mainly get the class array as np.array
        
    """
    num_vertices = adj_full.shape[0]
    # check whether it is multi-class or multi-label task
    if isinstance(list(class_map.values())[0], list):   # one node belongs to multi-labels
        num_classes = len(list(class_map.values())[0])
        class_arr = np.zeros((num_vertices, num_classes))
        for k, v in class_map.items():
            class_arr[k] = v            # assign a list directly to a row of numpy array
    else:
        num_classes = max(class_map.values()) - min(class_map.values()) + 1   # assume all the classes are continuous
        class_arr = np.zeros((num_vertices, num_classes))
        # if multi-class task: then shift the class label value, starting from 0
        offset = min(class_map.values())
        for k, v in class_map.items():
            class_arr[k][v-offset] = 1
    return adj_full, adj_train, feats, class_arr, role



def adj_norm(adj, deg = None, sort_indices = True):
    """
    Normalize adj according to two methods: symmetric normalization and rw normalization.
    sym norm is used in the original GCN paper (kipf)
    rw norm is used in graphsage and some other variants.

    # Procedure: 
    #       1. adj add self-connection --> adj'
    #       2. D' deg matrix from adj'
    #       3. norm by D^{-1} x adj'
    if sort_indices is True, we re-sort the indices of the returned adj
    Note that after 'dot' the indices of a node would be in descending order rather than ascending order
    """
    diag_shape = (adj.shape[0], adj.shape[1])
    D = adj.sum(1).flatten() if deg is None else deg
    norm_diag = sp.dia_matrix((1/D,0),shape=diag_shape)  # offset is 0
    adj_norm = norm_diag.dot(adj)
    if sort_indices:
        adj_norm.sort_indices()
    return adj_norm


def adj_norm_diag_enhance(adj, deg = None, sort_indices = True, diag_lambda = -1):
    """
    Normalization by  A'=(D+I)^{-1}(A+I), A'=A'+lambda*diag(A')

    # Procedure: 
    #       1. adj add self-connection --> adj'
    #       2. D' deg matrix from adj'
    #       3. norm by D^{-1} x adj'
    if sort_indices is True, we re-sort the indices of the returned adj
    Note that after 'dot' the indices of a node would be in descending order rather than ascending order
    """
    adj = adj + sp.eye(adj.shape[0])
    diag_shape = (adj.shape[0], adj.shape[1])
    D = adj.sum(1).flatten() if deg is None else deg
    norm_diag = sp.dia_matrix((1/D,0),shape=diag_shape)  # offset is 0
    adj_norm = norm_diag.dot(adj)
    if diag_lambda != -1:
        adj_norm = adj_norm + diag_lambda * sp.diags(adj_norm.diagonal(), 0)

    if sort_indices:
        adj_norm.sort_indices()
    return adj_norm



def _coo_scipy2torch(adj):
    """
    convert a scipy sparse COO matrix to torch
    
    Torch supports sparse tensors in COO(rdinate) format, which can efficiently store and process tensors 
    for which the majority of elements are zeros.
    A sparse tensor is represented as a pair of dense tensors: a tensor of values and a 2D tensor of indices. 
    A sparse tensor can be constructed by providing these two tensors, as well as the size of the sparse tensor 
    """
    values = adj.data
    indices = np.vstack((adj.row, adj.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    
    return torch.sparse.FloatTensor(i,v, torch.Size(adj.shape))