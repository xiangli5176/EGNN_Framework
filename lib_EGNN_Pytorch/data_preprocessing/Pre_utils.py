import pickle
import shutil
import numpy as np
import os
import sys
import scipy.sparse as sp
import networkx as nx
from itertools import chain
import json
import sklearn.preprocessing as skp

import torch
from torch.utils.data import Dataset


### =======================   General data preprocessing for RwCL and RwSL ===========================
def aug_normalized_adjacency(adj, aug_val = 1.0):
    """ Borrowed from SSGC
        Normalize the adjacency matrix by:
        D(-1/2) (A + aug_val*I) D(-1/2)
    Args:
        adj (sp.sparse.csr_matrix): adajcency matrix, without self-loops
        aug_val (float, optional):  . Defaults to 1.0.
    Returns:
        sp.sparse.csr_matrix: the normalized adjacency matrix
    """
    adj = adj + sp.eye(adj.shape[0]) * aug_val
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    """ Check if the input numpy array (a) is symmetric
        rtol (float): The relative tolerance parameter (see Notes).
        atol (float): The absolute tolerance parameter (see Notes).
    """
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def row_normalize(mx):
    """
        Row-normalize matrix, so far only make sense for kipf datasets as an extra trick for data preprocessing as in their GCN paper 2017
        input should be an numpy array, or sparse.csr_matrix or scipy.sparse.lil.lil_matrix'
    """
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx        
    

def split_input_classification(data_name, features, node_split_file_path = None, redo_save = False, 
                                        val_frac = 0.05, test_frac = 0.1):
    """
    Train, validation, test splitting according to the specified fraction
    Args:
        data_name (string): name of the dataset, all lowercase
        features (np.array): node attributes
        node_split_file_path (string, optional): file path to store the data splitting with indices stored in np.array
        redo_save (bool, optional): whether or not we need to update the saved adj files. Defaults to False.

    Returns:
        node indices for : train, validation and test
    """
    n_node = features.shape[0]
    all_node_idx = np.array(range(n_node), dtype = np.int64)
    np.random.shuffle(all_node_idx)

    num_test = int(np.floor(n_node * test_frac))
    num_val = int(np.floor(n_node * val_frac))

    val_node_idx = all_node_idx[:num_val]
    test_node_idx = all_node_idx[num_val:(num_val + num_test)]
    train_node_idx = all_node_idx[(num_val + num_test):]

    if node_split_file_path is not None and os.path.exists(node_split_file_path):
        if redo_save:
            print(f"Regenerate the node splitting at : {node_split_file_path}")
            np.savez(node_split_file_path, val_node_idx = val_node_idx, 
                    test_node_idx = test_node_idx, train_node_idx = train_node_idx) 
        else:
            print(f"Loading the pre-existent node splitting at : {node_split_file_path}")
            with np.load(open(node_split_file_path, 'rb'), allow_pickle=True) as data_input:
                val_node_idx = data_input['val_node_idx']
                test_node_idx = data_input['test_node_idx']
                train_node_idx = data_input['train_node_idx']
    else:
        print(f"Produce a brand new node splitting at : {node_split_file_path}")
        os.makedirs(os.path.dirname(node_split_file_path), exist_ok=True)
        np.savez(node_split_file_path, val_node_idx = val_node_idx, 
                    test_node_idx = test_node_idx, train_node_idx = train_node_idx) 
        
    return train_node_idx, val_node_idx, test_node_idx


### =================== Generate the edges splitting for the link prediction ==================
def mask_test_edges(adj, val_frac = 0.05, test_frac = 0.1):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    # adj.eliminate_zeros()

    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)   # upper triangular portiaon of the matrix 
    adj_tuple = sparse_to_tuple(adj_triu)   # get the edge coordiates with coordinates

    edges = adj_tuple[0]  # these edges are single-direction
    edges_all = sparse_to_tuple(adj)[0]   # all edges , double direction
    num_test = int(np.floor(edges.shape[0] * test_frac))
    num_val = int(np.floor(edges.shape[0] * val_frac))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]

    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    # construct train edge coordinates by removing val and test,  shape (n_train, 2)
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        """  numpy.round_(a, decimals=0, out=None)[source], Round an array to the given number of decimals. """
        rows_close = np.all( np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        # generating a tuple of randomized coordinates, i.e. fake edges
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    #assert ~ismember(test_edges_false, edges_all)
    #assert ~ismember(val_edges_false, edges_all)
    #assert ~ismember(val_edges, train_edges)
    #assert ~ismember(test_edges, train_edges)
    #assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])
    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return ( adj_train, train_edges, np.array(val_edges, dtype = np.int64), np.array(val_edges_false, dtype = np.int64), 
                  np.array(test_edges, dtype = np.int64), np.array(test_edges_false, dtype = np.int64) )


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape




### ========================== SDCN input preprocessing ===============================
def load_sdcn_graph(data_folder, data_name, k = None, add_self_loop = False):
    """
        Whether elemeint zeros after removing the self-loops , Potential issue with DBLP
        Return:
            if normalize_tensor = True: return sparse pytorch tensor
            else: return scipy.sparse.csr.csr_matrix 
    """
    if k:
        path = os.path.join(data_folder, f'graph/{data_name}{k}_graph.txt')
    else:
        path = os.path.join(data_folder, f'graph/{data_name}_graph.txt')
        
    data = np.loadtxt(os.path.join(data_folder, f'data/{data_name}.txt'))
    n, _ = data.shape
#     print(data.shape, data.dtype)
    
    idx = np.array([i for i in range(n)], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    
    edges_unordered = np.genfromtxt(path, dtype=np.int32)
#     print(edges_unordered.shape, edges_unordered.dtype)
    
    edges = np.array([idx_map.get(i) for i in edges_unordered.flatten()],
                     dtype=np.int32).reshape(edges_unordered.shape)
    
    ### actually edges and unedges are the same, did not see the necessity of this procedure

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(n, n), dtype=np.float32) # adj : scipy.sparse.coo.coo_matrix

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)  # adj becomes: scipy.sparse.csr.csr_matrix 
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape = adj.shape)
    if add_self_loop:
        adj = adj + sp.eye(adj.shape[0]) 
    # adj.eliminate_zeros()

    # adj = adj + sp.eye(adj.shape[0])    # this sp.eye will change the dtype from np.float32 into np.float64
    # normalized_adj = normalize_sdcn(adj)
    # normalized_adj = sparse_mx_to_torch_sparse_tensor(normalized_adj)
    return adj

def normalize_sdcn(mx):
    """
        Row-normalize sparse matrix : D(-1/2)AD(-1/2)
        since this is undirected graph, A is symmetric
        Reduced to : D(-1)A
    """
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

class load_sdcn_data(Dataset):
    def __init__(self, data_folder, data_name):
        self.x = np.loadtxt(os.path.join(data_folder, f'data/{data_name}.txt'), dtype=np.float32)
        self.y = np.loadtxt(os.path.join(data_folder, f'data/{data_name}_label.txt'), dtype=np.int64)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])),\
               torch.from_numpy(np.array(self.y[idx])),\
               torch.from_numpy(np.array(idx))

def load_sdcn_data_func(data_folder, data_name):
    feature = np.loadtxt(os.path.join(data_folder, f'data/{data_name}.txt'), dtype=np.float32)
    label = np.loadtxt(os.path.join(data_folder, f'data/{data_name}_label.txt'), dtype=np.int64)

    return feature, label


### ========================== GNN benchmark paper shchur input preprocessing ===============================
def load_shchur_gnnbenchmark_data(data_path, data_name, standardize=True, add_self_loop = False):
    """Loads an attributed graph with sparse features from a specified Numpy file.

    Args:
        data_path (string) : a folder containing all the needed .npz datasets
        file_name (string) : A valid file name of a .npz file containing the input data.

    Returns:
        A tuple (graph, features, labels, label_indices) with the sparse adjacency
        matrix of a graph, sparse feature matrix, dense label array, and dense label
        index array (indices of nodes that have the labels in the label array).
    """
    
    with np.load(open(os.path.join(data_path, f"{data_name}.npz"), 'rb'), allow_pickle=True) as loader:
#         If the file is a .npz file, then a dictionary-like object is returned, containing {file_name: array} key-value pairs, one for each file in the archive.
#         If the file is a .npz file, the returned value supports the context manager protocol in a similar fashion to the open function:
        
        # allow_pickle:  Allow loading pickled object arrays stored in npy files.
        # loader is of type:  <class 'numpy.lib.npyio.NpzFile'>
        loader = dict(loader)
        print(loader.keys())
#         print(loader["class_names"])
        
        adj_full = sp.csr_matrix(
            (loader['adj_data'], loader['adj_indices'], loader['adj_indptr']),
            shape=loader['adj_shape'])

        features = sp.csr_matrix(
            (loader['attr_data'], loader['attr_indices'],
             loader['attr_indptr']),
            shape=loader['attr_shape'])

#         label_indices = loader['label_indices']
        full_labels = loader['labels']
        
    assert adj_full.shape[0] == features.shape[0], 'Adjacency and feature size must be equal!'

    adj_full = adj_full - sp.dia_matrix((adj_full.diagonal()[np.newaxis, :], [0]), shape=adj_full.shape)
    if add_self_loop:
        adj_full = adj_full + sp.eye(adj_full.shape[0]) 
    # adj_full.eliminate_zeros()

    features = features.toarray()

    if standardize:
        scaler = skp.StandardScaler()
        scaler.fit(features)
        features = scaler.transform(features)
#     assert labels.shape[0] == label_indices.shape[0], 'Labels and label_indices size must be equal!'
    print(f"Dtype of features is : {features.dtype}")

    return adj_full, features, full_labels


### ========================== kipf input preprocessing ===============================
def load_gcn_tkipf_data(data_path, dataset_str, normalize = True, 
            redo_save = False, add_self_loop = False):
    """
    Loads input data from gcn/data directory
    From the original GCN tkipf package

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    Args:
            dataset_str: Dataset name
            data_path: ~/projects/tmpdata/GCN/Graph_Clustering/tkipf_gcn_data
            normalize: if True, perform the row-normalized as in the tkipf gcn preprocessing
            redo_save: whether to resave all the packed datasets
    :return: All data input files loaded (as well the training/test data).
    """
    row_norm_folder = "row_normalize" if normalize else "no_row_normalize"
    packed_data_path = os.path.join(data_path, "Packed_data", row_norm_folder, dataset_str.lower())

    if os.path.exists(packed_data_path):
        if redo_save:
            print("Regenerate the Packed dataset, remove the previous version ...")
            shutil.rmtree(packed_data_path)
        else:
            # packed data already exists, just load it....
            print(f"Packed data already exists at: {packed_data_path}, LOADING...")

            adj_full = sp.load_npz(os.path.join(packed_data_path, 'adj_full.npz')).astype(np.float32)
            features = np.load(os.path.join(packed_data_path, 'feats.npy'))
            labels = np.load(os.path.join(packed_data_path, 'labels_full.npy'))
            with open(os.path.join(packed_data_path, "node_index_dict.pkl"), "rb") as fp:
                node_index = pickle.load(fp)
        
            return adj_full, features, labels, node_index

    os.makedirs(packed_data_path, exist_ok=True)
    

    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(os.path.join(data_path, f"ind.{dataset_str}.{names[i]}"), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pickle.load(f, encoding='latin1'))
            else:
                objects.append(pickle.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    
    test_idx_reorder = parse_index_file(os.path.join(data_path, f"ind.{dataset_str}.test.index") )
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str.lower() == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]), dtype=np.float32)
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
#         print(f"extended tx type is : {type(tx)}")
        
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

#     features = np.vstack( (allx.todense(), tx.todense()) )
    features = sp.vstack((allx, tx)).todense()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    if normalize:
        rowsum = np.array(features.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)

        features = r_mat_inv.dot(features)
    
    adj_full = nx.adjacency_matrix(nx.from_dict_of_lists(graph)).astype(np.float32)

    # remove all the self-loops if existent
    adj_full = adj_full - sp.dia_matrix((adj_full.diagonal()[np.newaxis, :], [0]), shape=adj_full.shape)
    if add_self_loop:
        adj_full = adj_full + sp.eye(adj_full.shape[0]) 
    # adj_full.eliminate_zeros()
    
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    labels = np.argmax(labels, axis=1)

    idx_test = test_idx_range.tolist()   # after sorting
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)
    
    label_indices = np.array( list(chain(idx_train, idx_val, idx_test) ), dtype=np.int64)

    node_index = {"train_index" : np.array(idx_train, dtype=np.int64), 
                  "val_index" : np.array(idx_val, dtype=np.int64), 
                  "test_index" : np.array(idx_test, dtype=np.int64), 
                  "label_indices" : label_indices}

    print(f"Start saving the packed datasets at: {packed_data_path}")
    sp.save_npz(os.path.join(packed_data_path, 'adj_full.npz'), adj_full)
    np.save(os.path.join(packed_data_path, 'feats.npy'), features)
    np.save(os.path.join(packed_data_path, 'labels_full.npy'), labels)
    with open(os.path.join(packed_data_path, "node_index_dict.pkl"), "wb") as fp:
        pickle.dump(node_index, fp)
    
    return adj_full, features, labels, node_index


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


### ========================== GraphSaint input preprocessing for RwCL ===============================
def load_graphsaint_data(prefix, datapath = './', standardize=True, add_self_loop = False):
    """
        prefix : should be the dataname: flickr, PPI_small, Reddit, Yelp, PPI_large
        datapath : location for all dataset files
        standardize : whether or not apply the standardization
    """
    
    adj_full = sp.load_npz(os.path.join(datapath, f'{prefix}/raw/adj_full.npz')).astype(np.float32)
    feats = np.load(os.path.join(datapath, f'{prefix}/raw/feats.npy')).astype(np.float32)
    labels = json.load(open(os.path.join(datapath + f'{prefix}/raw/class_map.json')) )
    labels = np.array([int(labels[key]) for key in sorted(labels.keys())], dtype = np.int64)
    
    assert len(labels) == feats.shape[0]
    adj_full = adj_full - sp.dia_matrix((adj_full.diagonal()[np.newaxis, :], [0]), shape=adj_full.shape)
    if add_self_loop:
        adj_full = adj_full + sp.eye(adj_full.shape[0]) 
    # adj_full.eliminate_zeros()

    # ---- Standardize feats ----
    if standardize:
        scaler = skp.StandardScaler()
        scaler.fit(feats)
        feats = scaler.transform(feats)
    # -------------------------
    return adj_full, feats, labels


### ========================== Meta framework data preprocessing ===============================

