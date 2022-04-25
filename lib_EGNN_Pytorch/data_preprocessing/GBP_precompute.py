import numpy as np
import os
import scipy.sparse as sp

from . import Pre_utils

from .gbp_cython_lib.graph import Graph   # since Base(Graph), due to this inheritance, must load this Graph base class
from .gbp_cython_lib.base import Base



### ============================ compute the gbp smoothed features =========================
def precompute_Cython_GBP_feat(data_name, num_proc, alpha, rmax, rrr, 
                                rwnum = 0, directed = False, add_self_loop = False,
                                rand_seed = 10, 
                                feats = None, adj_matrix = None, data_path = None,  n_parts = 0):
    """Geneate the GBP precomputation features
        Args:
                data_name ([type]): [description]
                num_proc ([type]): [description]
                alpha ([type]): [description]
                rmax ([type]): [description]
                rrr ([type]): [description]
                rwnum (int, optional): [description]. Defaults to 0.
                rand_seed (int, optional): [description]. Defaults to 10.
                feats ([type], optional): [description]. Defaults to None.
                adj_matrix ([type], optional): [description]. Defaults to None.
                data_path ([type], optional): [description]. Defaults to None.
                n_parts (int, optional): [description]. Defaults to 0.

        Returns:
                [ndarray(np.float32)]: [description]
    """
    a = Base(num_proc, alpha, rmax, rrr, rwnum,
            data_name, directed = directed, add_self_loop = add_self_loop,
            rand_seed = rand_seed, feats = feats, adj_matrix = adj_matrix)
    
    a.ppr_push()
    return a.get_gbp_feat()


### ============================ Prepare the input for cython-GBP computation  ===========

def graphsave(adj, data_name, data_path, need_save = True, directed = False):
    """
        generate the edges for cython-GBP pre-computation, direct should be consistent with GBP_feat_precomputation
        Save the adjacency matrix (scipy.sparse.csr.csr_matrix)  into coo format
        Args:
            adj (scipy.sparse.csr.csr_matrix): adjacency matrix 
            data_path (str) : to be saved file location
    """
    adj = adj.tocoo()
    
    graph_adj = np.transpose(np.vstack([adj.row, adj.col]))
    print(f'Adj matrix shape: {graph_adj.shape} !')
    # if not directed, than we will extract a single-direct edges as input to the cython-GBP precomputation
    if not directed:
        graph_adj = graph_adj[graph_adj[:, 0] < graph_adj[:, 1]]
        print(f'If undirected, adj matrix shape: {graph_adj.shape} !')
        
    graph_adj = np.ascontiguousarray(graph_adj, dtype=np.int64)  # all the graph node id should be long type
    if need_save:
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        np.save(data_path, graph_adj)
    return graph_adj


### =================== Load the input data for Cython GBP  ==================
def load_Cython_GBP_input_node(data_name, features, adj_full, adj_GBP_file_path = None, directed = False, redo_save = False):
    """ Load the two inputs for cython gbp precomputation: features and adj_matrix_cython
        adj_matrix_cython can be auto-saved for repeating usage in the future at the location: adj_GBP_file_path
        The output element types are important!!! 
    Args:
        data_name (string): name of the dataset, all lowercase
        features (np.array): node attributes
        adj_full (sp.sparse.csr_matrix): adjacency matrix, with no self-loops, default to be undirected edges and symmetric 
        adj_GBP_file_path (string, optional): file path to store the processed adjacency matrix for repeated usage
        directed (bool, optional): whether or not the graph is directed or not. Defaults to False.
        redo_save (bool, optional): whether or not we need to update the saved adj files. Defaults to False.

    Returns:
        features (np.array(np.float32)): features forced with a type for element np.float32
        adj_matrix_cython (np.array(np.int64)): adjacency matrix, aimed to be compatible with cython precomputation code
    """

    if adj_GBP_file_path is not None and os.path.exists(adj_GBP_file_path):
        if redo_save:
            print(f"Regenerate the adj_GBP_file at : {adj_GBP_file_path}")
            adj_matrix_cython = graphsave(adj_full, data_name.lower(), adj_GBP_file_path, need_save = False, directed = directed)
            np.save(adj_GBP_file_path, adj_matrix_cython) 
        else:
            print(f"Loading the pre-existent adj_GBP_file at : {adj_GBP_file_path}")
            adj_matrix_cython = np.load(adj_GBP_file_path)
            # with np.load(open(adj_GBP_file_path, 'rb'), allow_pickle=True) as data_input:
            #     adj_matrix_cython = data_input
    else:
        print(f"Produce a brand new adj_GBP_file at : {adj_GBP_file_path}")
        os.makedirs(os.path.dirname(adj_GBP_file_path), exist_ok=True)
        adj_matrix_cython = graphsave(adj_full, data_name.lower(), adj_GBP_file_path, need_save = False, directed = directed)
        np.save(adj_GBP_file_path, adj_matrix_cython) 
        
    return np.ascontiguousarray(features, dtype = np.float32), adj_matrix_cython



def load_Cython_GBP_input_link(data_name, features, adj_full, adj_GBP_file_path = None, directed = False, redo_save = False, 
                                val_frac = 0.05, test_frac = 0.1):
    """Load the two inputs for cython gbp precomputation: features and adj_matrix_cython
        adj_matrix_cython can be auto-saved for repeating usage in the future at the location: adj_GBP_file_path
        The output element types are important!!! 
    Args:
        data_name (string): name of the dataset, all lowercase
        features (np.array): node attributes
        adj_full (sp.sparse.csr_matrix): adjacency matrix, with no self-loops, default to be undirected edges and symmetric 
        adj_GBP_file_path (string, optional): file path to store the processed adjacency matrix for repeated usage
        directed (bool, optional): whether or not the graph is directed or not. Defaults to False.
        redo_save (bool, optional): whether or not we need to update the saved adj files. Defaults to False.

    Returns:
        features (np.array(np.float32)): features forced with a type for element np.float32
        adj_matrix_cython (np.array(np.int64)): adjacency matrix, aimed to be compatible with cython precomputation code
    """

    if adj_GBP_file_path is not None and os.path.exists(adj_GBP_file_path):
        if redo_save:
            print(f"Regenerate the adj_GBP_file at : {adj_GBP_file_path}")
            adj_train, _, val_edges, val_edges_false, test_edges, test_edges_false = Pre_utils.mask_test_edges(adj_full, val_frac = val_frac, test_frac = test_frac)
            adj_matrix_cython = graphsave(adj_train, data_name.lower(), adj_GBP_file_path, need_save = False, directed = directed)
            np.savez(adj_GBP_file_path, adj_matrix_cython = adj_matrix_cython, 
                    val_edges = val_edges, val_edges_false = val_edges_false, 
                    test_edges = test_edges, test_edges_false = test_edges_false)
            adj_train_file_path = os.path.join(os.path.dirname(adj_GBP_file_path), f'{data_name.lower()}_adj_train.npz')
            sp.save_npz(adj_train_file_path, adj_train)
        else:
            print(f"Loading the pre-existent adj_GBP_file at : {adj_GBP_file_path}")
            adj_train_file_path = os.path.join(os.path.dirname(adj_GBP_file_path), f'{data_name.lower()}_adj_train.npz')
            with np.load(open(adj_GBP_file_path, 'rb'), allow_pickle=True) as data_input:
                adj_matrix_cython = data_input['adj_matrix_cython']
                val_edges = data_input['val_edges']
                val_edges_false = data_input['val_edges_false']
                test_edges = data_input['test_edges']
                test_edges_false = data_input['test_edges_false']
            adj_train = sp.load_npz(adj_train_file_path)
    else:
        print(f"Produce a brand new adj_GBP_file at : {adj_GBP_file_path}")
        os.makedirs(os.path.dirname(adj_GBP_file_path), exist_ok=True)
        adj_train, _, val_edges, val_edges_false, test_edges, test_edges_false = Pre_utils.mask_test_edges(adj_full, val_frac = val_frac, test_frac = test_frac)
        adj_matrix_cython = graphsave(adj_train, data_name.lower(), adj_GBP_file_path, need_save = False, directed = directed)
        np.savez(adj_GBP_file_path, adj_matrix_cython = adj_matrix_cython, 
                    val_edges = val_edges, val_edges_false = val_edges_false, 
                    test_edges = test_edges, test_edges_false = test_edges_false)
        adj_train_file_path = os.path.join(os.path.dirname(adj_GBP_file_path), f'{data_name.lower()}_adj_train.npz')
        sp.save_npz(adj_train_file_path, adj_train)
        
    return np.ascontiguousarray(features, dtype = np.float32), adj_matrix_cython, val_edges, val_edges_false, test_edges, test_edges_false, adj_train

