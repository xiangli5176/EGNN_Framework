import abc
import time
import math
import pdb
from math import ceil
import numpy as np
import scipy

from .graphsaint_cython_lib import norm_aggr, cython_utils  
from .graphsaint_cython_lib import cython_sampler as cy    # this needs the cython_utils as in its .so file, therefore must import it first as above 
from .. import utils

# use multiple global variables as stated in the globals

# core_par_sampler = 2
# samples_per_processor = -(-200 // core_par_sampler) # round up division

class graph_sampler:
    __metaclass__ = abc.ABCMeta
    def __init__(self, adj_train, node_train,size_subgraph, args_preproc):
        self.adj_train = adj_train
        self.node_train = np.unique(node_train).astype(np.int32)
        # size in terms of number of vertices in subgraph
        self.size_subgraph = size_subgraph
        self.name_sampler = 'None'
        self.node_subgraph = None
        self.preproc(**args_preproc)

    @abc.abstractmethod
    def preproc(self,**kwargs):
        pass

    def par_sample(self,stage, **kwargs):
        # call the cython sampler
        return self.cy_sampler.par_sample()


class rw_sampling(graph_sampler):
    def __init__(self, adj_train, node_train, size_subgraph, size_root, size_depth, 
                 core_par_sampler = 1, samples_per_processor = 200):
        self.size_root = size_root
        self.size_depth = size_depth
        size_subgraph = size_root * size_depth
        super().__init__(adj_train, node_train, size_subgraph, dict())
        
        self.cy_sampler = cy.RW(self.adj_train.indptr, self.adj_train.indices, self.node_train,\
            core_par_sampler, samples_per_processor, self.size_root, self.size_depth)
        
    def preproc(self, **kwargs):
        pass

class edge_sampling(graph_sampler):
    def __init__(self,adj_train, node_train, num_edges_subgraph, 
                 core_par_sampler = 1, samples_per_processor = 200):
        """
        num_edges_subgraph: specify the size of subgraph by the edge budget. NOTE: other samplers specify node budget.
        """
        self.num_edges_subgraph = num_edges_subgraph
        self.size_subgraph = num_edges_subgraph * 2       # this may not be true in many cases. But for now just use this.
        self.deg_train = np.array(adj_train.sum(1)).flatten()
        self.adj_train_norm = scipy.sparse.dia_matrix((1 / self.deg_train,0),shape=adj_train.shape).dot(adj_train)
        
        super().__init__(adj_train,node_train,self.size_subgraph,dict())
        #self.cy_sampler = cy.Edge(self.adj_train.indptr,self.adj_train.indices,self.node_train,\
        #    core_par_sampler,samples_per_processor,self.edge_prob_tri.row,self.edge_prob_tri.col,self.edge_prob_tri.data)
        
        self.cy_sampler = cy.Edge2(self.adj_train.indptr, self.adj_train.indices, self.node_train,
                            core_par_sampler, samples_per_processor, 
                        self.edge_prob_tri.row, self.edge_prob_tri.col, self.edge_prob_tri.data.cumsum(), self.num_edges_subgraph)
    
    def preproc(self,**kwargs):
        self.edge_prob = scipy.sparse.csr_matrix((np.zeros(self.adj_train.size), \
                self.adj_train.indices, self.adj_train.indptr), shape=self.adj_train.shape)
        self.edge_prob.data[:] = self.adj_train_norm.data[:]
        _adj_trans = scipy.sparse.csr_matrix.tocsc(self.adj_train_norm)
        
        self.edge_prob.data += _adj_trans.data      # P_e \propto a_{u,v} + a_{v,u}
        self.edge_prob.data *= 2 * self.num_edges_subgraph/self.edge_prob.data.sum()
        
        # now edge_prob is a symmetric matrix, we only keep the upper triangle part, since adj is assumed to be undirected.
        self.edge_prob_tri = scipy.sparse.triu(self.edge_prob).astype(np.float32)  # NOTE: in coo format



class mrw_sampling(graph_sampler):

    def __init__(self,adj_train,node_train,size_subgraph,size_frontier,max_deg=10000, 
                 core_par_sampler = 1, samples_per_processor = 200):
        self.p_dist = None
        super().__init__(adj_train,node_train,size_subgraph,dict())
        self.size_frontier = size_frontier
        self.deg_train = np.bincount(self.adj_train.nonzero()[0])
        self.name_sampler = 'MRW'
        self.max_deg = int(max_deg)
        self.cy_sampler = cy.MRW(self.adj_train.indptr, self.adj_train.indices, self.node_train,\
            core_par_sampler, samples_per_processor, self.p_dist, self.max_deg, self.size_frontier, self.size_subgraph)

    def preproc(self,**kwargs):
        _adj_hop = self.adj_train
        self.p_dist = np.array([_adj_hop.data[_adj_hop.indptr[v]:_adj_hop.indptr[v+1]].sum() for v in range(_adj_hop.shape[0])], dtype=np.int32)



class node_sampling(graph_sampler):
    
    def __init__(self, adj_train, node_train, size_subgraph, 
                 core_par_sampler = 1, samples_per_processor = 200):
        self.p_dist = np.zeros(len(node_train))
        super().__init__(adj_train,node_train,size_subgraph,dict())
        
        self.cy_sampler = cy.Node(self.adj_train.indptr, self.adj_train.indices, self.node_train,\
            core_par_sampler, samples_per_processor, self.p_dist, self.size_subgraph)

    def preproc(self,**kwargs):
        _p_dist = np.array([self.adj_train.data[self.adj_train.indptr[v]:self.adj_train.indptr[v+1]].sum() for v in self.node_train], dtype=np.int64)
        
        self.p_dist = _p_dist.cumsum()
        if self.p_dist[-1] > 2**31-1:
            print('warning: total deg exceeds 2**31')
            self.p_dist = self.p_dist.astype(np.float64)
            self.p_dist /= self.p_dist[-1]/(2**31-1)
        self.p_dist = self.p_dist.astype(np.int32)


class full_batch_sampling(graph_sampler):
    
    def __init__(self, adj_train,node_train, size_subgraph, 
                 core_par_sampler = 1, samples_per_processor = 200):
        super().__init__(adj_train,node_train,size_subgraph,dict())
        self.cy_sampler = cy.FullBatch(self.adj_train.indptr, self.adj_train.indices, self.node_train,\
            core_par_sampler, samples_per_processor)


class sage_sampling(graph_sampler):
    
    def __init__(self, adj_train, node_train, size_subgraph, input_neigh_deg = [10, 5],
                 core_par_sampler = 1, samples_per_processor = 200):

        self.input_neigh_deg = input_neigh_deg
        self.neigh_adj_train = None
        super().__init__(adj_train,node_train,size_subgraph, dict())  # this will call the self.preproc
        
        

        neigh_train_indptr = self.neigh_adj_train.indptr.astype(np.int32)   # need to do the type coersion into int
        neigh_train_data = self.neigh_adj_train.data.astype(np.int32)

        print("Start shuffling the node_train...")
        np.random.shuffle(self.node_train)

        self.cy_sampler = cy.Sage_Node(self.adj_train.indptr, self.adj_train.indices, self.node_train,\
            core_par_sampler, samples_per_processor, neigh_train_indptr, neigh_train_data, self.size_subgraph)

    def preproc(self,**kwargs):

        self.neigh_adj_train = utils.generate_train_neighbor(self.node_train, self.adj_train, input_neigh_deg = self.input_neigh_deg)
