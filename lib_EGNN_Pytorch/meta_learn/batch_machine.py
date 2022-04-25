import math
import torch
import scipy.sparse as sp

import numpy as np
import time

from .. import utils, evaluation
from ..data_preprocessing import Pre_utils_graphsaint
from .graphsaint_cython_lib.norm_aggr import norm_aggr
from . import samplers




class Minibatch_clustergcn:
    """
        This minibatch iterator iterates over nodes for supervised learning.
        Data transferred to GPU:     A  init: 1) self.adj_full_norm;  2) self.norm_loss_test;
                                     B  set_sampler:  1) self.norm_loss_train
                                     C  one_batch : 1) subgraph adjacency matrix (adj)
    """

    def __init__(self, adj_full, adj_train, role, train_params, 
                 cpu_eval = False, mode = "train", 
                 num_clusters = 128, batch_num = 32):
        """
        role:       array of string (length |V|)
                    storing role of the node ('tr'/'va'/'te')
        """
        self.use_cuda = torch.cuda.is_available()
        if cpu_eval:
            self.use_cuda = False
        
        # store all the node roles as the numpy array:
        self.node_train = np.array(role['tr'])
        self.node_val = np.array(role['va'])
        self.node_test = np.array(role['te'])

        
        self.adj_train = adj_train
        # print("adj train type is: {}; and shape is {}".format(type(adj_train), adj_train.shape))

        # norm_loss_test is used in full batch evaluation (without sampling). so neighbor features are simply averaged.
        self.norm_loss_test = np.zeros(adj_full.shape[0])
        
        _denom = len(self.node_train) + len(self.node_val) +  len(self.node_test)
        
        # instead of assign all elements of self.norm_loss_test to the same averaged denominator, separately assingment instead. 
        # does this mean there are other meaningless roles beyond: test, train and validation?
        self.norm_loss_test[self.node_train] = 1./_denom     
        self.norm_loss_test[self.node_val] = 1./_denom
        self.norm_loss_test[self.node_test] = 1./_denom
        self.norm_loss_test = torch.from_numpy(self.norm_loss_test.astype(np.float32))
            
        self.deg_train = np.array(self.adj_train.sum(1)).flatten()   # sum the degree of each train node, here sum along column for adjacency matrix
        
        # for train part: use the modified adjacency matrix: with inter-cluster edges broken
        if mode == "train": 
            self.adj_full, self.parts = Pre_utils_graphsaint.partition_graph(adj_train, self.node_train, num_clusters)
            self.generate_norm_loss_train(num_clusters, batch_num)
            self.num_training_batches = batch_num
            self.num_mini_clusters = num_clusters
        else:
            self.adj_full = adj_full




    def generate_norm_loss_train(self, num_clusters, batch_num):
        """
            Train_phases (a dict defined in the .yml file) : usually including : end, smapler, size_subg_edge
            end:  number of total epochs to stop
            sampler: category for sampler (e.g. edge)
            size_subg_edge:  size of the subgraph in number of edges
        """
        self.norm_loss_train = np.zeros(self.adj_train.shape[0])

        self.norm_loss_train[self.node_train] += 1
        assert self.norm_loss_train[self.node_val].sum() + self.norm_loss_train[self.node_test].sum() == 0
        
        # normalize the self.norm_loss_train:
        self.norm_loss_train[np.where(self.norm_loss_train==0)[0]] = 0.1
        self.norm_loss_train[self.node_val] = 0
        self.norm_loss_train[self.node_test] = 0
        self.norm_loss_train[self.node_train] = batch_num/self.norm_loss_train[self.node_train]/self.node_train.size
        self.norm_loss_train = torch.from_numpy(self.norm_loss_train.astype(np.float32))

        
    def generate_train_batch(self, diag_lambda=-1):
        """
        Train batch Generator: Generate the batch for multiple clusters.
        """

        block_size = self.num_mini_clusters // self.num_training_batches
        np.random.shuffle(self.parts)  # each time shuffle different mini-clusters so that the combined batches are shuffled correspondingly
        
        for _, st in enumerate(range(0, self.num_mini_clusters, block_size)):
            # recombine mini-clusters into a single batch: pt
            node_subgraph = self.parts[st]
            for pt_idx in range(st + 1, min(st + block_size, self.num_mini_clusters)):
                node_subgraph = np.concatenate((node_subgraph, self.parts[pt_idx]), axis=0)
            
            norm_loss = self.norm_loss_train[node_subgraph]
            subgraph_adj = self.adj_full[node_subgraph, :][:, node_subgraph]

            # normlize subgraph_adj locally for each isolate subgraph
            if diag_lambda == -1:
                subgraph_adj = Pre_utils_graphsaint.adj_norm(subgraph_adj, deg = self.deg_train[node_subgraph])
            else:
                subgraph_adj = Pre_utils_graphsaint.adj_norm_diag_enhance(subgraph_adj, deg = self.deg_train[node_subgraph], diag_lambda = diag_lambda)
            subgraph_adj = Pre_utils_graphsaint._coo_scipy2torch(subgraph_adj.tocoo())
            
            yield (node_subgraph, subgraph_adj, norm_loss)    
    
    def generate_eval_batch(self):
        """
            Generate evaluation batch for validation/test procedures, whole graph 
        """
        node_subgraph = np.arange(self.adj_full.shape[0])  # include all the nodes inside the graph
        adj_full_norm = Pre_utils_graphsaint.adj_norm(self.adj_full)  # return the normalized whole graph adj matrix; optional: diag_enhanced normalization: adj_norm_diag_enhance(...)
#             adj = adj_norm_diag_enhance(self.adj_full, diag_lambda = -1)
        adj_full_norm = Pre_utils_graphsaint._coo_scipy2torch(adj_full_norm.tocoo())

        return node_subgraph, adj_full_norm, self.norm_loss_test


class Minibatch_Saint:
    """
        This minibatch iterator iterates over nodes for supervised learning.
        Data transferred to GPU:     A  init: 1) self.adj_full_norm;  2) self.norm_loss_test;
                                     B  set_sampler:  1) self.norm_loss_train
                                     C  one_batch : 1) subgraph adjacency matrix (adj)
    """

    def __init__(self, adj_full_norm, adj_train, role, train_params, 
                 cpu_eval = False, num_cpu_core = 1):
        """
        role:       array of string (length |V|)
                    storing role of the node ('tr'/'va'/'te')
        """
        self.num_cpu_core = num_cpu_core
        self.use_cuda = torch.cuda.is_available()
        if cpu_eval:
            self.use_cuda = False
        
        # store all the node roles as the numpy array:
        self.node_train = np.array(role['tr'])
        self.node_val = np.array(role['va'])
        self.node_test = np.array(role['te'])

        # self.adj_full_norm : torch sparse tensor
        self.adj_full_norm = Pre_utils_graphsaint._coo_scipy2torch(adj_full_norm.tocoo())
        self.adj_train = adj_train

        # below: book-keeping for mini-batch
        self.node_subgraph = None
        self.batch_num = -1

        # all the subgraph attributes should be used for the training process
        self.method_sample = None
        self.subgraphs_remaining_indptr = []
        self.subgraphs_remaining_indices = []
        self.subgraphs_remaining_data = []
        self.subgraphs_remaining_nodes = []
        self.subgraphs_remaining_edge_index = []
        
        # What is this norm_loss aimed at?
        self.norm_loss_train = np.zeros(self.adj_train.shape[0])
        # norm_loss_test is used in full batch evaluation (without sampling). so neighbor features are simply averaged.
        self.norm_loss_test = np.zeros(self.adj_full_norm.shape[0])
        
        _denom = len(self.node_train) + len(self.node_val) +  len(self.node_test)
        
        # instead of assign all elements of self.norm_loss_test to the same averaged denominator, separately assingment instead. 
        # does this mean there are other meaningless roles beyond: test, train and validation?
        self.norm_loss_test[self.node_train] = 1./_denom     
        self.norm_loss_test[self.node_val] = 1./_denom
        self.norm_loss_test[self.node_test] = 1./_denom
        self.norm_loss_test = torch.from_numpy(self.norm_loss_test.astype(np.float32))
        
            
        self.norm_aggr_train = np.zeros(self.adj_train.size)
        
        self.sample_coverage = train_params['sample_coverage']
        self.deg_train = np.array(self.adj_train.sum(1)).flatten()   # sum the degree of each train node, here sum along column for adjacency matrix


    def set_sampler(self, train_phases, input_neigh_deg = [10, 5], core_par_sampler = 1, samples_per_processor = 200):
        """
            Train_phases (a dict defined in the .yml file) : usually including : end, smapler, size_subg_edge
            end:  number of total epochs to stop
            sampler: category for sampler (e.g. edge)
            size_subg_edge:  size of the subgraph in number of edges
        """
        
        self.subgraphs_remaining_indptr = list()
        self.subgraphs_remaining_indices = list()
        self.subgraphs_remaining_data = list()
        self.subgraphs_remaining_nodes = list()
        self.subgraphs_remaining_edge_index = list()
        
        self.method_sample = train_phases['sampler']   # one of the string indicators regarding sampler methods
        if self.method_sample == 'mrw':
            if 'deg_clip' in train_phases:
                _deg_clip = int(train_phases['deg_clip'])
            else:
                _deg_clip = 100000      # setting this to a large number so essentially there is no clipping in probability
            self.size_subg_budget = train_phases['size_subgraph']
            self.graph_sampler = samplers.mrw_sampling(self.adj_train, self.node_train,
                                self.size_subg_budget, train_phases['size_frontier'], _deg_clip, 
                                        core_par_sampler = core_par_sampler, samples_per_processor = samples_per_processor)
        elif self.method_sample == 'rw':
            self.size_subg_budget = train_phases['num_root'] * train_phases['depth']
            self.graph_sampler = samplers.rw_sampling(self.adj_train, self.node_train,
                                self.size_subg_budget, int(train_phases['num_root']), int(train_phases['depth']), 
                                        core_par_sampler = core_par_sampler, samples_per_processor = samples_per_processor)
        elif self.method_sample == 'edge':
            self.size_subg_budget = train_phases['size_subg_edge'] * 2
            self.graph_sampler = samplers.edge_sampling(self.adj_train, self.node_train, train_phases['size_subg_edge'], 
                                        core_par_sampler = core_par_sampler, samples_per_processor = samples_per_processor)
        elif self.method_sample == 'node':
            self.size_subg_budget = train_phases['size_subgraph']
            self.graph_sampler = samplers.node_sampling(self.adj_train,self.node_train, self.size_subg_budget, 
                                        core_par_sampler = core_par_sampler, samples_per_processor = samples_per_processor)
        elif self.method_sample == 'full_batch':
            self.size_subg_budget = self.node_train.size
            self.graph_sampler = samplers.full_batch_sampling(self.adj_train,self.node_train, self.size_subg_budget, 
                                        core_par_sampler = core_par_sampler, samples_per_processor = samples_per_processor)
        elif self.method_sample == 'sage_node':
            self.size_subg_budget = train_phases['size_subgraph']
            self.graph_sampler = samplers.sage_sampling(self.adj_train,self.node_train, self.size_subg_budget, input_neigh_deg = input_neigh_deg,
                                        core_par_sampler = core_par_sampler, samples_per_processor = samples_per_processor)
            print("using sage node sampler! ")
        else:
            raise NotImplementedError

        self.norm_loss_train = np.zeros(self.adj_train.shape[0])
        self.norm_aggr_train = np.zeros(self.adj_train.size).astype(np.float32)

        # For edge sampler, no need to estimate norm factors, we can calculate directly.
        # However, for integrity of the framework, we decide to follow the same procedure for all samplers: 
        # 1. sample enough number of subgraphs
        # 2. estimate norm factor alpha and lambda
        tot_sampled_nodes = 0
        while True:
            self.par_graph_sample('train')
            tot_sampled_nodes = sum([len(n) for n in self.subgraphs_remaining_nodes])
            if tot_sampled_nodes > self.sample_coverage * self.node_train.size:
                break
        print()
        num_subg = len(self.subgraphs_remaining_nodes)  # each subgraph nodes are stored as one list inside the self.subgraphs_remaining_nodes
        for i in range(num_subg):
            self.norm_aggr_train[self.subgraphs_remaining_edge_index[i]] += 1
            self.norm_loss_train[self.subgraphs_remaining_nodes[i]] += 1
        assert self.norm_loss_train[self.node_val].sum() + self.norm_loss_train[self.node_test].sum() == 0
        for v in range(self.adj_train.shape[0]):
            i_s = self.adj_train.indptr[v]
            i_e = self.adj_train.indptr[v+1]
            val = np.clip(self.norm_loss_train[v]/self.norm_aggr_train[i_s:i_e], 0, 1e4)
            val[np.isnan(val)] = 0.1
            self.norm_aggr_train[i_s:i_e] = val
        
        # normalize the self.norm_loss_train:
        self.norm_loss_train[np.where(self.norm_loss_train==0)[0]] = 0.1
        self.norm_loss_train[self.node_val] = 0
        self.norm_loss_train[self.node_test] = 0
        self.norm_loss_train[self.node_train] = num_subg/self.norm_loss_train[self.node_train]/self.node_train.size
        self.norm_loss_train = torch.from_numpy(self.norm_loss_train.astype(np.float32))

    # each time finish one-time sampling: generate a single sample subgraph
    def par_graph_sample(self, phase):
        """
           Phase: can be a string "train"
        """
        t0 = time.time()
        _indptr, _indices, _data, _v, _edge_index = self.graph_sampler.par_sample(phase)
        t1 = time.time()
        # create 200 subgraphs per CPU, these 200 graphs may be generated by different cores, but 200 each time, not to exceed the memory limit
        print('sampling 200 subgraphs:   time = {:.3f} sec'.format(t1 - t0), end="\r")
        self.subgraphs_remaining_indptr.extend(_indptr)   # add lists into the subgraphs_remaining_indptr, each list is a subgraph
        self.subgraphs_remaining_indices.extend(_indices)
        self.subgraphs_remaining_data.extend(_data)
        self.subgraphs_remaining_nodes.extend(_v)
        self.subgraphs_remaining_edge_index.extend(_edge_index)


    def generate_train_batch(self):
        if len(self.subgraphs_remaining_nodes) == 0:
            self.par_graph_sample('train')   # if there is no sampled subgraphs, then make one
            print()

        self.node_subgraph = self.subgraphs_remaining_nodes.pop()
        self.size_subgraph = len(self.node_subgraph)
        adj = sp.csr_matrix((self.subgraphs_remaining_data.pop(),\
                                self.subgraphs_remaining_indices.pop(),\
                                self.subgraphs_remaining_indptr.pop()),\
                                shape=(self.size_subgraph,self.size_subgraph))
        adj_edge_index = self.subgraphs_remaining_edge_index.pop()
        #print("{} nodes, {} edges, {} degree".format(self.node_subgraph.size,adj.size,adj.size/self.node_subgraph.size))
        norm_aggr(adj.data, adj_edge_index, self.norm_aggr_train, num_proc = self.num_cpu_core)
        adj = Pre_utils_graphsaint.adj_norm(adj, deg = self.deg_train[self.node_subgraph])
        adj = Pre_utils_graphsaint._coo_scipy2torch(adj.tocoo())
        
        self.batch_num += 1          # create one batch
            
        norm_loss = self.norm_loss_train
        norm_loss = norm_loss[self.node_subgraph]
        # this self.node_subgraph is to select the target nodes, can be left on the CPU
        
        # for evaluation: all nodes, its adj_full_norm and norm_loss_test 
        return self.node_subgraph, adj, norm_loss
    
    def generate_eval_batch(self):
        
        self.node_subgraph = np.arange(self.adj_full_norm.shape[0])  # include all the nodes inside the graph
        adj = self.adj_full_norm
        
        norm_loss = self.norm_loss_test 
        norm_loss = norm_loss[self.node_subgraph]
        # this self.node_subgraph is to select the target nodes, can be left on the CPU
        
        # for evaluation: all nodes, its adj_full_norm and norm_loss_test 
        return self.node_subgraph, adj, norm_loss
        

    def num_training_batches(self):
        return math.ceil(self.node_train.shape[0] / float(self.size_subg_budget))

    def shuffle(self):
        self.node_train = np.random.permutation(self.node_train)
        self.batch_num = -1

    def end(self):
        return (self.batch_num + 1) * self.size_subg_budget >= self.node_train.shape[0]   # greater or equal to the number of train nodes



def evaluate(snap_model_folder, minibatch_eval, model_eval, epoch_idx, mode='val'):
    """
        Perform the evaluation: either validaiton or test offline from saved snapshot of the models
        generate evaluation results from a single timepoint snapshot of the trained model
        return : micro_f1 score, macro_f1 score
        
    """
    ### location to output the evaluation result
    
    snap_model_file = snap_model_folder + 'snapshot_epoch_' + str(epoch_idx) + '.pkl'

    model_eval.load_state_dict(torch.load(snap_model_file, map_location=lambda storage, loc: storage))

    loss_val, f1mic_val, f1mac_val = evaluate_full_batch(model_eval, minibatch_eval, mode = mode)

    return f1mic_val, f1mac_val            


def evaluate_full_batch(model, minibatch, mode='val'):
    """
        Full batch evaluation: for validation and test sets only.
        When calculating the F1 score, we will mask the relevant root nodes.
        mode: can be val or test
    """
    loss, preds, labels = model.eval_step(*minibatch.generate_eval_batch())
    node_val_test = minibatch.node_val if mode=='val' else minibatch.node_test
    # may not be necessary 
    f1_scores = evaluation.calc_f1(utils.to_numpy(labels[node_val_test]), utils.to_numpy(preds[node_val_test]), model.sigmoid_loss)
    return loss, f1_scores[0], f1_scores[1]


