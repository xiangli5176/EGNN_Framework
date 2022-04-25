import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from .GNN_basic import GBP_GNNLayer, Encoder, DNN, Dense, AE
from . import GNN_basic   # for model establishment


class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret


### ============================== RwCL model =====================
class RwCL_Model(torch.nn.Module):
    def __init__(self, config):
        super(RwCL_Model, self).__init__()
        enc_dims, mlp_dims, n_z = set_dims_RwCL(config)

        self.encoder = Encoder(config, enc_dims, base_model = GBP_GNNLayer, activation_func = F.relu)
        self.mlp_arch = DNN(config, mlp_dims, base_model = GBP_GNNLayer, activation_func = F.elu) if mlp_dims else lambda x : x
        # self.mlp_arch = trial_Encoder(config)

        self.tau: float = config["tau"]

    def forward(self, x, train = True):
        return self.encoder(x, train = train)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        """ Should be cosine similarity..."""
        z1 = F.normalize(z1)    # forbenius normalization
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        """Claculate the loss fuction: l (u_i, v_i)
            between_sim.sum(1): 
                include 1) postive pairs: between_sim.diag()
                        2) Intra-view pairs, all the others in each sum along axis=1
            refl_sim.sum(1) - refl_sim.diag(): all the inter-view pairs for u_i
        Args:
            z1 (torch.Tensor): ui, the anchor, N by K
            z2 (torch.Tensor): vi

        Returns:
            [torch.Tensor]: the contrastive loss 
        """

        # all the postive smaples are along the diagonal of : between_sim
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1)) # should be N by N
        between_sim = f(self.sim(z1, z2))
        
        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))


    def loss(self, z1: torch.Tensor, z2: torch.Tensor,
             mean: bool = True):

        h1 = self.mlp_arch(z1)
        h2 = self.mlp_arch(z2)

        l1 = self.semi_loss(h1, h2)
        l2 = self.semi_loss(h2, h1)
        # else:
        #     l1 = self.batched_semi_loss(h1, h2, batch_size)
        #     l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5       # element wise for symmetric
        ret = ret.mean() if mean else ret.sum()  # along all n nodes

        return ret


def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x


# auto-encoder
def set_dims_RwCL(config):
    """
        dims_feat: Obtain the dimension of each embedding layer
        dims_weight:  Obtain the dimension of each weight
    """
    enc_arch = [int(val) for val in config["enc_arch"].split('-')  if val] 
    enc_dims = [(config["feat_dim"], enc_arch[0])]
    enc_dims += [(enc_arch[i], enc_arch[i+1]) for i in range(len(enc_arch)-1)]

    mlp_dims = []
    if config["mlp_arch"]:
        mlp_arch = [int(val) for val in config["mlp_arch"].split('-') if val]
        mlp_dims = [(enc_arch[-1], mlp_arch[0])]
        mlp_dims += [(mlp_arch[i], mlp_arch[i+1]) for i in range(len(mlp_arch)-1)]

    return enc_dims, mlp_dims, enc_arch[-1] 



### ============================== RwSL model =====================

class RwSL_Model(nn.Module):

    def __init__(self, config, n_clusters, v = 1, pretrain_path = None):
                
        super(RwSL_Model, self).__init__()

        # autoencoder for intra information
        enc_dims, dec_dims, n_z = set_dims_RwSL(config)

        self.ae = AE(enc_dims, dec_dims, config)
        self.ae.load_state_dict(torch.load(pretrain_path, map_location='cpu')) # 

        # GCN for inter information
        self.GNN_layers = nn.ModuleList([GBP_GNNLayer(*dim, batchnorm=config["batchnorm"]) for dim in enc_dims] + [GBP_GNNLayer(n_z, n_clusters, batchnorm=config["batchnorm"])])
        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        # degree
        self.v = v
        self.sigma = config["sigma"]
        self.f_dropout = nn.Dropout(p = config["dropout_rate"])

    def forward(self, x):
        # DNN Module
        x_bar, z, enc_tran = self.ae(x)
        
        # GCN Module
        h = self.GNN_layers[0](x)
        for i, GNN_layer in enumerate(self.GNN_layers[1:-1]):
            h = self.f_dropout(h)
            h = GNN_layer((1 - self.sigma) * h + self.sigma * enc_tran[i])

        h = self.GNN_layers[-1]((1 - self.sigma) * h + self.sigma * z, active=False)

        predict = F.softmax(h, dim=1)

        # Dual Self-supervised Module
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return x_bar, q, predict, z
    
    
def set_dims_RwSL(config):
    """
        dims_feat: Obtain the dimension of each embedding layer
        dims_weight:  Obtain the dimension of each weight
    """
    arch = [int(val) for val in config["arch"].split('-')] 
    enc_dims = [(config["n_input"], arch[0])]
    enc_dims += [(arch[i], arch[i+1]) for i in range(len(arch)-1)]
    dec_dims = [(right, left) for left, right in reversed(enc_dims)]

    return enc_dims, dec_dims, arch[-1]    



### ============================== GraphSaint model =====================

def parse_layer_yml(arch_gcn, dim_input):
    """
        arch_gcn (dict of network structure): architecture of GCN 
    """
    num_layers = len(arch_gcn['arch'].split('-'))
    # set default values, then update by arch_gcn
    bias_layer = [arch_gcn['bias']] * num_layers
    act_layer = [arch_gcn['act']] * num_layers
    aggr_layer = [arch_gcn['aggr']] * num_layers
    dims_layer = [arch_gcn['dim']] * num_layers
    
    order_layer = [int(o) for o in arch_gcn['arch'].split('-')]
    
    return [dim_input] + dims_layer, order_layer, act_layer, bias_layer, aggr_layer




class GraphSAINT(nn.Module):
    """
        Trainer model
        Transfer data to GPU:   A  init:  1) feat_full   2) label_full   3) label_full_cat
    """
    def __init__(self, num_classes, arch_gcn, train_params, feat_full, label_full, cpu_eval=False):
        """
        Inputs:
            arch_gcn            parsed arch of GCN
            train_params        parameters for training
            cpu_eval (bool)  :   whether use CPU side for evalution
        """
        super(GraphSAINT,self).__init__()
        
        self.use_cuda = torch.cuda.is_available()
        if cpu_eval:
            self.use_cuda=False
        
        self.aggregator_cls = GNN_basic.HighOrderAggregator
        self.mulhead=1
        
        # each layer in arch_gcn['arch']:  is a string separated by '-'
        self.num_layers = len(arch_gcn['arch'].split('-'))
        self.weight_decay = train_params['weight_decay']
        self.dropout = train_params['dropout']
        self.lr = train_params['lr']
        self.arch_gcn = arch_gcn
        
        # check if the task is a multi-label task
        # sigmoid: means this is a multi-label task
        self.sigmoid_loss = (arch_gcn['loss']=='sigmoid')   # use sigmoid for multi-label loss function
        
        self.feat_full = torch.from_numpy(feat_full.astype(np.float32))
        self.label_full = torch.from_numpy(label_full.astype(np.float32))
        
        
        self.num_classes = num_classes
        _dims, self.order_layer, self.act_layer, self.bias_layer, self.aggr_layer \
                        = parse_layer_yml(arch_gcn, self.feat_full.shape[1])
        
        # get layer index for each conv layer, useful for jk net last layer aggregation
        self.set_dims(_dims)

        self.loss = 0

        # build the model below        
        self.aggregators = self.get_aggregators()
        self.conv_layers = nn.Sequential(*self.aggregators)  # make sure the previous layer's output is the next one's output
        # calculate the final embeddings:
        self.classifier = GNN_basic.HighOrderAggregator(self.dims_feat[-1], self.num_classes,\
                            act='I', order=0, dropout=self.dropout, bias='bias')
        
        self.optimizer = torch.optim.Adam(self.parameters(),lr=self.lr)

    def set_dims(self,dims):
        """
            dims_feat: Obtain the dimension of each embedding layer
            dims_weight:  Obtain the dimension of each weight
        """
        # dims[0] will the number of features
        # if using the concatenation as the feature aggration pattern, then need to count the dimension expansion
        self.dims_feat = [dims[0]] + [( (self.aggr_layer[l]=='concat') * self.order_layer[l] + 1) * dims[l+1] for l in range(len(dims)-1)]
        
        # set the current model weights dimensions, generated based on the embedding dimension:
        self.dims_weight = [(self.dims_feat[l], dims[l+1]) for l in range(len(dims)-1)]


    def forward(self, adj_subgraph, feat_subg):
        
        _, emb_subg = self.conv_layers((adj_subgraph, feat_subg))
        emb_subg_norm = F.normalize(emb_subg, p=2, dim=1)
        
        # obtain the prediction
        pred_subg = self.classifier((None, emb_subg_norm))[1]
        return pred_subg


    def _loss(self, preds, labels, norm_loss):
        """
            use the norm_loss as the weight factor
        """
        if self.sigmoid_loss:
            norm_loss = norm_loss.unsqueeze(1)
            return torch.nn.BCEWithLogitsLoss(weight = norm_loss,reduction='sum')(preds, labels)
        else:
            _ls = torch.nn.CrossEntropyLoss(reduction='none')(preds, labels)
            return (norm_loss * _ls).sum()


    def get_aggregators(self):
        """
        Return a list of aggregator instances. to be used in self.build()
        """
        aggregators = []
        for l in range(self.num_layers):
            aggrr = self.aggregator_cls(*self.dims_weight[l], dropout=self.dropout,\
                    act=self.act_layer[l], order=self.order_layer[l], \
                    aggr=self.aggr_layer[l], bias=self.bias_layer[l], mulhead=self.mulhead)
            aggregators.append(aggrr)
        return aggregators

    def predict(self, preds):
        return nn.Sigmoid()(preds) if self.sigmoid_loss else F.softmax(preds, dim=1)
        
        
    def train_step(self, node_subgraph, adj_subgraph, norm_loss_subgraph, feat_subg, label_subg_converted):
        """
        Purpose:  only count the time for the training process, including forward and backward propogation
        Forward and backward propagation
        norm_loss_subgraph : is the key to rescale the current batch/subgraph
        """
        self.train()
        # =============== start of the training process for one step: =================
        self.optimizer.zero_grad()
        # here call the forward propagation
        preds = self(adj_subgraph, feat_subg)    # will call the forward function
        loss = self._loss(preds, label_subg_converted, norm_loss_subgraph) # labels.squeeze()?
        
        # call the back propagation
        loss.backward()
        # any clipin ggradient optimization?
        torch.nn.utils.clip_grad_norm_(self.parameters(), 5)  #
#         Clips gradient norm of an iterable of parameters.
#         The norm is computed over all gradients together, as if they were concatenated into a single vector. Gradients are modified in-place.
        
        self.optimizer.step()
        # ending of the training process
        
        # also return the total training time and uploading time in seconds
        return loss, self.predict(preds)

    def eval_step(self, node_subgraph, adj_subgraph, norm_loss_subgraph):
        """
        Purpose: evaluation only on the CPU side
        Forward propagation only
        No backpropagation and thus no need for gradients
        """
        self.eval()
        feat_subg = self.feat_full[node_subgraph]
        label_subg = self.label_full[node_subgraph]
        
        if not self.sigmoid_loss:
            self.label_full_cat = torch.from_numpy(self.label_full.numpy().argmax(axis=1).astype(np.int64))
                
        label_subg_converted = label_subg if self.sigmoid_loss else self.label_full_cat[node_subgraph]
        
        with torch.no_grad():
            # only call the forward propagation
            print("during the evaluation step, report the input matrices size: ")
            print("adj_subgraph size :  {}; \t feat_subg size : {}".format(adj_subgraph.size(), feat_subg.size()) )

            preds = self(adj_subgraph, feat_subg)
            loss = self._loss(preds, label_subg_converted, norm_loss_subgraph)
            
        return loss, self.predict(preds), label_subg