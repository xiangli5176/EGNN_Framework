import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from .GNN_basic import GBP_GNNLayer, Encoder, DNN, Dense, AE


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