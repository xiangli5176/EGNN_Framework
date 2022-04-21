import torch
import torch.nn as nn
import torch.nn.functional as F

from .GNN_basic import GBP_GNNLayer, Encoder, DNN


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



class RwCL_Model(torch.nn.Module):
    def __init__(self, config):
        super(RwCL_Model, self).__init__()
        enc_dims, mlp_dims, n_z = set_dims(config)

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
def set_dims(config):
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