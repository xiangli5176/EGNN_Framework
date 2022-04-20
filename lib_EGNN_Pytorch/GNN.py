import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Linear
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module



### ===================== DNN based  models ========================================
class GBP_GNNLayer(Module):
    def __init__(self, in_features, out_features, 
                batchnorm = False,
                activation_func=F.relu):
        """Basic GBP nn layer

        Args:
            in_features ([type]): [description]
            out_features ([type]): [description]
            batchnorm (bool, optional): whether to apply batch normalization. Defaults to False.
        """
        super(GBP_GNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.activation_func = activation_func

        self.batchnorm = batchnorm
        if self.batchnorm:
            self.batchnorm_fn = nn.BatchNorm1d(out_features)

        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, features, active=True):
        output = torch.mm(features, self.weight)
        if self.batchnorm:
            output = self.batchnorm_fn(output)
        if active:
            output = self.activation_func(output)
        return output


class Dense(nn.Module):
    def __init__(self, in_features, out_features, 
                batchnorm = False, 
                activation_func=F.relu):
        super(Dense, self).__init__()
        
        self.basic_linear = Linear(in_features, out_features)
        self.in_features = in_features
        self.out_features = out_features
        self.activation_func = activation_func

        self.batchnorm = batchnorm
        if self.batchnorm:
            self.batchnorm_fn = nn.BatchNorm1d(out_features)

    def forward(self, input, active=True):
        output = self.basic_linear(input)
        if self.batchnorm:
            output = self.batchnorm_fn(output)
        if active:
            output = self.activation_func(output)
        return output


class GNNLayer(Module):
    def __init__(self, in_features, out_features):
        super(GNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, features, adj, active=True):
        support = torch.mm(features, self.weight)
        output = torch.spmm(adj, support)
        # output and support are of the same shape
        if active:
            output = F.relu(output)
        return output
    
    
class Encoder(torch.nn.Module):
    def __init__(self, config, layer_dims, 
                base_model = Dense, activation_func = F.relu):
        """base_model could be : GNNLayer or GCNConv"""
        super(Encoder, self).__init__()

        # GCN for inter information
        self.GNN_layers = nn.ModuleList([base_model(*dim, batchnorm=config["batchnorm"], 
                                            activation_func = activation_func) 
                                            for dim in layer_dims] )
        self.f_dropout = nn.Dropout(p = config["dropout_rate"])

    def forward(self, x, train = True):
        for i, GNN_layer in enumerate(self.GNN_layers[:-1]):
            x = GNN_layer(x, active=True)
            if train:
                x = self.f_dropout(x)
        x = self.GNN_layers[-1](x, active=False)

        return x

 
class DNN(torch.nn.Module):
    def __init__(self, config, layer_dims, 
                base_model = Dense, activation_func = F.elu):
        """base_model could be : GNNLayer or GCNConv"""
        super(DNN, self).__init__()

        self.GNN_layers = nn.ModuleList([base_model(*dim, batchnorm = config["batchnorm"], 
                                            activation_func = activation_func) 
                                            for dim in layer_dims] )
        # self.f_dropout = nn.Dropout(p = config["dropout_rate"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            This is the Non-linear two-layer MLP porjection g
            for stronger expression
        """
        for i, GNN_layer in enumerate(self.GNN_layers[:-1]):
            x = GNN_layer(x, active=True)
        x = self.GNN_layers[-1](x, active=False)

        return x    
    
    
### ====================== GCN models ============================    