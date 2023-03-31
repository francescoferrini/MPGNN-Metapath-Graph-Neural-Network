import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import FastRGCNConv, RGCNConv
from mp_rgcn_layer import *



class Net(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_rel, output_dim, ll_output_dim, n_layers, metapath_length):
        super().__init__()
        self.n_layers = n_layers
        self.metapath_length = metapath_length

        self.conv1 = RGCNConv(input_dim, hidden_dim, num_rel, flow='target_to_source') 
        self.conv2 = RGCNConv(hidden_dim, output_dim, num_rel, flow='target_to_source')
        self.LinearLayer = torch.nn.Linear(output_dim, ll_output_dim)

    def forward(self, x, edge_index, edge_type):
        for layer_index in range(0, self.metapath_length):
            if layer_index == 0:
                x = F.relu(self.conv1(x, edge_index, edge_type))
            else:
                x = F.relu(self.conv2(x, edge_index, edge_type))
        x = self.LinearLayer(x)
        #print(F.log_softmax(x, dim=1), F.log_softmax(x, dim=1).size())
        return F.log_softmax(x, dim=1)


'''
    MPGNN: To create a MPGNN I used the RGCNConv layer modifying the fact that for each 
           layer instead of considering all relation types we only consider one that
           is indicated by layer_index since I'm working with the case of 
           num_rel_types = num_layers
'''
class MPNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_rel, output_dim, ll_output_dim, n_layers, metapath):
        super().__init__()
        self.n_layers = n_layers
        self.metapath = metapath

        self.conv1 = CustomRGCNConv(input_dim, hidden_dim, num_rel, flow='target_to_source') 
        self.conv2 = CustomRGCNConv(hidden_dim, output_dim, num_rel, flow='target_to_source')
        self.LinearLayer = torch.nn.Linear(output_dim, ll_output_dim)

    def forward(self, x, edge_index, edge_type):
        '''
        Layer index is both the index of the layer and also the relation type
        we are considering in that specific layer
        '''

        for layer_index in range(0, len(self.metapath)):
            if layer_index == 0:
                x = F.relu(self.conv1(self.metapath[layer_index], x, edge_index, edge_type))
            else:
                x = F.relu(self.conv2(self.metapath[layer_index], x, edge_index, edge_type))
        x = self.LinearLayer(x)
        return F.log_softmax(x, dim=1)


