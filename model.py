import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn
import random
import numpy as np


FEATURES_DIM = 2#4231#3066#6##

class InputLayer(torch.nn.Module):
    def __init__(self, weights):
        super(InputLayer, self).__init__()
        # Trainable weights
        self.weights = nn.Parameter(weights.unsqueeze(-1))
        #self.weights = nn.Parameter(torch.rand(len(weights)).unsqueeze(-1))
        #self.weights = nn.ParameterList([nn.Parameter(weights[i]) for i in range(weights.shape[0])])
    def forward(self):
        return self.weights


class OutputLayer(torch.nn.Module):
    def __init__(self):
        super(OutputLayer, self).__init__()

        # # Linear layer for features
        self.LinearLayerAttri = nn.Linear(FEATURES_DIM, 1, bias=False)
        #self.Linear = nn.Parameter(torch.ones(FEATURES_DIM))
        
    def forward(self, weights, data: Data, node_dict, BAGS, COMPLEX, feat):
        # retrieve features and multiply for a linear function
        #features = data.x.type(torch.FloatTensor)
        #linear_features = self.LinearLayerAttri(features)
        # if BAGS == true means the model is predicting the bags
        # otherwise it is predicting each single source node
        if BAGS:
            # Tensor to save the max destination weight for each bag.
            # The size of the tensor is the total number of bags
            max_weights = torch.zeros(len(data.bags), 1)

            # dictionary of max destination nodes (keys are bags as strings since it is not possible to have
            # lists as keys of a dictionary)
            max_destination_node_for_bag = {}
            # dictionary of max destination nodes (keys are source nodes)
            max_destination_node_for_source = {}
            # max destination nodes
            max_destination_nodes = []
            for i, bag in enumerate(data.bags):
                max_weight_for_current_bag = -10
                for source_node in bag:
                    if source_node in node_dict:
                        # retrieve weights of nodes connected to source_node
                        weights_of_source = weights[node_dict[source_node]].squeeze(-1)
                        #print('----')
                        #print('weights of source: ', weights_of_source.tolist())
                        # If we are in the complex case, multiply with the linear layer of the features
                        weights_of_source *= self.LinearLayerAttri(feat[source_node])
                        #print('linear : ', self.LinearLayerAttri.weight.tolist())
                        #print('feat: ', feat[source_node])
                        #print('linear*feat: ', self.LinearLayerAttri(feat[source_node]))
                        #print('weights of source: ', weights_of_source.tolist())
                        #print('index: ', torch.argmax(weights_of_source).item())
                        max_node = node_dict[source_node][torch.argmax(weights_of_source).item()]   
                        # retrieve the max weight for the source_node in the bag
                        max_destination_node_for_source[source_node] = weights[max_node]*self.LinearLayerAttri(feat[source_node])
                        if max_node not in max_destination_nodes: max_destination_nodes.append(max_node)
                        # put in max_node_for_current_bag the max node so far for this bag
                        if max_destination_node_for_source[source_node] > max_weight_for_current_bag: 
                            max_weight_for_current_bag = max_destination_node_for_source[source_node]
                            max_destination_node_for_bag[str(bag)] = max_node
                            max_weights[i] = max_destination_node_for_source[source_node]
            max_weights.requires_grad_(True)
            return max_weights, max_destination_node_for_bag, max_destination_node_for_source
            #return max_weights,  max_destination_node_for_source_no_weight, max_destination_node_for_source
            
        else:
            # Retrieve data from data object
            source_nodes, num_nodes = list(node_dict.keys()), data.num_nodes
            # Tensor for saving the max destination weight for each source node. 
            # The size of the tensor is the total number of nodes
            max_weights =  torch.zeros(num_nodes, 1)
            # dictionary of max destination nodes (keys are source nodes)
            max_destination_node_for_source = {}
            for source_node in source_nodes:
                #Get the subset of parameters using PyTorch indexing
                weights_of_source = weights[node_dict[source_node]].squeeze(-1)
                max_node = node_dict[source_node][torch.argmax(weights_of_source).item()]
                max_destination_node_for_source[source_node] = max_node
                max_weights[source_node] = weights[max_node]
            max_weights.requires_grad_(True)
        return max_weights, max_destination_node_for_source, max_destination_node_for_source


class Score(nn.Module):
    def __init__(self, weights, COMPLEX):
        super(Score, self).__init__()
        # if COMPLEX == True, is complex case 
        self.COMPLEX = COMPLEX
        # Learnable weights
        self.input = InputLayer(weights)
        # Output layer
        self.output = OutputLayer()
        # Linear layer for features
        #self.LinearLayerAttri = nn.Linear(FEATURES_DIM, 1, bias=False)
        
    def frz_weights(self, indices):
        #     if i in indices:
        #         param.requires_grad_(False)
        #         print(param.requires_grad)
        # for i, param in enumerate(self.input.weights):
        #     print(i, param, param.requires_grad)
        for i in indices:
            self.input.weights[i].requires_grad = False


    def forward(self, data: Data, node_dict, BAGS):
        x = self.input()
        features = data.x.type(torch.FloatTensor)
        #linear_features = self.LinearLayerAttri(features)
        x, max_destination_node_for_bag, max_destination_node_for_source = self.output(x, data, node_dict, BAGS, self.COMPLEX, features)

        # # Linear layer multiplied for the features of each node (only for the complex case)
        # if self.COMPLEX:
        #     linear_features = self.LinearLayerAttri(features)
        #     # Multiplication between features of each source node for the max weight
        #     x = torch.mul(linear_features, x)
        return x, max_destination_node_for_bag, max_destination_node_for_source







'''
MPGNN
'''
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
                h = F.relu(self.conv1(self.metapath[layer_index], x, edge_index, edge_type))
            else:
                h = F.relu(self.conv2(self.metapath[layer_index], h, edge_index, edge_type))
        h = self.LinearLayer(h)
        #return F.log_softmax(x, dim=1)
        return h
    

class MPNetm(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_rel, output_dim, ll_output_dim, n_metapaths, metapaths):
        super().__init__()
        self.n_metapaths = n_metapaths
        self.metapaths = metapaths

        self.layers_list = torch.nn.ModuleList()
        
        for i in range(0, len(metapaths)):
            convs = torch.nn.ModuleList()
            convs.append(CustomRGCNConv(input_dim, hidden_dim, num_rel, flow='target_to_source'))
            for j in range(0, len(metapaths[i])-1):
                convs.append(CustomRGCNConv(hidden_dim, hidden_dim, num_rel, flow='target_to_source'))
            self.layers_list.append(convs)
        
        self.fc1 = torch.nn.Linear(hidden_dim * len(metapaths), hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, ll_output_dim)
        self.log_softmax = torch.nn.LogSoftmax(dim=1)
        #self.softmax = torch.nn.Softmax(dim=1)

        #self.dropout1 = nn.Dropout(0.5)
        #self.dropout2 = nn.Dropout(0.5) #togli p

    def forward(self, x, edge_index, edge_type):
        
        embeddings = []
        for i in range(0, len(self.metapaths)):
            for layer_index in range(0, len(self.metapaths[i])):
                if layer_index == 0:
                    h = F.relu(self.layers_list[i][layer_index](self.metapaths[i][layer_index], x, edge_index, edge_type))
                    #h = self.dropout1(h)
                else:
                    h = F.relu(self.layers_list[i][layer_index](self.metapaths[i][layer_index], h, edge_index, edge_type))
                    #h = self.dropout2(h)
            embeddings.append(h)

        #for e in embeddings:
        #    sum += e
        #embeddings.append(sum)
        concatenated_embedding = torch.cat(embeddings, dim=1)

        #print(concatenated_embedding)
        h = F.relu(self.fc1(concatenated_embedding))
        #h = F.relu(self.fc2(h))
        h = self.fc2(h)
        h = self.log_softmax(h)
        return h




'''
            # Retrieve data from data object
            source_nodes, num_nodes = list(node_dict.keys()), data.num_nodes
            # Tensor for saving the max destination weight for each source node. 
            # The size of the tensor is the total number of nodes
            max_weights =  torch.zeros(num_nodes, 1)
            # dictionary of max destination nodes (keys are source nodes)
            max_destination_node_for_source = {}
            for source_node in source_nodes:
                #Get the subset of parameters using PyTorch indexing
                weights_of_source = weights[node_dict[source_node]].squeeze(-1)
                #weights_of_source = [weights[i] for i in node_dict[source_node]]
                # Concatenate the subset of parameters into a new tensor
                # retrieve weights of nodes connected to source_node
                #weights_of_source = weights[node_dict[source_node]].squeeze(-1)
                # retrive the index of the max weight (may be more than one, in this case I use one of the 2)
                #index_max_destination_node = random.choice((weights_of_source == torch.max(weights_of_source)).nonzero(as_tuple=True)[0].tolist())
                #index_max_destination_node = weights_of_source.index(max(weights_of_source))
                # retrive the node with this weight and put in a dictionary
                max_node = node_dict[source_node][torch.argmax(weights_of_source).item()]
                max_destination_node_for_source[source_node] = max_node
                # max_destination_node_for_source[source_node] = torch.mul(weights[max_node],self.LinearLayerAttri(feat[source_node]))
                # update max_weight tensor

                #max_weights[source_node] = torch.max(weights[node_dict[source_node]].squeeze(-1))
                max_weights[source_node] = weights[max_node]
            max_weights.requires_grad_(True)
                #max_weights[source_node] = torch.mul(weights[max_node],self.LinearLayerAttri(feat[source_node]))
'''
'''
# retrieve data from data object
            bags = data.bags
            # Tensor to save the max destination weight for each bag.
            # The size of the tensor is the total number of bags
            max_weights = torch.zeros(len(bags), 1)

            # dictionary of max destination nodes (keys are bags as strings since it is not possible to have
            # lists as keys of a dictionary)
            max_destination_node_for_bag = {}
            # dictionary of max destination nodes (keys are source nodes)
            max_destination_node_for_source = {}
            # max destination nodes
            max_destination_nodes = []
            for i in range(0, len(bags)):
                max_weight_for_current_bag = -10
                for source_node in bags[i]:
                    if source_node in node_dict:
                        # retrieve weights of nodes connected to source_node
                        weights_of_source = weights[node_dict[source_node]].squeeze(-1)
                        #weights_of_source = [weights[i] for i in node_dict[source_node]]
                        # If we are in the complex case, multiply with the linear layer of the features
                        if COMPLEX:
                            weights_of_source = torch.mul(weights_of_source, self.LinearLayerAttri(feat[source_node]))
                            #weights_of_source = [i * self.LinearLayerAttri(feat[source_node]) for i in weights_of_source]
                        # retrive the index of the max weight (may be more than one, in this case I use one of them)
                        #index_max_destination_node = weights_of_source.index(max(weights_of_source))
                        #index_max_destination_node = random.choice((weights_of_source == torch.max(weights_of_source)).nonzero(as_tuple=True)[0].tolist())
                        # retrive the node with this weight and put in list
                        max_node = node_dict[source_node][torch.argmax(weights_of_source).item()]   
                        # retrieve the max weight for the source_node in the bag
                        #max_destination_node_for_source[source_node] = torch.mul(weights[max_node], torch.dot(torch.tensor([0., 0., 1., 0., 0., 0.]), feat[source_node]))
                        max_destination_node_for_source[source_node] = torch.mul(weights[max_node], self.LinearLayerAttri(feat[source_node]))
                        if max_node not in max_destination_nodes: max_destination_nodes.append(max_node)
                        # put in max_node_for_current_bag the max node so far for this bag
                        if max_destination_node_for_source[source_node] > max_weight_for_current_bag: 
                            max_weight_for_current_bag = max_destination_node_for_source[source_node]
                            max_destination_node_for_bag[str(bags[i])] = max_node
                            max_weights[i] = max_destination_node_for_source[source_node]
            max_weights.requires_grad_(True)'''


