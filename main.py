from model import *
from utils import *
from torch_geometric.loader import DataLoader
from torch_geometric.loader import ClusterData, ClusterLoader, NeighborSampler
import torch.nn.functional as F

import pandas as pd
import numpy as np
from tqdm import tqdm
import random
import seaborn as sns
from mlxtend.plotting import plot_confusion_matrix
import pickle
import os
from sklearn.metrics import f1_score

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from functools import partial
import multiprocess as mp
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans

from mpi4py import MPI

seed= 10
torch.manual_seed(seed)


def masked_edge_index(edge_index, edge_mask):
    if isinstance(edge_index, Tensor):
        return edge_index[:, edge_mask]
    else:
        return print('Error')

def one_hot_encoding(l):
    label_types = torch.unique(l).tolist()
    new_labels = []
    for i in range(0, len(l)):
        tmp = []
        for j in range(0, len(label_types)):
            tmp.append(0.)
        tmp[l[i].item()] = 1.
        new_labels.append(tmp)
    return torch.tensor(new_labels)     

def load_files(node_file_path, links_file_path, label_file_path, embedding_file_path, dataset):
    colors = pd.read_csv(node_file_path, sep='\t', header = None)
    colors = colors.dropna(axis=1,how='all')
    labels = pd.read_csv(label_file_path, sep='\t', header = None)
    links = pd.read_csv(links_file_path, sep='\t', header = None)
    labels.rename(columns = {0: 'node', 1: 'label'}, inplace = True)
    source_nodes_with_labels = labels['node'].values.tolist()
    labels = torch.tensor(labels['label'].values)
    colors.rename(columns = {0: 'node', 1: 'color'}, inplace = True)
    links.rename(columns = {0: 'node_1', 1: 'relation_type', 2: 'node_2'}, inplace = True)
    if dataset == 'complex' or dataset == 'simple':
        embedding = pd.read_csv(embedding_file_path, sep='\t', header = None)
        embedding_number = len(embedding.columns)-2
        if embedding_number == 3:
            embedding.rename(columns = {0: 'index', 1: 'second embedding', 2: 'first embedding', 3: 'labels'}, inplace = True)
        elif embedding_number == 4:
            embedding.rename(columns = {0: 'index', 1: 'third embedding', 2: 'second embedding', 3: 'first embedding', 4: 'labels'}, inplace = True)
        elif embedding_number == 5:
            embedding.rename(columns = {0: 'index', 1: 'fourth embedding', 2: 'third embedding', 3: 'second embedding', 4: 'first_embdding', 5: 'labels'}, inplace = True)
        elif embedding_number == 2:
            embedding.rename(columns = {0: 'index', 1: 'first embedding', 2: 'labels'}, inplace = True)
        return labels, colors, links, embedding
    else:
        labels_multi  = one_hot_encoding(labels)
        # for i in range(0, len(labels)):
        #     if labels[i].item() == 0:
        #         labels[i] = 1
        #     else:
        #         labels[i] = 0
        return labels, colors, links, source_nodes_with_labels, labels_multi

# def splitting_node_and_labels(lab, feat, src, dataset):
#     if dataset == 'complex' or dataset == 'simple':
#         node_idx = torch.tensor(feat['node'].values)
#     else:
#         node_idx = torch.tensor(src)
#     train_split = int(len(node_idx)*0.8)
#     test_split = len(node_idx) - train_split
#     train_idx = node_idx[:train_split]
#     test_idx = node_idx[-test_split:]

#     train_y = lab[:train_split]
#     test_y = lab[-test_split:]
#     return node_idx, train_idx, train_y, test_idx, test_y

def splitting_node_and_labels(lab, feat, src, dataset):
    if dataset == 'complex' or dataset == 'simple' or dataset == 'synthetic_multiple':
        node_idx = list(feat['node'].values)
    else:
        node_idx = src.copy()

    train_idx,test_idx,train_y,test_y = train_test_split(node_idx, lab,
                                                            random_state=415,#415,
                                                            stratify=lab, 
                                                            test_size=0.2)
    print(train_y.tolist().count(0),train_y.tolist().count(1))
    print(test_y.tolist().count(0), test_y.tolist().count(1))
    
    return torch.tensor(node_idx), train_idx, train_y, test_idx, test_y

def get_node_features(colors):
    node_features = pd.get_dummies(colors)
    
    node_features.drop(["node"], axis=1, inplace=True)
    
    x = node_features.to_numpy().astype(np.float32)
    x = np.flip(x, 1).copy()
    x = torch.from_numpy(x) 
    return x

def get_edge_index_and_type_no_reverse(links):
    edge_index = links.drop(['relation_type'], axis=1)
    edge_index = torch.tensor([list(edge_index['node_1'].values), list(edge_index['node_2'].values)])
    
    edge_type = links['relation_type']
    edge_type = torch.tensor(edge_type)
    return edge_index, edge_type

def get_dest_labels(node_dict, data):
    bags = data.bags
    bag_labels = data.bag_labels.tolist()
    dest_labels = {}
    for src, dest_list in node_dict.items():
        for dest in dest_list:
            if dest not in dest_labels:
                dest_labels[dest] = []
            for i, bag in enumerate(bags):
                if src in bag or dest in bag:
                    dest_labels[dest].append(bag_labels[i])
    return dest_labels

def create_edge_dictionary(data, relation, source_nodes_mask, BAGS):
    '''
        edge_dictionary is a dictionary where keys are source nodes and values are destination
        nodes, connected with the respective source node via relation 'relation'.
        The source nodes are in source_nodes_mask list
    '''
    edge_dictionary = {}
    edge_index = masked_edge_index(data.edge_index, data.edge_type == relation)
    
    for index in source_nodes_mask:
        if index in edge_index[0].tolist(): edge_dictionary[index] = []
        
    for src, dst in zip(edge_index[0], edge_index[1]):
        if src.item() in source_nodes_mask:
            edge_dictionary[src.item()].append(dst.item())
    
    edge_dictionary_copy = edge_dictionary.copy()
    for src, dst in edge_dictionary.items():
        if not dst:
            del edge_dictionary_copy[src]
    
    '''
        destination_dictionary is a dictionary where keys are destination nodes and values are the labels of their 
        specific source nodes. It is used to initialize the weights.
    '''
    if not BAGS:
        destination_dictionary = {}
        edge_index = masked_edge_index(data.edge_index, data.edge_type == relation)
            
        for src, dst in zip(edge_index[0], edge_index[1]):
            if src.item() in source_nodes_mask and dst.item() not in destination_dictionary: destination_dictionary[dst.item()] = []
        for src, dst in zip(edge_index[0], edge_index[1]):
            if src.item() in source_nodes_mask:
                destination_dictionary[dst.item()].append(data.labels[src.item()].item())
        return edge_dictionary_copy, destination_dictionary
    
    else:       
        destination_bag_dictionary = {}
        tmp_dict = {}
        for i in range(0, len(data.bags)):
            for j in range(0, len(data.bags[i])):
                if data.bags[i][j] not in tmp_dict: tmp_dict[data.bags[i][j]] = []
                tmp_dict[data.bags[i][j]].append(data.bag_labels[i].item())

        for src, dst in zip(edge_index[0], edge_index[1]):
            if src.item() in tmp_dict:
                if dst.item() not in destination_bag_dictionary: destination_bag_dictionary[dst.item()] = []
                destination_bag_dictionary[dst.item()].extend(tmp_dict[src.item()])
        return edge_dictionary_copy, destination_bag_dictionary
    

def create_destination_labels_dictionary(data, relation, source_nodes_mask):
    '''
        dictionary is a dictionary where keys are destination nodes and values are the labels of their 
        specific source nodes. It is used to initialize the weights.
    '''
    destination_dictionary = {}
    edge_index = masked_edge_index(data.edge_index, data.edge_type == relation)
        
    for src, dst in zip(edge_index[0], edge_index[1]):
        if src.item() in source_nodes_mask and dst.item() not in destination_dictionary: destination_dictionary[dst.item()] = []

    for src, dst in zip(edge_index[0], edge_index[1]):
        if src.item() in source_nodes_mask:
            destination_dictionary[dst.item()].append(data.labels[src.item()].item())
    return destination_dictionary

def clean_dictionaries(data, edg_dict, dest_dict, mod):
    edge_dictionary_copy, dest_dictionary_copy = edg_dict.copy(), dest_dict.copy()
    #print(mod.output.LinearLayerAttri.weight[0])
    for key, value in edg_dict.items():
        if torch.dot(data.x[key], mod.output.LinearLayerAttri.weight[0]).item() < 0.01:
            dsts = edge_dictionary_copy[key]
            for destination in dsts:
                if 0 in dest_dictionary_copy[destination]:
                    dest_dictionary_copy[destination].remove(0)
            del edge_dictionary_copy[key]
        #if torch.dot(data.x[key], test).item() < 0.1:
            # for dest in value:
            #     if max(dest_dict[dest]) == 1:
            #         edge_dictionary_copy[key].remove(dest)
            #         dest_dictionary_copy[dest].remove(0)
            #         if not edge_dictionary_copy[key]:
            #             del edge_dictionary_copy[key]
            #         if not dest_dictionary_copy[dest]:
            #             del dest_dictionary_copy[dest]

            
    return edge_dictionary_copy, dest_dictionary_copy

def initialize_weights(data, destination_dictionary, BAGS):
    '''
        Initialize weights for destination nodes. For each destination node the initialized weight is 
        the minimum label among its source nodes' labels. 
        If a destination node is not taken into account his weight is simply a random between 0 and 1.
    '''
    weights = torch.Tensor(data.num_nodes)
    # if BAGS:
    #     start = 0.0
    #     end = 1.2
    #     for idx in range(0, len(weights)):
    #         weights[idx] = random.uniform(start, end)
    start = - 0.3
    end = 0.3
    for key, values in destination_dictionary.items():
        weights[key] = abs(min(values) + random.uniform(start, end))
        #weights[key] = random.uniform(0., 1.2)

    return weights

def reinitialize_weights(data, destination_dictionary, previous_weights, destination_nodes_with_freezed_weights, BAGS):
    weights = torch.Tensor(data.num_nodes)
    # if BAGS:
    #     start = 0.0
    #     end = 1.2
    #     for idx in range(0, len(weights)):
    #         weights[idx] = random.uniform(start, end)
    # else:
    start = -0.4
    end = 0.4
    for key, values in destination_dictionary.items():
        if key in destination_nodes_with_freezed_weights:
            weights[key] = previous_weights[key]
        else:
            #weights[key] = abs(min(values) + random.uniform(start, end))
            weights[key] = random.uniform(0., 1.)
            #weights[key] = np.random.normal(mu, sigma)
    return weights

def get_model(weights):
    return Score(weights, COMPLEX)

def get_optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=0.1)

def get_loss():
    return nn.MSELoss(reduction='mean')

def get_loss_per_node():
    return nn.MSELoss(reduction='none')

def retrieve_destinations_low_loss(max_destination_node_dict, loss_per_node, source_nodes_mask):
    '''
        Function that output a list of destination nodes. Those destination nodes are selected to be 
        the ones whom their source nodes (or source bags) have a loss lower than a threshold.
        max_destination_node_dict is a dict of source nodes only when there are no bags, otherwise 
        is a dict of bags
    '''
    max_destinations = []
    index = 0
    for key, value in max_destination_node_dict.items():
        if loss_per_node[index] < 0.0001 and value not in max_destinations:
            max_destinations.append(value)
        index+=1
    return max_destinations

def create_bags(edg_dictionary, dest_dictionary, data):
    print('creo bags')
    bag = []
    labels = []
    for key in edg_dictionary.keys():
        list = []
        for value in edg_dictionary[key]:
            if min(dest_dictionary[value]) > 0.9:
                    list.append(value)
            else: 
                if [value] not in bag:
                    bag.append([value])
                    labels.append(0)
        if list:
            bag.append(list)
            labels.append(1)

    #eliminate duplicates
    new_bag = []
    new_labels = []
    for idx in range(0, len(bag)):
        if bag[idx] not in new_bag:
            new_bag.append(bag[idx])
            new_labels.append(labels[idx])
        
    data.bags = new_bag
    data.bag_labels = torch.Tensor(new_labels).unsqueeze(-1)

def clean_bags_for_relation_type(data, edge_dictionary): 
    to_keep = []
    to_keep_labels = []
    c = 0
    for bag in data.bags:
        tmp = []
        for node in bag:
            if node in edge_dictionary:
                tmp.append(node)
        if tmp:
            to_keep.append(tmp)
            to_keep_labels.append(data.bag_labels[c])
        c = c + 1
    #data.bags = to_keep
    #data.bag_labels = torch.Tensor(to_keep_labels).unsqueeze(-1)
    return to_keep, torch.Tensor(to_keep_labels).unsqueeze(-1)

def relabel_nodes_into_bags(predictions_for_each_restart, data, mod):
    # if a node into the bags is predicted as 1 at least one time, then his label willl be 1 otherwise 0
    data.labels = torch.zeros(data.num_nodes, 1)
    for k, v in predictions_for_each_restart.items():
        data.labels[k] = 0
        for w in v:
            if w > 0.9:
                data.labels[k] = 1
    # delete from source mask, nodes that have a color value similar to zero (a zero in the linear layer)
    
    src = list(predictions_for_each_restart.keys())
    # print('before: ', len(src))
    # src_copy = src
    # for n in src:
    #     dot_prod = torch.dot(data.x[n], mod.output.LinearLayerAttri.weight[0])
    #     if dot_prod.item() < 0.1:
    #         src_copy.remove(n)
    # print('after: ', len(src_copy))
    return src
    
def freeze_weights(model, destination_nodes_with_freezed_weights, previous_weights):
    with torch.no_grad():
        for idx in range(0, len(model.weights())):
            if idx in destination_nodes_with_freezed_weights: model.weights()[idx].requires_grad=False# = previous_weights[idx]

def train(data, edge_dictionary, model, optimizer, criterion, source_nodes_mask, criterion_per_node, destination_nodes_with_freezed_weights, previous_weights, grad_mask, BAGS):
    model.train()
    optimizer.zero_grad()
    predictions, max_destination_node_for_bag, max_destination_node_for_source = model(data, edge_dictionary, BAGS)
    if not BAGS:
        labels = data.labels
        predictions, labels = predictions[source_nodes_mask].to(torch.float32), labels[source_nodes_mask].to(torch.float32)
    else:
        labels = data.bag_labels
    loss = criterion(predictions, labels)
    loss_per_node = criterion_per_node(predictions, labels)
    loss.backward()
    # freeze weights (multiply the grad tensor with a mask of 0s and 1s).
    if destination_nodes_with_freezed_weights:
        model.input.weights.grad = model.input.weights.grad*grad_mask
    optimizer.step()

    with torch.no_grad():
        model.input.weights[:] = torch.clamp(model.input.weights, min = 0.0)
        model.output.LinearLayerAttri.weight[:] = torch.clamp(model.output.LinearLayerAttri.weight, min = 0.0)
    # with torch.no_grad():
    #     for i in range(0, len(model.input.weights)):
    #         model.input.weights[i].clamp_min_(0)
    #     #model.input.weights[:] = torch.clamp(model.input.weights, min = 0.0)
    #     model.output.LinearLayerAttri.weight[:] = torch.clamp(model.output.LinearLayerAttri.weight, min = 0.0)

    # if we have freezed weights of some destination node in the previous iteration 
    # than we restore the weights as they were before
    # if destination_nodes_with_freezed_weights:
    #     model.frz_weights(destination_nodes_with_freezed_weights)

    #     for i, param in enumerate(model.weights.parameters()):
    #         if i in destination_nodes_with_freezed_weights:
    #             assert not param.requires_grad, f"Weight at index {i} is not frozen"
        #freeze_weights(model, destination_nodes_with_freezed_weights, previous_weights)
    return loss, max_destination_node_for_source, loss_per_node, max_destination_node_for_bag, predictions

def retrain(data, source_nodes_mask, relation, BAGS):
    current_loss = 100

    if not source_nodes_mask:
        first = True
    else:
        first = False

    # All source nodes with relation type 'relation' are considered (first iteration)
    if first:
        source_nodes_mask = masked_edge_index(data.edge_index, data.edge_type == relation)
        source_nodes_mask = torch.unique(source_nodes_mask[0]).tolist()# = list(np.array(torch.unique(source_nodes_mask[0])))

    # Create dictionary of source and destinaiton nodes connected with a specific relation type
    edge_dictionary = create_edge_dictionary(data, relation, source_nodes_mask, BAGS=False)

    # Create dictionary of destination nodes and their specific source labels
    destination_dictionary = create_destination_labels_dictionary(data, relation, source_nodes_mask)

    # Initialize weights
    weights = initialize_weights(data, destination_dictionary, BAGS=False)

    # Retrieve loss
    criterion = get_loss()
    criterion_per_node = get_loss_per_node()

    # In each restart, the weights of the good destination nodes are freezed for the next restarts
    destination_nodes_with_freezed_weights = []
    RESTARTS=1
    for i in range(0, RESTARTS):
        # Retrieve model
        model = get_model(weights)
        # Retrieve optimizer
        optimizer = get_optimizer(model)
        # Training
        EPOCHS = 2
        for epoch in tqdm(range(0, EPOCHS)):
            loss, max_destination_node_for_source, loss_per_node, max_destination_node_for_bag, predictions, max_destination_for_each_source = train(data, edge_dictionary, model, optimizer, criterion, source_nodes_mask, criterion_per_node, destination_nodes_with_freezed_weights, weights, BAGS)
        # if in this restart the loss drops with respect to the previous, then I freeze weights
        if loss < current_loss:
            print('RESTART ', i, ' LOSS: ', loss.item()) 
            # retrieve destination nodes who give a low loss to their source nodes
            destination_nodes_with_freezed_weights = retrieve_destinations_low_loss(max_destination_node_for_source, loss_per_node, source_nodes_mask)
            # reinitialize weights but the ones in destination_nodes_with_freezed_weights list
            weights = reinitialize_weights(data, destination_dictionary, model.weights().detach(), destination_nodes_with_freezed_weights, BAGS=False)
            current_loss = loss
        else:
            print('PREVIOUS LOSS WAS BETTER SO RESTART AGAIN: ', current_loss.item()) 
        # for n, p in model.named_parameters():
        #     print(n, p)
    return destination_nodes_with_freezed_weights, model

def score_relation_parallel(data, relation, source_nodes_mask):
    if not source_nodes_mask:
        first = True
    else:
        first = False
    # All source nodes with relation type 'relation' are considered (first iteration)
    if first:
        source_nodes_mask = masked_edge_index(data.edge_index, data.edge_type == relation)
        source_nodes_mask = torch.unique(source_nodes_mask[0]).tolist()
    # Create dictionary of source and destinaiton nodes connected with a specific relation type
    edge_dictionary, destination_dictionary = create_edge_dictionary(data, relation, source_nodes_mask, BAGS=False)
    # Create dictionary of destination nodes and their specific source labels
    #destination_dictionary = create_destination_labels_dictionary(data, relation, source_nodes_mask)

    # Initialize weights
    weights = initialize_weights(data, destination_dictionary, BAGS=False)
    # Retrieve model
    model = get_model(weights)

    # Retrieve optimizer
    optimizer = get_optimizer(model)

    # Retrieve loss
    criterion = get_loss()
    criterion_per_node = get_loss_per_node()

    # Training
    EPOCHS = 100
    for epoch in tqdm(range(0, EPOCHS)):
        loss, max_destination_node_for_source, loss_per_node, max_destination_node_for_bag, predictions = train(data, edge_dictionary, model, optimizer, criterion, source_nodes_mask, criterion_per_node, [], weights, torch.tensor(0), BAGS=False)

    return relation, loss.item(), edge_dictionary, destination_dictionary
                    
def score_relations(data, source_nodes_mask, BAGS):
    # Variables
    best_loss = 100

    if not source_nodes_mask:
        first = True
    else:
        first = False
    relations = torch.unique(data.edge_type).tolist()
    for relation in relations:
        # All source nodes with relation type 'relation' are considered (first iteration)
        if first:
            source_nodes_mask = masked_edge_index(data.edge_index, data.edge_type == relation)
            source_nodes_mask = torch.unique(source_nodes_mask[0]).tolist()
        # Create dictionary of source and destinaiton nodes connected with a specific relation type
        edge_dictionary, destination_dictionary = create_edge_dictionary(data, relation, source_nodes_mask, BAGS=False)

        # Create dictionary of destination nodes and their specific source labels
        #destination_dictionary = create_destination_labels_dictionary(data, relation, source_nodes_mask)

        # Initialize weights
        weights = initialize_weights(data, destination_dictionary, BAGS=False)

        # Retrieve model
        model = get_model(weights)

        # Retrieve optimizer
        optimizer = get_optimizer(model)

        # Retrieve loss
        criterion = get_loss()
        criterion_per_node = get_loss_per_node()

        # Training
        EPOCHS = 100
        for epoch in tqdm(range(0, EPOCHS)):
            loss, max_destination_node_for_source, loss_per_node, max_destination_node_for_bag, predictions = train(data, edge_dictionary, model, optimizer, criterion, source_nodes_mask, criterion_per_node, [], weights, [], BAGS) 
        print('Relation: ', relation, loss.item())

        # Take relation with lowest loss
        if loss < best_loss:
            best_loss = loss
            best_max_destination_node_for_source = max_destination_node_for_source
            best_edge_dictionary = edge_dictionary
            best_relation = relation
            best_model = model
            best_destination_dictionary = destination_dictionary
            best_predictions = predictions
            best_source_nodes_mask = source_nodes_mask

    return best_relation, best_edge_dictionary, best_model, best_destination_dictionary, best_max_destination_node_for_source, loss_per_node, best_predictions

def retrain_bags(data, relation, best_pred_for_each_restart, BAGS):
    current_loss = 100
    # source nodes are all the nodes in the bags
    source_nodes_mask = []
    for bag in data.bags:
        for elm in bag:
            if elm not in source_nodes_mask: source_nodes_mask.append(elm)
    # Create dictionary of source and destinaiton nodes connected with a specific relation type
    edge_dictionary, destination_dictionary = create_edge_dictionary(data, relation, source_nodes_mask, BAGS=True)
    # Create dictionary of destination nodes and their specific source labels
    #estination_dictionary = create_destination_labels_dictionary(data, relation, source_nodes_mask)

    # Initialize weights
    weights = initialize_weights(data, destination_dictionary, BAGS=True)

    # Retrieve loss
    criterion = get_loss()
    criterion_per_node = get_loss_per_node()

    destination_nodes_with_freezed_weights = []
    RESTARTS=10
    EPOCHS=50
    for i in tqdm(range(0, RESTARTS)):
        # Retrieve model
        model = get_model(weights)
        # Retrieve optimizer
        optimizer = get_optimizer(model)
        # Training
        for epoch in tqdm(range(0, EPOCHS)):
            loss, max_destination_node_for_source, loss_per_bag, max_destination_node_for_bag, predictions = train(data, edge_dictionary, model, optimizer, criterion, source_nodes_mask, criterion_per_node, destination_nodes_with_freezed_weights, weights, torch.tensor(0), BAGS)
        # save all predictions
        for key, value in max_destination_node_for_source.items():
            best_pred_for_each_restart[key].append(value.item())
        #print('RESTART ', i, ' LOSS: ', loss.item()) 
        destination_nodes_with_freezed_weights = []
        weights = reinitialize_weights(data, destination_dictionary, model.input.weights.detach(), destination_nodes_with_freezed_weights, BAGS=True)
    return best_pred_for_each_restart
    #return max_destination_node_for_source, model, max_destination_node_for_bag, loss_per_bag, predictions, destination_nodes_with_freezed_weights, edge_dictionary, best_pred_for_each_restart

def score_relation_bags_parallel(data, relation):
    R, current_loss = 0, 100
    # Create a mask for source nodes which are all the nodes into bags
    source_nodes_mask = []
    for bag in data.bags:
        for elm in bag:
            if elm not in source_nodes_mask: source_nodes_mask.append(elm)
    current_loss = 100
    # Create dictionary of source and destinaiton nodes connected with a specific relation type
    edge_dictionary, destination_dictionary = create_edge_dictionary(data, relation, source_nodes_mask, BAGS=True)
    # Create bags and labels for this specific relation. It is possible that a source node in a bag
    # has no any connection with a specific relation type and so it is not considered
    bags, bag_labels = clean_bags_for_relation_type(data, edge_dictionary)
    # Initialize weights
    weights = initialize_weights(data, destination_dictionary, BAGS=True)
    grad_mask = torch.ones(len(weights), 1)
    # Retrieve loss
    criterion = get_loss()
    criterion_per_node = get_loss_per_node()
    # For each restart save predictions
    predictions_for_each_restart = {}
    # In each restart, the weights of the good destination nodes are freezed for the next restarts
    destination_nodes_with_freezed_weights = []
    
    while (R<1):
        # Retrieve model
        model = get_model(weights)
         # Retrieve optimizer
        optimizer = get_optimizer(model)
        # Training
        EPOCHS=50
        for epoch in tqdm(range(0, EPOCHS)):
            loss, max_destination_node_for_source, loss_per_bag, max_destination_node_for_bag, predictions = train(data, edge_dictionary, model, optimizer, criterion, source_nodes_mask, criterion_per_node, destination_nodes_with_freezed_weights, weights, grad_mask, BAGS=True)
        # save predictions
        for key, value in max_destination_node_for_source.items():
            if key not in predictions_for_each_restart:
                predictions_for_each_restart[key] = []
            predictions_for_each_restart[key].append(value.item())
        if loss.item() < current_loss:
            destination_nodes_with_freezed_weights = retrieve_destinations_low_loss(max_destination_node_for_bag, loss_per_bag, source_nodes_mask)
            current_loss = loss
            R=0
            #print('\n Relation ', relation, ' loss: ', current_loss.item())
        else:
            #print('\n Relation ', relation, ' end')
            R+=1
        for node in destination_nodes_with_freezed_weights:
            grad_mask[node] = 0
        weights = reinitialize_weights(data, destination_dictionary, model.input.weights.detach(), destination_nodes_with_freezed_weights, BAGS=False)
    #print('\nRelation ', relation, ' loss: ', current_loss.item())
    return relation, loss.item(), model, predictions_for_each_restart

def score_relation_bags_with_restarts(data, BAGS, VAL):
    best_loss = 100

    if VAL:
        relations = torch.unique(data.edge_type).tolist()
        #relations = [1]
    else: 
        #relations = [1]
        relations = torch.unique(data.edge_type).tolist()
        
    # Save the original bags and labels
    original_bags, original_labels = data.bags, data.bag_labels
    # source nodes are all the nodes in the bags
    source_nodes_mask = []
    for bag in data.bags:
        for elm in bag:
            if elm not in source_nodes_mask: source_nodes_mask.append(elm)


    for relation in relations:
        R = 0
        current_loss = 100
        print('\tBAG RELATION ', relation)
        # Create dictionary of source and destinaiton nodes connected with a specific relation type
        edge_dictionary, destination_dictionary, destination_bag_dictionary = create_edge_dictionary(data, relation, source_nodes_mask, BAGS=True)
        # Create dictionary of destination nodes and their specific source labels
        #destination_dictionary = create_destination_labels_dictionary(data, relation, source_nodes_mask)
        # Create bags and labels for this specific relation. It is possible that a source node in a bag
        # has no any connection with a specific relation type and so it is not considered
        clean_bags_for_relation_type(data, edge_dictionary)
        # Initialize weights
        weights = initialize_weights(data, destination_bag_dictionary, BAGS=True)

        # Retrieve loss
        criterion = get_loss()
        criterion_per_node = get_loss_per_node()

        predictions_for_each_restart = {}
        # In each restart, the weights of the good destination nodes are freezed for the next restarts
        destination_nodes_with_freezed_weights = []
        RESTARTS=0
        while(R < 1):
            # Retrieve model
            model = get_model(weights)
            # if destination_nodes_with_freezed_weights:
            #     model.frz_weights(destination_nodes_with_freezed_weights)
            # Retrieve optimizer
            optimizer = get_optimizer(model)
            # Training
            EPOCHS=50
            for epoch in tqdm(range(0, EPOCHS)):
                loss, max_destination_node_for_source, loss_per_bag, max_destination_node_for_bag, predictions = train(data, edge_dictionary, model, optimizer, criterion, source_nodes_mask, criterion_per_node, destination_nodes_with_freezed_weights, weights, BAGS)
                #curve.append(loss.item())
            # f = np.array(f)
            # curve = np.array(curve)
            # plt.plot(f, curve)
            # plt.show()
            # save predictions
            for key, value in max_destination_node_for_source.items():
                if key not in predictions_for_each_restart:
                    predictions_for_each_restart[key] = []
                predictions_for_each_restart[key].append(value.item())
            if loss < current_loss:
                print('RESTART ', RESTARTS, ' LOSS: ', loss.item()) 
                # retrieve destination nodes who give a low loss to their source nodes
                destination_nodes_with_freezed_weights = retrieve_destinations_low_loss(max_destination_node_for_bag, loss_per_bag, source_nodes_mask)
                print(len(destination_nodes_with_freezed_weights))
                # reinitialize weights but the ones in destination_nodes_with_freezed_weights list
            
                weights = reinitialize_weights(data, destination_bag_dictionary, torch.cat([p.data.view(-1) for p in model.input()]), destination_nodes_with_freezed_weights, BAGS=True)
                current_loss = loss
                R=0
            else:
                print('PREVIOUS LOSS WAS BETTER SO STOP: ', current_loss.item()) 
                R+=1
                weights = reinitialize_weights(data, destination_bag_dictionary, torch.cat([p.data.view(-1) for p in model.input()]), destination_nodes_with_freezed_weights, BAGS=True)
            
            
        print('Loss: ', current_loss.item())
        #print(model.output.LinearLayerAttri.weight[0])
        # Take lower loss
        if current_loss < best_loss:
            best_loss = current_loss
            best_max_destination_node_for_source = max_destination_node_for_source #
            best_edge_dictionary = edge_dictionary # 
            best_relation = relation # 
            best_model = model # 
            best_destination_dictionary = destination_dictionary # 
            best_max_destination_node_for_bag = max_destination_node_for_bag #
            best_bags, best_bag_labels = data.bags, data.bag_labels
            best_loss_per_node = loss_per_bag # 
            best_predictions = predictions #
            best_prediction_for_each_restart = predictions_for_each_restart #
        # Put the original bags and labels in data object
        data.bags, data.bag_labels = original_bags, original_labels
    # Save the bags and labels of the best relation
    data.bags, data.bag_labels = best_bags, best_bag_labels
    print('### Best loss is for relation: ', best_relation)
    return best_relation, best_model, best_max_destination_node_for_source, best_max_destination_node_for_bag, best_loss_per_node, best_predictions, best_prediction_for_each_restart, best_edge_dictionary, best_destination_dictionary

def negative_sampling(data):
    mask = []

    list_of_label_indices = list(range(0,len(data.labels.squeeze(-1).tolist())))
    for i in range(0, len(data.labels.squeeze(-1).tolist())):
        if data.labels.squeeze(-1).tolist()[i] == 1:
            mask.append(i)
    positive_samples = len(mask)
    count = 0
    while (count <= positive_samples):
        negative_sample = random.sample(list_of_label_indices, 1)[0]
        if data.labels.squeeze(-1).tolist()[negative_sample] == 0:
            mask.append(negative_sample)
            count += 1
            
    return mask

def save_confusion_matrix(node_mask, data, emb, index):
    prd = data.labels[node_mask].squeeze(-1).tolist()
    if index == 1:
        true = (torch.tensor(emb['first embedding'].tolist())[node_mask]).tolist()
    elif index == 2:
        true = (torch.tensor(emb['second embedding'].tolist())[node_mask]).tolist()
    elif index == 3:
        true = (torch.tensor(emb['third embedding'].tolist())[node_mask]).tolist()
    elif index == 4:
        true = (torch.tensor(emb['fourth embedding'].tolist())[node_mask]).tolist()
    cm = confusion_matrix(true, prd)
    print(cm)
    fig, ax = plot_confusion_matrix(conf_mat=cm, figsize=(6, 6), cmap=plt.cm.Greens)
    plt.title('Confusion matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(folder + 'confusion_matrix_iteration_'+str(index+1)+'.jpg')

def mpgnn_train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    #weight_loss = torch.tensor([1., 10.])
    out = model(data.x, data.edge_index, data.edge_type)
    #print(out)
    # for i in range(0, len(data.train_y)):
    #     print(out[data.train_idx].squeeze(-1)[i].item(), data.tra9#in_y[i].item())
    loss = F.nll_loss(out[data.train_idx].squeeze(-1), data.train_y)#, weight = weight_loss)
    #loss = F.cross_entropy(out[data.train_idx], data.train_y)
    loss.backward()
    optimizer.step()
    return float(loss)

@torch.no_grad()
def mpgnn_test(model, data):
    model.eval()
    pred = model(data.x, data.edge_index, data.edge_type)#.argmax(dim=-1)
    loss_test = F.nll_loss(pred[data.test_idx].squeeze(-1), data.test_y)
    
    train_predictions = torch.argmax(pred[data.train_idx], 1).tolist()
    test_predictions = torch.argmax(pred[data.test_idx], 1).tolist()
    train_y = data.train_y.tolist()
    test_y = data.test_y.tolist()
    # train_acc = (train_predictions == train_y).float().mean()
    # test_acc = (test_predictions == test_y).float().mean()
    f1_train = f1_score(train_predictions, train_y, average='micro')
    f1_test_macro = f1_score(test_predictions, test_y, average = 'macro')
    f1_test_micro = f1_score(test_predictions, test_y, average = 'micro')
    return f1_train, f1_test_micro, f1_test_macro,loss_test

def mpgnn_parallel(data_mpgnn, input_dim, hidden_dim, num_rel, output_dim, ll_output_dim, metapath):
    metapath=[0, 1, 2]
    mpgnn_model = MPNet(input_dim, hidden_dim, num_rel, output_dim, ll_output_dim, len(metapath), metapath)
    print(mpgnn_model)
    # for name, param in mpgnn_model.named_parameters():
    #     print(name, param, param.size())
    mpgnn_optimizer = torch.optim.Adam(mpgnn_model.parameters(), lr=0.01, weight_decay=0.0005)
    best_macro, best_micro = 0., 0.
    for epoch in tqdm(range(1, 100)):
        loss = mpgnn_train(mpgnn_model, mpgnn_optimizer, data_mpgnn)
        train_acc, f1_test_micro, f1_test_macro = mpgnn_test(mpgnn_model, data_mpgnn)
        if f1_test_macro > best_macro:
            best_macro = f1_test_micro
        if f1_test_micro > best_micro:
            best_micro = f1_test_micro
    return best_micro

def mpgnn_parallel_multiple(data_mpgnn, input_dim, hidden_dim, num_rel, output_dim, ll_output_dim, metapaths):
    #metapaths = [[2, 0]]#, [3, 1]]
    #metapaths = [[1, 4, 2, 0], [1, 0], [1, 5, 3, 0]]
    #metapaths = [[4, 3, 0], [1, 0], [0, 4, 2]]
    mpgnn_model = MPNetm(input_dim, hidden_dim, num_rel, output_dim, ll_output_dim, len(metapaths), metapaths)
    print(mpgnn_model)
    # for name, param in mpgnn_model.named_parameters():
    #     print(name, param, param.size())
    mpgnn_optimizer = torch.optim.Adam(mpgnn_model.parameters(), lr=0.01, weight_decay=0.0005)
    best_macro, best_micro = 0., 0.
    for epoch in range(1, 2000):
        loss = mpgnn_train(mpgnn_model, mpgnn_optimizer, data_mpgnn)
        if epoch % 10 == 0:
            train_acc, f1_test_micro, f1_test_macro,loss_test = mpgnn_test(mpgnn_model, data_mpgnn)
            print(epoch, "train loss %0.3f" % loss, "test loss %0.3f" % loss_test,
                  'train micro: %0.3f'% train_acc, 'test micro: %0.3f'% f1_test_micro)
            if f1_test_macro > best_macro:
                best_macro = f1_test_micro
            if f1_test_micro > best_micro:
                best_micro = f1_test_micro
    return best_micro

def main(node_file_path, link_file_path, label_file_path, embedding_file_path, metapath_length, pickle_filename, input_dim, hidden_dim, num_rel, output_dim, ll_output_dim, dataset):
    # Obtain true 0|1 labels for each node, feature matrix (1-hot encoding) and links among nodes
    if dataset == 'complex' or dataset == 'simple':
        sources = []
        true_labels, features, edges, embedding = load_files(node_file_path, link_file_path, label_file_path, embedding_file_path, dataset)
    else: 
        true_labels, features, edges, sources, labels_multi = load_files(node_file_path, link_file_path, label_file_path, embedding_file_path, dataset)
    # Get features' matrix
    x = get_node_features(features)
    # Get edge_index and types
    edge_index, edge_type = get_edge_index_and_type_no_reverse(edges)

    # Split data into train and test
    node_idx, train_idx, train_y, test_idx, test_y = splitting_node_and_labels(true_labels, features, sources, dataset)
    #node_idx, train_idx, train_y, test_idx, test_y = splitting_node_and_labels(labels_multi, features, sources, dataset)

    # Dataset for MPGNN
    data_mpgnn = Data()
    data_mpgnn.x = x
    data_mpgnn.edge_index = edge_index
    data_mpgnn.edge_type = edge_type
    data_mpgnn.train_idx = train_idx
    data_mpgnn.test_idx = test_idx
    data_mpgnn.train_y = train_y
    data_mpgnn.test_y = test_y
    data_mpgnn.num_nodes = node_idx.size(0)
    # Variables
    if sources:
        source_nodes_mask = sources
    else:
        source_nodes_mask = []
    metapath = []

    # Dataset for score function
    data = Data()
    data.x = x
    data.edge_index = edge_index
    data.edge_type = edge_type
    data.labels = true_labels
    data.labels = data.labels.unsqueeze(-1)
    data.num_nodes = x.size(0)
    data.bags = torch.empty(1)
    data.bag_labels = torch.empty(1)

    # All possible relations
    relations = torch.unique(data.edge_type).tolist()
    mp = []
    #mpgnn_f1_micro = mpgnn_parallel_multiple(data_mpgnn, input_dim, hidden_dim, num_rel, output_dim, ll_output_dim, mp)
    #print(mpgnn_f1_micro)

    # comm = MPI.COMM_WORLD
    # rank = comm.Get_rank()
    # size = comm.Get_size()

    # # numero totale di relazioni del grafo
    # num_relazioni = 30

    # # le relazioni vengono divise in parti uguali
    # chunk_size = num_relazioni // size
    # remainder = num_relazioni % size
    # print('chunk size: ', chunk_size, 'remainder: ', remainder)

    # # ogni processo figlio prende una parte diversa del grafo
    # if rank < remainder:
    #     start_idx = rank * (chunk_size + 1)
    #     end_idx = start_idx + chunk_size + 1
    # else:
    #     start_idx = rank * chunk_size + remainder
    #     end_idx = start_idx + chunk_size
    # print(f"Rank {rank}: start_idx={start_idx}, end_idx={end_idx}")

    # BLOCK = False
    # l = list(data.labels)
    # if BLOCK ==True:
    #     with open(pickle_filename, "rb") as f:
    #         edg_dictionary, dest_dictionary, data, mod = pickle.load(f)

    #     print(" \t\t ITERARION 3 ")
    #     # create bags and put them in data object
    #     VAL = False
    #     FRST = True
    #     create_bags(edg_dictionary, dest_dictionary, data, FRST)
    #     # return data, embedding
    #     l = list(data.bag_labels)
    #     # score each relation but this time on bags. The fuond relation will be the next one in the meta-path
    #     relation, trained_model, max_destination_nodes_source, max_destination_for_bag, loss_per_bag, pred, best_pred_for_each_restart, e_dictionary, d_dictionary = score_relation_bags_with_restarts(data, BAGS=True, VAL=VAL)
    #     mod = trained_model
        
    #     # retrain bags
    #     max_destination_nodes_source, trained_model, max_destination_for_bag, loss_per_bag, pred, destination_nodes_with_freezed_weights, edge_dictionary, max_destination_for_each_source, predictions_for_each_restart = retrain_bags(data, relation, best_pred_for_each_restart, BAGS=True)
    #     # relabel nodes into the bags
    #     source_nodes_mask = relabel_nodes_into_bags(predictions_for_each_restart, data, mod)
    #     #print(len(source_nodes_mask))
    #     # draw and save confusion matrix and scatter
    #     save_confusion_matrix(source_nodes_mask, data, embedding, 2)
    #     edg_dictionary  = create_edge_dictionary(data, relation, source_nodes_mask, BAGS=False)
    #     dest_dictionary = create_destination_labels_dictionary(data, relation, source_nodes_mask)
    #     new_edg_dictionary, new_dest_dictionary = clean_dictionaries(data, edg_dictionary, dest_dictionary, mod)
    #     edg_dictionary, dest_dictionary = new_edg_dictionary.copy(), new_dest_dictionary.copy()
    #     return edg_dictionary, dest_dictionary, data, mod
    # else: 
    '''
    metapath = []
    multiple_metapaths = []
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    # rank = comm.Get_rank()

    # Father process
    if comm.rank == 0:
        # Dataset for score function
        data = Data()
        data.x = x
        data.edge_index = edge_index
        data.edge_type = edge_type
        data.labels = true_labels
        data.labels = data.labels.unsqueeze(-1)
        data.num_nodes = x.size(0)
        data.bags = torch.empty(1)
        data.bag_labels = torch.empty(1)

        # All possible relations
        relations = torch.unique(data.edge_type).tolist()
        print(" \t\t ITERARION 1 ")
        for i, element in enumerate(relations):
            # copy of dataset
            data_copy = data.clone()
            # Index of child process 
            rank = i % (size - 1) + 1
            comm.send((data_copy, element, source_nodes_mask), dest=rank)
        
        # Results from child processes
        results = []
        for i in range(len(relations)):
            result = comm.recv()
            results.append(result)
        losses = []
        for r in results:
                losses.append(r[1])
                print(r[0], r[1])
        # calculate  a loss-threshold (we want to keep only relations with the smallest losses
        # but may be more than one -> multiple meta-paths)
        threshold = np.mean(losses)
        best = [item for item in results if item[1] < threshold]
        for i, element in enumerate(best):
            # copy of dataset
            data_mpgnn_copy = data_mpgnn.clone()
            # Index of child process 
            rank = i % (size - 1) + 1
            comm.send((data_mpgnn_copy, element), dest=rank)

        # Results from child processes
        mpgnn_accuracies = []
        for i in range(len(relations)):
            result = comm.recv()
            mpgnn_accuracies.append(result)
        for a in mpgnn_accuracies:
            print('a:', a)
         #####################################################
        # Take information from the relation with the best score
        best = min(results, key=lambda t: t[1])
        best_relation = best[0]
        best_loss = best[1]
        best_edge_dictionary = best[2]
        best_dest_dictionary = best[3]
        
        metapath.insert(0, best_relation)
        # Try MPGNN with actual metapath
        mpgnn_f1_macro, mpgnn_f1_micro = mpgnn(data_mpgnn, input_dim, hidden_dim, num_rel, output_dim, ll_output_dim, metapath)
        print('actual metapath: ', metapath, ' accuracy: ', mpgnn_f1_macro)
        if mpgnn_f1_macro >= 0.98:
            return metapath
        print('MPGNN f1 macro = ', mpgnn_f1_macro, 'MPGNN f1 micro = ', mpgnn_f1_micro)
        
        saved_best_mpgnn_acc = 0.
        for idx in range(0, metapath_length-1):
            # metapath.insert(0, best_relation)
            # print('actual metapath: ', metapath)
            # # Try MPGNN with actual metapath
            # mpgnn_accuracy = mpgnn(data_mpgnn, input_dim, hidden_dim, num_rel, output_dim, ll_output_dim, metapath)
            # if mpgnn_accuracy >= 0.98:
            #     return metapath
            # print('MPGNN accuracy = ', mpgnn_accuracy)
            tmp_metapath = []
            results_dict = {}
            print(" \t\t ITERARION ", idx+2)
            # Create node bags
            create_bags(best_edge_dictionary, best_dest_dictionary, data)
            for i, element in enumerate(relations):
                # copy of dataset
                data_copy = data.clone()
                # Index of child process 
                rank = i % (size - 1) + 1
                comm.send((data_copy, element), dest=rank)
            # Results from child processes
            results = []
            for i in range(len(relations)):
                result = comm.recv()
                results.append(result)
            for r in results:
                print(r[0], r[1])
            # check all relations. If the loss is small for more than one than keep more than one
            for r in results:
                tmp = []
                if r[1] < 0.01:
                    tmp.append(r[3]) # predictions for each restart
                    tmp.append(r[2]) # trained model
                    tmp.append(r[0]) # relation
                    tmp.append(r[1]) # loss
                    results_dict[r[0]] = tmp
            if bool(results_dict) == False:
                for r in results:
                    tmp = []
                    tmp.append(r[3]) # predictions for each restart
                    tmp.append(r[2]) # trained model
                    tmp.append(r[0]) # relation
                    tmp.append(r[1]) # loss
                    results_dict[r[0]] = tmp
            # Test all best relations with MPGNN
            print('siamo qui', len(results_dict))
            for key, items in results_dict.items():
                tmp_metapath = metapath.copy()
                tmp_metapath.insert(0, key)
                mpgnn_f1_macro, mpgnn_f1_micro = mpgnn(data_mpgnn, input_dim, hidden_dim, num_rel, output_dim, ll_output_dim, tmp_metapath)
                print('tmp: ', tmp_metapath, mpgnn_f1_macro, mpgnn_f1_micro)
                results_dict[key].append(mpgnn_f1_macro)

            # Take information from the relation with the best score 
            # if more than 1 relation gives a small loss (< 0.01), then keep candidates
            minimo = min(results_dict, key=lambda x: results_dict[x][3])
            best = {minimo: results_dict[minimo]}
            key = list(best.keys())[0]
            best_relation = key 
            best_loss = best[key][3]
            best_predictions_for_each_restart = best[key][0]
            best_model = best[key][1]
            best_mpgnn_acc = best[key][4]
            metapath.insert(0, best_relation)
            print('metapath: ', metapath, ' acc: ', best_mpgnn_acc)
            
            if best_mpgnn_acc > saved_best_mpgnn_acc:
                saved_best_mpgnn_acc = best_mpgnn_acc
            else:
                print('FINAL METAPATH: ', metapath)
                # Terminazione dei processi figli
                comm.Disconnect()
                return metapath

            # retrain bags
            predictions_for_each_restart = retrain_bags(data, best_relation, best_predictions_for_each_restart, BAGS=True)
            # relabel nodes into the bags
            source_nodes_mask = relabel_nodes_into_bags(predictions_for_each_restart, data, best_model)
            # draw and save confusion matrix and scatter
            #save_confusion_matrix(source_nodes_mask, data, embedding, idx+1)
            # create dictionaries
            edg_dictionary, dest_dictionary  = create_edge_dictionary(data, best_relation, source_nodes_mask, BAGS=False)
            # clean dictionaries
            best_edge_dictionary, best_dest_dictionary = clean_dictionaries(data, edg_dictionary, dest_dictionary, best_model)
        print(metapath)
        #return metapath
    # Children processes
    else:
        # Receive dataset and relation from father process
        data, element, source_nodes_mask = comm.recv()

        # Execute the function
        result = score_relation_parallel(data, element, source_nodes_mask)

        # Send result to father process
        comm.send(result, dest=0)

        # Receive dataset and relation from father process
        data_mp, element = comm.recv()

        # run mpgnn
        mp = []
        best_relation = element[0]
        mp.insert(0, best_relation)
        mpgnn_f1_micro = mpgnn_parallel(data_mpgnn, input_dim, hidden_dim, num_rel, output_dim, ll_output_dim, mp)

        # Send result to father process
        comm.send(mpgnn_f1_micro, dest=0)

        while(True):
            #Receive new information
            data, element = comm.recv()
            # Execute the function
            result = score_relation_bags_parallel(data, element)
            # Send result to father process
            comm.send(result, dest=0)
#'''
######################################################################################
######################################################################################
######################################################################################
######################################################################################
######################################################################################

    # MPI variables
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # metapaths variables
    current_metapaths_list, current_metapaths_dict = [], {}
    final_metapaths_list, final_metapaths_dict = [], {}

    if rank == 0:
        # Dataset for score function
        data = Data()
        data.x = x
        data.edge_index = edge_index
        data.edge_type = edge_type
        data.labels = true_labels
        data.labels = data.labels.unsqueeze(-1)
        data.num_nodes = x.size(0)
        data.bags = torch.empty(1)
        data.bag_labels = torch.empty(1)
        data.source_nodes_mask = source_nodes_mask

        # All possible relations
        relations = torch.unique(data.edge_type).tolist()
    else:
        data = None
        relations = None

    # Il processo padre invia i dati ai processi figli
    data = comm.bcast(data, root=0)
    relations = comm.bcast(relations, root=0)

    # Ogni processo figlio riceve solo una parte della lista graph
    local_relations = np.array_split(relations, size)[rank]

    # Execute the function
    result = []
    for rel in local_relations:
        partial_result = score_relation_parallel(data, rel, data.source_nodes_mask)
        result.append(partial_result)

    # Ogni processo figlio invia il risultato al processo padre
    result = comm.gather(result, root=0)

    if rank == 0:
        # Il processo padre raccoglie i risultati dai processi figli e li combina in una singola lista
        final_result = []
        for list in result:
            for tuple in list:
                final_result.append(tuple)
        # print(final_result)
        # final_result[0][0] -> relation
        # final_result[0][1] -> loss 
        # final_result[0][2] -> edge_dictionary
        # final_result[0][3] -> dest_dictionary

        # calculate  a loss-threshold (we want to keep only relations with the smallest losses
        # but may be more than one -> multiple meta-paths
        mean = np.mean([t[1] for t in final_result]) 
        # take only the relations under the mean threshold (best is a list of tuples)
        best = [item for item in final_result if item[1] < mean]
        # save relations in metapaths list
        for tuple in best:
            current_metapaths_list.append([tuple[0]])
            current_metapaths_dict[str([tuple[0]])] = []
            current_metapaths_dict[str([tuple[0]])].append(tuple[2])
            current_metapaths_dict[str([tuple[0]])].append(tuple[3])
        for i in range(0, len(current_metapaths_list)):
            mpgnn_f1_micro = mpgnn_parallel_multiple(data_mpgnn, input_dim, hidden_dim, num_rel, output_dim, ll_output_dim, [current_metapaths_list[i]])
            current_metapaths_dict[str(current_metapaths_list[i])].insert(0, mpgnn_f1_micro)
        # print(current_metapaths_list)
        # print(current_metapaths_dict)    

    # send current metapaths list and dict to children
    current_metapaths_list = comm.bcast(current_metapaths_list, root=0)
    current_metapaths_dict = comm.bcast(current_metapaths_dict, root=0)

    while current_metapaths_list:
        print('after while')
        for i in range(0, len(current_metapaths_list)):
            if rank == 0:
                create_bags(current_metapaths_dict[str(current_metapaths_list[i])][1], current_metapaths_dict[str(current_metapaths_list[i])][2], data)
            # Il processo padre invia i dati ai processi figli
            data = comm.bcast(data, root=0)
            relations = comm.bcast(relations, root=0)

            # Ogni processo figlio riceve solo una parte della lista graph
            local_relations = np.array_split(relations, size)[rank]

            # Execute the function
            result = []
            for rel in local_relations:
                partial_result = score_relation_bags_parallel(data, rel)
                result.append(partial_result)

            # Ogni processo figlio invia il risultato al processo padre
            result = comm.gather(result, root=0)
            
            if rank == 0:
                bool = False
                # Il processo padre raccoglie i risultati dai processi figli e li combina in una singola lista
                final_result = []
                for list in result:
                    for tuple in list:
                        final_result.append(tuple)
                # relation, loss.item(), model, predictions_for_each_restart
                for j in range(0, len(final_result)):
                    if final_result[j][1] <= 0.01:
                        tmp_meta = current_metapaths_list[i]
                        tmp_meta.insert(0, final_result[0])
                        mpgnn_f1_micro = mpgnn_parallel(data_mpgnn, input_dim, hidden_dim, num_rel, output_dim, ll_output_dim, tmp_meta)
                        if mpgnn_f1_micro > current_metapaths_dict[str(current_metapaths_list[i])][0]
                            current_metapaths_list.append(tmp_meta)


            # if rank == 0:
            #     # Il processo padre raccoglie i risultati dai processi figli e li combina in una singola lista
            #     final_result = []
            #     for list in result:
            #         for tuple in list:
            #             final_result.append(tuple)
            #     for res in final_result:
            #         print(res[0])
            #     current_metapaths_list = []





    



        






    
 
'''
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0: 
        # Dataset for score function
        data = Data()
        data.x = x
        data.edge_index = edge_index
        data.edge_type = edge_type
        data.labels = true_labels
        data.labels = data.labels.unsqueeze(-1)
        data.num_nodes = x.size(0)
        data.bags = torch.empty(1)
        data.bag_labels = torch.empty(1)

        # All possible relations
        relations = torch.unique(data.edge_type).tolist()
    else:
        data = None
        relations = None
        source_nodes_mask = None

    data = comm.bcast(data, root=0)
    relations = comm.bcast(relations, root=0)
    source_nodes_mask = comm.bcast(source_nodes_mask, root=0)

    result = None
    if rank != 0:
        result = score_relation_parallel(data, rank, source_nodes_mask)
    
    result = comm.gather(result, root=0)

    if rank == 0:
        print(result)
'''




'''
    print(" \t\t ITERARION 1 ")
    # score functions to obtain the relation with the lowest loss
    relation, edg_dictionary, trained_model, dest_dictionary, max_destination_node_for_source, loss_per_node, predictions = score_relations(data, source_nodes_mask, BAGS=False)
    # Append best relation in metapath list
    metapath.insert(0, relation)
    # Try mpgnn on the metapath
    mpgnn_accuracy = mpgnn(data_mpgnn, input_dim, hidden_dim, num_rel, output_dim, ll_output_dim, metapath)
    if mpgnn_accuracy >= 0.98:
        return metapath
    # clean source node mask
    source_nodes_mask = []
    # retrain on the previously obtained relation to get optimal trained weights
    for idx in tqdm(range(0, metapath_length-1)):
        print(" \t\t ITERARION ", idx+2)
        # create bags and put them in data object
        if idx == 0:
            FRST = True
            VAL = True
        else:
            FRST = False
            VAL = False
        create_bags(edg_dictionary, dest_dictionary, data, FRST)
        relation, trained_model, max_destination_nodes_source, max_destination_for_bag, loss_per_bag, pred, best_pred_for_each_restart, e_dictionary, d_dictionary = score_relation_bags_with_restarts(data, BAGS=True, VAL=VAL)

        metapath.insert(0, relation)
        # Try mpgnn on the metapath
        mpgnn_accuracy = mpgnn(data_mpgnn, input_dim, hidden_dim, num_rel, output_dim, ll_output_dim, metapath)
        if mpgnn_accuracy >= 0.98:
            return metapath
        mod = trained_model
        # retrain bags
        max_destination_nodes_source, trained_model, max_destination_for_bag, loss_per_bag, pred, destination_nodes_with_freezed_weights, edge_dictionary, max_destination_for_each_source, predictions_for_each_restart = retrain_bags(data, relation, best_pred_for_each_restart, BAGS=True)
        # relabel nodes into the bags
        source_nodes_mask = relabel_nodes_into_bags(predictions_for_each_restart, data, mod)
        # draw and save confusion matrix and scatter
        save_confusion_matrix(source_nodes_mask, data, embedding, idx+1)
        edg_dictionary, dest_dictionary  = create_edge_dictionary(data, relation, source_nodes_mask, BAGS=False)
        #dest_dictionary = create_destination_labels_dictionary(data, relation, source_nodes_mask)
        new_edg_dictionary, new_dest_dictionary = clean_dictionaries(data, edg_dictionary, dest_dictionary, mod)
        edg_dictionary, dest_dictionary = new_edg_dictionary.copy(), new_dest_dictionary.copy()
        # Save the variables to a file
        with open(pickle_filename, "wb") as f:
            pickle.dump((edg_dictionary, dest_dictionary, data, mod), f)
        #return edg_dictionary, dest_dictionary, mod, data, source_nodes_mask
    return max_destination_nodes_source, trained_model, max_destination_for_bag, loss_per_bag, pred, data, destination_nodes_with_freezed_weights, edge_dictionary, max_destination_for_each_source, predictions_for_each_restart

    #return max_destination_nodes, trained_model, first_embedding
'''

if __name__ == '__main__':

    
    EPOCHS = 200
    COMPLEX = True
    RESTARTS = 5
    NEGATIVE_SAMPLING = False

    metapath_length= 3
    tot_rel=5
    aggregation= 'max'
    epochs_relations = 150
    epochs_train = 150
    if COMPLEX == True:
        input_dim = 6
        ll_output_dim = 2
        dataset = "complex"
        folder= "/Users/francescoferrini/VScode/MultirelationalGNN/data/" + dataset + "/length_m_" + str(metapath_length) + "__tot_rel_" + str(tot_rel) + "/"
    elif COMPLEX == False:
        input_dim = 6
        ll_output_dim = 2
        dataset = "simple"
        folder= "/Users/francescoferrini/VScode/MultirelationalGNN/data/" + dataset + "/length_m_" + str(metapath_length) + "__tot_rel_" + str(tot_rel) + "/"
    elif COMPLEX == 'IMDB':
        tot_rel=4
        input_dim = 3066
        ll_output_dim = 3
        dataset = 'IMDB' ## 5
        folder= "/Users/francescoferrini/VScode/MultirelationalGNN/data/" + dataset + "/"
    elif COMPLEX == 'DBLP':
        input_dim = 4231
        tot_rel=6
        ll_output_dim = 4
        dataset = 'DBLP' ## 7
        folder= "/Users/francescoferrini/VScode/MultirelationalGNN/data/" + dataset + "/"
    elif COMPLEX == 'synthetic_multi':
        input_dim=6
        tot_rel=5
        ll_output_dim=2
        dataset = 'tot_rel_5'
        folder="/Users/francescoferrini/VScode/MultirelationalGNN/data/synthetic_multi/" + dataset + "/"
    
    node_file= folder + "node.dat"
    link_file= folder + "link.dat"
    label_file= folder + "label.dat"
    embedding_file = folder + "embedding.dat"
    # Define the filename for saving the variables
    pickle_filename = folder + "iteration_variables.pkl"
    # mpgnn variables
    hidden_dim = 32
    num_rel = tot_rel
    output_dim = 64

   
    meta = main(node_file, link_file, label_file, embedding_file, metapath_length, pickle_filename, input_dim, hidden_dim, num_rel, output_dim, ll_output_dim, dataset)
    



# src, prd = [], []
# for k, v in predictions_for_each_restart.items():
#     var = False
#     src.append(k)
#     for val in v:
#         if val >=0.98:
#             var = True
#     if var == True:
#         prd.append(1)
#     else:
#         prd.append(0)
# emb = torch.tensor(first_emb)
# a = torch.tensor(np.random.rand(len(src))/10)
# b = torch.tensor(np.random.rand(len(src))/10)

# prd = torch.tensor(prd)

# plt.scatter(prd+a, emb[src]+b, alpha=0.1)
# plt.show()

# correct_positive = 0
# uncorrect_positive = 0
# correct_negative = 0
# uncorrect_negative = 0

# for i in range(0, len(data.bags)):
#     if len(data.bags[i]) == 1 and data.bag_labels[i] == 0 and emb[data.bags[i][0]] == 0:
#         correct_negative += 1
#     if len(data.bags[i]) == 1 and data.bag_labels[i] == 0 and emb[data.bags[i][0]] == 1:
#         uncorrect_negative += 1
#     if data.bag_labels[i] == 1:
#         t = False
#         for node in data.bags[i]:
#             if emb[node] == 1:
#                 t = True
#         if t == True: correct_positive += 1
#         else: uncorrect_positive += 1

# correct_positive 
# uncorrect_positive 
# correct_negative 
# uncorrect_negative