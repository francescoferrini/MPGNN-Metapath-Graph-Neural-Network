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

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from functools import partial
import multiprocess as mp


def masked_edge_index(edge_index, edge_mask):
    print('c')
    if isinstance(edge_index, Tensor):
        return edge_index[:, edge_mask]
    else:
        return print('Error')

def load_files(node_file_path, links_file_path, label_file_path, embedding_file_path):
    colors = pd.read_csv(node_file_path, sep='\t', header = None)
    labels = pd.read_csv(label_file_path, sep='\t', header = None)
    links = pd.read_csv(links_file_path, sep='\t', header = None)
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
    # emb = embedding['first embedding'].tolist()
    # last_emb = embedding['second embedding'].tolist()
    labels.rename(columns = {0: 'node', 1: 'label'}, inplace = True)
    labels = torch.tensor(labels['label'].values)
    colors.rename(columns = {0: 'node', 1: 'color'}, inplace = True)
    links.rename(columns = {0: 'node_1', 1: 'relation_type', 2: 'node_2'}, inplace = True)

    return labels, colors, links, embedding

def splitting_node_and_labels(lab, feat):
    node_idx = torch.tensor(feat['node'].values)
    train_split = int(len(node_idx)*0.8)
    test_split = len(node_idx) - train_split
    train_idx = node_idx[:train_split]
    test_idx = node_idx[-test_split:]

    train_y = lab[:train_split]
    test_y = lab[-test_split:]
    return node_idx, train_idx, train_y, test_idx, test_y

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
    print(relation)
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
    destination_dictionary = {}
    edge_index = masked_edge_index(data.edge_index, data.edge_type == relation)
        
    for src, dst in zip(edge_index[0], edge_index[1]):
        if src.item() in source_nodes_mask and dst.item() not in destination_dictionary: destination_dictionary[dst.item()] = []

    for src, dst in zip(edge_index[0], edge_index[1]):
        if src.item() in source_nodes_mask:
            destination_dictionary[dst.item()].append(data.labels[src.item()].item())

    '''
        destination_bag_dictionary is a dictionary where keys are string of bags and values are destination nodes attached to them
    '''
    if BAGS:        
        destination_bag_dictionary = {}
        for src, dst in zip(edge_index[0], edge_index[1]):
            for i in range(0, len(data.bags)):
                if src.item() in data.bags[i]:
                    if dst.item() not in destination_bag_dictionary: destination_bag_dictionary[dst.item()] = []
                    destination_bag_dictionary[dst.item()].append(data.bag_labels[i].item())

        return edge_dictionary_copy, destination_dictionary, destination_bag_dictionary
    return edge_dictionary_copy, destination_dictionary

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
        if torch.dot(data.x[key], mod.output.LinearLayerAttri.weight[0]).item() < 0.1:
            dsts = edge_dictionary_copy[key]
            for destination in dsts:
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

def create_bags(edg_dictionary, dest_dictionary, data, FRST):
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
    print('this is slow')
    data.bags = to_keep
    data.bag_labels = torch.Tensor(to_keep_labels).unsqueeze(-1)

def relabel_nodes_into_bags(predictions_for_each_restart, data, mod):
    # if a node into the bags is predicted as 1 at least one time, then his label willl be 1 otherwise 0
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

def train(data, edge_dictionary, model, optimizer, criterion, source_nodes_mask, criterion_per_node, destination_nodes_with_freezed_weights, previous_weights, BAGS):
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
    optimizer.step()

    with torch.no_grad():
        for i in range(0, len(model.input.weights)):
            model.input.weights[i].clamp_min_(0)
        #model.input.weights[:] = torch.clamp(model.input.weights, min = 0.0)
        model.output.LinearLayerAttri.weight[:] = torch.clamp(model.output.LinearLayerAttri.weight, min = 0.0)

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
    print(relation)
    edge_dictionary, destination_dictionary = create_edge_dictionary(data, relation, source_nodes_mask, BAGS=False)
    return edge_dictionary
                    
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
            source_nodes_mask = torch.unique(source_nodes_mask[0]).tolist()# = list(np.array(torch.unique(source_nodes_mask[0])))
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
        EPOCHS = 1
        for epoch in tqdm(range(0, EPOCHS)):
            loss, max_destination_node_for_source, loss_per_node, max_destination_node_for_bag, predictions = train(data, edge_dictionary, model, optimizer, criterion, source_nodes_mask, criterion_per_node, [], weights, BAGS) 
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
    edge_dictionary, destination_dictionary, destination_bag_dictionary = create_edge_dictionary(data, relation, source_nodes_mask, BAGS=True)
    # Create dictionary of destination nodes and their specific source labels
    #estination_dictionary = create_destination_labels_dictionary(data, relation, source_nodes_mask)

    # Initialize weights
    weights = initialize_weights(data, destination_bag_dictionary, BAGS=True)

    # Retrieve loss
    criterion = get_loss()
    criterion_per_node = get_loss_per_node()

    # In each restart, the weights of the good destination nodes are freezed for the next restarts
    destination_nodes_with_freezed_weights = []
    RESTARTS=10
    EPOCHS=200
    print('Retrain bags for relation ', relation)
    for i in range(0, RESTARTS):
        # Early stopping variable
        early_stopping_counter = 0
        # Retrieve model
        model = get_model(weights)
        # Retrieve optimizer
        optimizer = get_optimizer(model)
        # Training
        for epoch in tqdm(range(0, EPOCHS)):
            loss, max_destination_node_for_source, loss_per_bag, max_destination_node_for_bag, predictions = train(data, edge_dictionary, model, optimizer, criterion, source_nodes_mask, criterion_per_node, destination_nodes_with_freezed_weights, weights, BAGS)
            # Early stopping
            #stop = early_stopping(epoch, loss, early_stopping_counter)
            if epoch == 0:
                prev_loss = loss
            elif loss >= prev_loss:# or abs(loss-prev_loss) < 0.001:
                early_stopping_counter += 1
            else:
                early_stopping_counter = 0
            if early_stopping_counter == 10:
                break
            else:
                prev_loss = loss
        # save all predictions
        for key, value in max_destination_node_for_source.items():
            best_pred_for_each_restart[key].append(value.item())
        #print('RESTART ', i, ' LOSS: ', loss.item()) 
        destination_nodes_with_freezed_weights = []
        weights = reinitialize_weights(data, destination_bag_dictionary, model.weights().detach(), destination_nodes_with_freezed_weights, BAGS=True)

    return max_destination_node_for_source, model, max_destination_node_for_bag, loss_per_bag, predictions, destination_nodes_with_freezed_weights, edge_dictionary, best_pred_for_each_restart

def parallel_score_bags(relation, data):
    # parallelize
    source_nodes_mask = []
    edge_dictionary, destination_dictionary, destination_bag_dictionary = create_edge_dictionary(data, relation, source_nodes_mask, BAGS=True)
    return edge_dictionary

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
            if destination_nodes_with_freezed_weights:
                model.frz_weights(destination_nodes_with_freezed_weights)
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
    out = model(data.x, data.edge_index, data.edge_type)
    #print(out[data.train_idx], data.train_y)
    loss = F.nll_loss(out[data.train_idx], data.train_y)
    loss.backward()
    optimizer.step()
    return float(loss)

@torch.no_grad()
def mpgnn_test(model, data):
    model.eval()
    pred = model(data.x, data.edge_index, data.edge_type).argmax(dim=-1)
    #print('pred: ', pred)
    train_acc = float((pred[data.train_idx] == data.train_y).float().mean())
    test_acc = float((pred[data.test_idx] == data.test_y).float().mean())
    return train_acc, test_acc

def mpgnn(data_mpgnn, input_dim, hidden_dim, num_rel, output_dim, ll_output_dim, metapath):
    metapath = [0, 1, 2]
    mpgnn_model = MPNet(input_dim, hidden_dim, num_rel, output_dim, ll_output_dim, len(metapath), metapath)
    for name, param in mpgnn_model.named_parameters():
        print(name, param, param.size())
    mpgnn_optimizer = torch.optim.Adam(mpgnn_model.parameters(), lr=0.01, weight_decay=0.0005)
    for epoch in range(1, 1000):
        loss = mpgnn_train(mpgnn_model, mpgnn_optimizer, data_mpgnn)
        train_acc, test_acc = mpgnn_test(mpgnn_model, data_mpgnn)
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {train_acc:.4f} '
            f'Test: {test_acc:.4f}')
    return test_acc


def main(node_file_path, link_file_path, label_file_path, embedding_file_path, metapath_length, pickle_filename, input_dim, hidden_dim, num_rel, output_dim, ll_output_dim):
    # Obtain true 0|1 labels for each node, feature matrix (1-hot encoding) and links among nodes
    true_labels, features, edges, embedding = load_files(node_file_path, link_file_path, label_file_path, embedding_file_path)
    # Get features' matrix
    x = get_node_features(features)

    # Get edge_index and types
    edge_index, edge_type = get_edge_index_and_type_no_reverse(edges)

    # Split data into train and test
    node_idx, train_idx, train_y, test_idx, test_y = splitting_node_and_labels(true_labels, features)
    
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
    source_nodes_mask = []
    metapath = []



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

    relations = [0, 1, 2, 3]
    print('daje')
    with mp.Pool(8) as pool:
        score_relation_partial = partial(score_relation_parallel, data, source_nodes_mask=source_nodes_mask)
        res = pool.map(score_relation_partial, relations)
    print('daje')
    print(res)
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

if __name__ == '__main__':

    
    EPOCHS = 200
    COMPLEX = True
    RESTARTS = 5
    NEGATIVE_SAMPLING = False

    metapath_length= 3
    tot_rel= 5
    aggregation= 'max'
    epochs_relations = 150
    epochs_train = 150
    if COMPLEX == True:
        dataset = "complex"
    else:
        dataset = "simple"
    folder= "/Users/francescoferrini/VScode/MultirelationalGNN/data/" + dataset + "/length_m_" + str(metapath_length) + "__tot_rel_" + str(tot_rel) + "/"
    node_file= folder + "node.dat"
    link_file= folder + "link.dat"
    label_file= folder + "label.dat"
    embedding_file = folder + "embedding.dat"
    # Define the filename for saving the variables
    pickle_filename = folder + "iteration_variables.pkl"
    # mpgnn variables
    input_dim = 6
    hidden_dim = 32
    num_rel = tot_rel
    output_dim = 32
    ll_output_dim = tot_rel

    #
    #dict1, dict2, model, data, src = main(node_file, link_file, label_file, embedding_file, metapath_length, pickle_filename)
    meta = main(node_file, link_file, label_file, embedding_file, metapath_length, pickle_filename, input_dim, hidden_dim, num_rel, output_dim, ll_output_dim)
    #dict, mod, emb_first, emb_second, max_dst_bag, loss, data = main(node_file, link_file, label_file, embedding_file, metapath_length)
    #max_destination_node_for_source, trained_model, first_emb, second_emb, max_destination_node_for_bag, loss, predictions, data, destination_nodes_with_freezed_weights, edge_dictionary, max_destination_for_each_source, predictions_for_each_restart = main(node_file, link_file, label_file, embedding_file, metapath_length)
    #m = main(node_file, link_file, label_file, embedding_file, metapath_length)



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


####################################################### MODEL
import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn
import random


FEATURES_DIM = 6

class InputLayer(torch.nn.Module):
    def __init__(self, weights):
        super(InputLayer, self).__init__()
        # Trainable weights
        self.weights = nn.Parameter(weights.unsqueeze(-1))
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
                        index_max_destination_node = random.choice((weights_of_source == torch.max(weights_of_source)).nonzero(as_tuple=True)[0].tolist())
                        # retrive the node with this weight and put in list
                        max_node = node_dict[source_node][index_max_destination_node]   
                        # retrieve the max weight for the source_node in the bag
                        #max_destination_node_for_source[source_node] = torch.mul(weights[max_node], torch.dot(torch.tensor([0., 0., 1., 0., 0., 0.]), feat[source_node]))
                        max_destination_node_for_source[source_node] = torch.mul(weights[max_node], self.LinearLayerAttri(feat[source_node]))
                        if max_node not in max_destination_nodes: max_destination_nodes.append(max_node)
                        # put in max_node_for_current_bag the max node so far for this bag
                        if max_destination_node_for_source[source_node] > max_weight_for_current_bag: 
                            max_weight_for_current_bag = max_destination_node_for_source[source_node]
                            max_destination_node_for_bag[str(bags[i])] = max_node
                            max_weights[i] = max_destination_node_for_source[source_node]
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
                #weights_of_source = [weights[i] for i in node_dict[source_node]]
                # Concatenate the subset of parameters into a new tensor
                # retrieve weights of nodes connected to source_node
                #weights_of_source = weights[node_dict[source_node]].squeeze(-1)
                # retrive the index of the max weight (may be more than one, in this case I use one of the 2)
                index_max_destination_node = random.choice((weights_of_source == torch.max(weights_of_source)).nonzero(as_tuple=True)[0].tolist())
                #index_max_destination_node = weights_of_source.index(max(weights_of_source))
                # retrive the node with this weight and put in a dictionary
                max_node = node_dict[source_node][index_max_destination_node]
                max_destination_node_for_source[source_node] = max_node
                # max_destination_node_for_source[source_node] = torch.mul(weights[max_node],self.LinearLayerAttri(feat[source_node]))
                # update max_weight tensor
                max_weights[source_node] = max(weights[node_dict[source_node]].squeeze(-1))#weights[max_node]
                #max_weights.requires_grad_(True)
                #max_weights[source_node] = torch.mul(weights[max_node],self.LinearLayerAttri(feat[source_node]))
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
                x = F.relu(self.conv1(self.metapath[layer_index], x, edge_index, edge_type))
            else:
                x = F.relu(self.conv2(self.metapath[layer_index], x, edge_index, edge_type))
        x = self.LinearLayer(x)
        return F.log_softmax(x, dim=1)

