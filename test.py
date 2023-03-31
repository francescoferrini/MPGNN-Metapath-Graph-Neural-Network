
from errno import EDEADLK
import argparse
from resource import struct_rusage
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import random
import math
import numpy as np
import torch
from torch import Tensor

def masked_edge_index(edge_index, edge_mask):
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