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
from sklearn.utils import class_weight

from mpi4py import MPI
from sklearn.cluster import DBSCAN
from imblearn.under_sampling import RandomUnderSampler

seed= 10
torch.manual_seed(seed)
C = 0

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

def node_types_and_connected_relations(data, BAGS):
    rels = []
    if BAGS:
        s = list(set(sum(data.bags, [])))
        for i in range(0, len(data.edge_type)):
            if data.edge_index[0][i].item() in s:
                if data.edge_type[i].item() not in rels: rels.append(data.edge_type[i].item())
    else:
        for i in range(0, len(data.edge_type)):
            #if data.edge_index[0][i].item() in data.source_nodes_mask and data.labels[data.edge_index[0][i].item()].item() == 1:
            if data.labels[data.edge_index[0][i].item()].item() == 1:
                if data.edge_type[i].item() not in rels: rels.append(data.edge_type[i].item())
    #if not data.source_nodes_mask:
    #    rels = torch.unique(data.edge_type).tolist()
    return rels
    
def load_files_fb15k237(node_file_path, link_file_path, label_file_path, dataset):
    colors = pd.read_csv(node_file_path, sep='\t', header = None)
    colors = colors.dropna(axis=1,how='all')
    labels = pd.read_csv(label_file_path, sep='\t', header = None)
    links = pd.read_csv(link_file_path, sep='\t', header = None)
    labels.rename(columns = {0: 'node', 1: 'label'}, inplace = True)
    source_nodes_with_labels = labels['node'].values.tolist()
    labels = torch.tensor(labels['label'].values)
    colors.rename(columns = {0: 'node', 1: 'color'}, inplace = True)
    links.rename(columns = {0: 'node_1', 1: 'relation_type', 2: 'node_2'}, inplace = True)
    num_relations = len(list(set(links['relation_type'].to_list())))
    new_l = []
    for i in range(0, len(labels)):
        if labels[i].item() == 1:
            new_l.append(1)
        else:
            new_l.append(0)
    new_l = torch.tensor(new_l)
    return labels, colors, links, source_nodes_with_labels, num_relations, new_l

def load_files_dblp(node_file_path, link_file_path):

    colors = pd.read_csv(node_file_path, sep='\t', header = None)
    colors = colors.dropna(axis=1,how='all')

    train_labels = pd.read_csv('/Users/francescoferrini/VScode/MultirelationalGNN/data2/DBLP/labels_train.dat', sep='\t', header = None)
    train_labels.rename(columns = {0: 'node', 1: 'label'}, inplace = True)
    val_labels = pd.read_csv('/Users/francescoferrini/VScode/MultirelationalGNN/data2/DBLP/labels_val.dat', sep='\t', header = None)
    val_labels.rename(columns = {0: 'node', 1: 'label'}, inplace = True)
    test_labels = pd.read_csv('/Users/francescoferrini/VScode/MultirelationalGNN/data2/DBLP/labels_test.dat', sep='\t', header = None)
    test_labels.rename(columns = {0: 'node', 1: 'label'}, inplace = True)
    labels = pd.concat([train_labels, val_labels, test_labels], ignore_index=True)
    links = pd.read_csv(link_file_path, sep='\t', header = None)
    labels.rename(columns = {0: 'node', 1: 'label'}, inplace = True)
    source_nodes_with_labels = labels['node'].values.tolist()
    labels = torch.tensor(labels['label'].values)
    colors.rename(columns = {0: 'node', 1: 'color'}, inplace = True)
    links.rename(columns = {0: 'node_1', 1: 'relation_type', 2: 'node_2'}, inplace = True)
    num_relations = len(list(set(links['relation_type'].to_list())))
    new_l = []
    for i in range(0, len(labels)):
        if labels[i].item() == 1:
            new_l.append(1)
        else:
            new_l.append(0)
    new_l = torch.tensor(new_l)

    train_idx = train_labels['node'].values.tolist()
    train_y = torch.tensor(train_labels['label'].values.tolist())
    val_idx = val_labels['node'].values.tolist()
    val_y = torch.tensor(val_labels['label'].values.tolist())
    test_idx = test_labels['node'].values.tolist()
    test_y = torch.tensor(test_labels['label'].values.tolist())
    return labels, colors, links, source_nodes_with_labels, num_relations, new_l, train_idx, train_y, val_idx, val_y, test_idx, test_y

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
        new_l = []
        for i in range(0, len(labels)):
            if labels[i].item() == 1:
                new_l.append(1)
            else:
                new_l.append(0)
        new_l = torch.tensor(new_l)
        return labels, colors, links, source_nodes_with_labels, new_l
    
def gtn_files(node_idx, train_idx, train_y, test_idx, test_y, val_idx, val_y, data, dataset_path):
    from scipy.sparse import csr_matrix
    if not os.path.exists(dataset_path):
        # Crea la cartella
        os.makedirs(dataset_path)
    edges_list = []
    rels = torch.unique(data.edge_type).tolist()
    for r in rels:
        e_idx = masked_edge_index(data.edge_index, data.edge_type == r)
        edges = []
        for i in range(0, len(e_idx[0])):
            edges.append((e_idx[0][i].item(), e_idx[1][i].item()))
        # ottieni la lista degli id dei nodi
        node_ids = sorted(list(set([v for e in edges for v in e])))

        # crea un dizionario che associa ogni id di nodo a un indice intero
        node_to_index = {node_id: i for i, node_id in enumerate(node_ids)}

        # crea le liste degli indici delle righe, delle colonne e dei valori
        row_indices = [node_to_index[e[0]] for e in edges]
        col_indices = [node_to_index[e[1]] for e in edges]
        values = np.ones(len(edges))

        # crea la matrice di adiacenza
        car_matrix = csr_matrix((values, (row_indices, col_indices)), shape=(len(node_ids), len(node_ids)))
        edges_list.append(car_matrix)
    with open(dataset_path + '/edges.pkl', 'wb') as handle:
        pickle.dump(edges_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
        

    # gtn file for labels
    labels_list = []
    train_labels_list = []
    val_labels_list = []
    test_labels_list = []
    for i in range(0, len(train_idx)):
        tmp = [train_idx[i], train_y[i].item()]
        train_labels_list.append(tmp)
    for i in range(0, len(val_idx)):
        tmp = [val_idx[i], val_y[i].item()]
        val_labels_list.append(tmp)
    for i in range(0, len(test_idx)):
        tmp = [test_idx[i], test_y[i].item()]
        test_labels_list.append(tmp)
    labels_list.append(train_labels_list)
    labels_list.append(val_labels_list)
    labels_list.append(test_labels_list)
    with open(dataset_path + '/labels.pkl', 'wb') as handle:
        pickle.dump(labels_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # node features
    features = data.x.type(torch.int64)
    with open(dataset_path + '/node_features.pkl', 'wb') as handle:
        pickle.dump(features.numpy(), handle, protocol=pickle.HIGHEST_PROTOCOL)



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

def find_unique_indices(nums):
    count = {}
    unique_indices = []
    
    # Conta l'occorrenza di ogni intero nella lista
    for i, num in enumerate(nums):
        if num in count:
            count[num][0] += 1
        else:
            count[num] = [1, i]
    
    # Aggiungi gli indici degli interi che compaiono una sola volta alla lista di indici unici
    for num, (occurrence, index) in count.items():
        if occurrence == 1:
            unique_indices.append(index)
    
    return unique_indices

def rimuovi_e_aggiungi_elementi(lista_elementi, lista_indici, tensor):
    elementi_rimossi = [lista_elementi.pop(indice) for indice in sorted(lista_indici, reverse=True)]
    if tensor == True: lista_elementi=torch.tensor(lista_elementi)
    return lista_elementi, elementi_rimossi

def splitting_node_and_labels(lab, feat, src, dataset):
    if dataset == 'complex' or dataset == 'simple' or dataset == 'synthetic_multiple':
        node_idx = list(feat['node'].values)
    else:
        node_idx = src.copy()
    # alcune classi potrebbero avere un solo elemento dunque tolgo i nodi associati e li inserisco successivaemnte
    # indici delle classi con un solo elemento
    unique_indices = find_unique_indices(lab.tolist())
    # rimuovo gli indici appena trovati da node_idx e lab
    if unique_indices:

        node_idx, nodes_removed = rimuovi_e_aggiungi_elementi(node_idx, unique_indices, tensor=False)
        lab, lab_removed = rimuovi_e_aggiungi_elementi(lab.tolist(), unique_indices, tensor=False)

    # splitto senza considerare gli elementi di una sola classe
    train_idx,test_idx,train_y,test_y = train_test_split(node_idx, lab,
                                                            random_state=415,
                                                            stratify=lab, 
                                                            test_size=0.1)
    
    train_idx,val_idx,train_y,val_y = train_test_split(train_idx, train_y,
                                                            random_state=415,
                                                            stratify=train_y, 
                                                            test_size=0.2)

    if unique_indices:
        train_idx.extend(nodes_removed)
        train_y.extend(lab_removed)
        return torch.tensor(node_idx), train_idx, torch.tensor(train_y), test_idx, torch.tensor(test_y), val_idx, torch.tensor(val_y)
    v = False
    if v == True:
        # Creazione di un array numpy per i nodi nel training set
        train_nodes = np.array(train_idx)

        # Creazione di un array numpy per le etichette
        labels = np.array(train_y)

        # Ottenere gli indici unici presenti nella lista train_nodes
        unique_nodes = np.unique(train_nodes)

        # Creazione di un array booleano indicando i nodi nel training set
        train_mask = np.isin(unique_nodes, train_nodes)

        # Estrazione delle etichette corrispondenti ai nodi nel training set
        train_labels = labels[train_mask]

        # Calcolo del numero di campioni in ciascuna classe
        class_counts = np.bincount(train_labels)

        # Calcolo del numero minimo di campioni tra tutte le classi
        num_campioni_minimo = np.min(class_counts)

        # Impostazione della strategia di campionamento per ottenere lo stesso numero di campioni per tutte le classi
        undersampling_strategy = {i: num_campioni_minimo for i in range(len(class_counts))}

        # Esecuzione dell'undersampling con la strategia personalizzata
        undersampler = RandomUnderSampler(sampling_strategy=undersampling_strategy, random_state=42)
        train_nodes_resampled, train_y = undersampler.fit_resample(unique_nodes.reshape(-1, 1), train_labels)

        # Stampa del numero di campioni per ciascuna classe dopo l'undersampling
        class_counts_resampled = np.bincount(train_y)
        for i, count in enumerate(class_counts_resampled):
            print(f"Numero di campioni nella classe {i} dopo l'undersampling:", count)

        # Creazione della lista finale degli indici
        final_train_idx = [i[0] for i in train_nodes_resampled]
        print(final_train_idx, torch.tensor(train_y))
        return torch.tensor(node_idx), torch.tensor(final_train_idx), torch.tensor(train_y), test_idx, test_y, val_idx, val_y
    return torch.tensor(node_idx), train_idx, train_y, test_idx, test_y, val_idx, val_y

def get_node_features(colors):
    node_features = pd.get_dummies(colors)
    
    node_features.drop(["node"], axis=1, inplace=True)
    
    x = node_features.to_numpy().astype(np.float32)
    x = np.flip(x, 1).copy()
    x = torch.from_numpy(x) 
    return x

def mask_features_test_nodes(test_index, val_index, feature_matrix):
    for indice in test_index:
        feature_matrix[indice] = torch.zeros_like(feature_matrix[indice])
    for indice in val_index:
        feature_matrix[indice] = torch.zeros_like(feature_matrix[indice])
    return feature_matrix


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
                #destination_dictionary[dst.item()].append(data.labels[source_nodes_mask.index(src.item())].item())
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
    start = - 0.2
    end = 0.2
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
    start = - 0.3
    end = 0.3
    for key, values in destination_dictionary.items():
        if key in destination_nodes_with_freezed_weights:
            weights[key] = previous_weights[key]
        else:
            weights[key] = random.uniform(0., 1.)
            #weights[key] = abs(min(values) + random.uniform(start, end))
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
    #print('creo bags', edg_dictionary[5], dest_dictionary[314], dest_dictionary[3768])
    #print(dest_dictionary)
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
        
    data.bags = new_bag.copy()
    data.bag_labels = torch.Tensor(new_labels).unsqueeze(-1)
    #print(new_labels.count(1))
    #for i, elm in enumerate(data.bags):
    #    if data.bag_labels[i].item() == 1:
    #        print(elm, data.bag_labels[i])
    #print('len bags: ', len(data.bags))

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

def relabel_nodes_inside_bags(predictions_for_each_restart, data, mod, embeddings):
    for name, param in mod.named_parameters():
            if name == 'output.LinearLayerAttri.weight':  
                param_list = param
    #print(param_list.tolist()[0])
    # if a node into the bags is predicted as 1 at least one time, then his label willl be 1 otherwise 0
    data.labels = torch.zeros(data.num_nodes, 1)
    for k, v in predictions_for_each_restart.items():
        '''print('---')
        print(v)
        print(param_list.tolist()[0])
        print(data.x[k])
        print(torch.dot(torch.tensor(param_list.tolist()[0]), data.x[k]).item())'''
        #v = [x * torch.dot(torch.tensor(param_list.tolist()[0]), data.x[k]).item() for x in v]
        data.labels[k] = 0
        if max(v) > 0.9:
            data.labels[k] = 1
    #for name, param in mod.named_parameters():
    #    print(name, param)
    # delete from source mask, nodes that have a color value similar to zero (a zero in the linear layer)
    #print('1: ', mod.output.LinearLayerAttri.weight)
    
    #print(mod.output.LinearLayerAttri.weight.detach.tolist())
    src = list(predictions_for_each_restart.keys())
    # Calcola la matrice di confusione
    '''cm = confusion_matrix(embeddings, data.labels.tolist())

    # Estrai i valori dalla matrice di confusione
    tn, fp, fn, tp = cm.ravel()

    # Stampa i valori
    print("True Positive:", tp)
    print("False Positive:", fp)
    print("True Negative:", tn)
    print("False Negative:", fn)'''
    return src
    
def freeze_weights(model, destination_nodes_with_freezed_weights, previous_weights):
    with torch.no_grad():
        for idx in range(0, len(model.input.weights())):
            if idx in destination_nodes_with_freezed_weights: model.input.weights()[idx] = previous_weights[idx]

def train(data, edge_dictionary, model, optimizer, criterion, source_nodes_mask, criterion_per_node, destination_nodes_with_freezed_weights, previous_weights, grad_mask, BAGS, bags_to_predict, bags_to_predict_labels):
    model.train()
    optimizer.zero_grad()
    if BAGS:
        current_data = data.clone()
        current_data.bags = bags_to_predict
        predictions, max_destination_node_for_bag, max_destination_node_for_source = model(current_data, edge_dictionary, BAGS)
        labels = bags_to_predict_labels
        #labels = data.bag_labels
    else:
        predictions, max_destination_node_for_bag, max_destination_node_for_source = model(data, edge_dictionary, BAGS)
        labels = data.labels
        predictions, labels = predictions[source_nodes_mask].to(torch.float32), labels[source_nodes_mask].to(torch.float32)
        #predictions, labels = predictions[source_nodes_mask].to(torch.float32), labels.to(torch.float32)
      
    loss = criterion(predictions, labels)
    loss_per_node = criterion_per_node(predictions, labels)
    loss.backward()
    # freeze weights (multiply the grad tensor with a mask of 0s and 1s).
    if destination_nodes_with_freezed_weights:
        model.input.weights.grad = model.input.weights.grad*grad_mask
    optimizer.step()

    with torch.no_grad():
        model.input.weights[:] = torch.clamp(model.input.weights, min = 0.0, max = 1.0)
        model.output.LinearLayerAttri.weight[:] = torch.clamp(model.output.LinearLayerAttri.weight, min = 0.0, max = 1.0)
        if destination_nodes_with_freezed_weights:
            for idx in range(0, len(model.input.weights[:])):
                if idx in destination_nodes_with_freezed_weights: model.input.weights[:][idx] = previous_weights[idx] 
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

def score_relation_parallel(data, relation, source_nodes):
    if not source_nodes:
        first = True
    else:
        first = False
    # All source nodes with relation type 'relation' are considered (first iteration)
    if first:
        source_nodes = masked_edge_index(data.edge_index, data.edge_type == relation)
        source_nodes = torch.unique(source_nodes[0]).tolist()
    # Create dictionary of source and destinaiton nodes connected with a specific relation type
    edge_dictionary, destination_dictionary = create_edge_dictionary(data, relation, source_nodes, BAGS=False)
    #print('relation', relation, destination_dictionary)
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
    #for epoch in tqdm(range(0, EPOCHS)):
    for epoch in range(0, EPOCHS):
        loss, max_destination_node_for_source, loss_per_node, max_destination_node_for_bag, predictions = train(data, edge_dictionary, model, optimizer, criterion, source_nodes, criterion_per_node, [], weights, torch.tensor(0), BAGS=False, bags_to_predict=None, bags_to_predict_labels=None)
        #print(relation, loss)
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
    bags, bag_labels = clean_bags_for_relation_type(data, edge_dictionary)
    # Initialize weights
    weights = initialize_weights(data, destination_dictionary, BAGS=True)

    # Retrieve loss
    criterion = get_loss()
    criterion_per_node = get_loss_per_node()

    destination_nodes_with_freezed_weights = []
    RESTARTS=1
    EPOCHS=50
    for i in tqdm(range(0, RESTARTS)):
        # Retrieve model
        model = get_model(weights)
        # Retrieve optimizer
        optimizer = get_optimizer(model)
        # Training
        for epoch in range(0, EPOCHS):
            loss, max_destination_node_for_source, loss_per_bag, max_destination_node_for_bag, predictions = train(data, edge_dictionary, model, optimizer, criterion, source_nodes_mask, criterion_per_node, destination_nodes_with_freezed_weights, weights, torch.tensor(0), BAGS, bags_to_predict=bags, bags_to_predict_labels=bag_labels)
        # save all predictions
        for key, value in max_destination_node_for_source.items():
            best_pred_for_each_restart[key].append(value.item())
        #print('RESTART ', i, ' LOSS: ', loss.item()) 
        destination_nodes_with_freezed_weights = []
        weights = reinitialize_weights(data, destination_dictionary, model.input.weights.detach(), destination_nodes_with_freezed_weights, BAGS=True)
    return best_pred_for_each_restart
    #return max_destination_node_for_source, model, max_destination_node_for_bag, loss_per_bag, predictions, destination_nodes_with_freezed_weights, edge_dictionary, best_pred_for_each_restart

def score_relation_bags_parallel(data, relation):
    rest, current_loss = 0, 100
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
    
    while (rest<2):
        # Retrieve model
        model = get_model(weights)
        # Retrieve optimizer
        optimizer = get_optimizer(model)
        # Training
        EPOCHS=50
        #for epoch in tqdm(range(0, EPOCHS)):
        for epoch in range(0, EPOCHS):
            loss, max_destination_node_for_source, loss_per_bag, max_destination_node_for_bag, predictions = train(data, edge_dictionary, model, optimizer, criterion, source_nodes_mask, criterion_per_node, destination_nodes_with_freezed_weights, weights, grad_mask, BAGS=True, bags_to_predict=bags, bags_to_predict_labels=bag_labels)
        # save predictions
        for key, value in max_destination_node_for_source.items():
            if key not in predictions_for_each_restart:
                predictions_for_each_restart[key] = []
            predictions_for_each_restart[key].append(value.item())
        if loss.item() < current_loss:
            #print(rest, 'relation: ', relation, 'new loss: ', loss.item())
            destination_nodes_with_freezed_weights = retrieve_destinations_low_loss(max_destination_node_for_bag, loss_per_bag, source_nodes_mask)
            current_loss = loss.item()
            rest=0
            #print('\n Relation ', relation, ' loss: ', current_loss.item())
        else:
            #print(rest, 'relation: ', relation, 'new loss: ', loss.item())
            #print('\n Relation ', relation, ' end')
            rest+=1
        for node in destination_nodes_with_freezed_weights:
            grad_mask[node] = 0
        weights = reinitialize_weights(data, destination_dictionary, model.input.weights.detach(), destination_nodes_with_freezed_weights, BAGS=False)
    #print('relation: ', relation , 'final loss: ', current_loss, len(destination_nodes_with_freezed_weights))
    #for name, param in model.named_parameters():
     #       if name == 'output.LinearLayerAttri.weight':
      #          print(relation, name, param)
    #print('\nRelation ', relation, ' loss: ', current_loss.item())
    return relation,  current_loss, model, predictions_for_each_restart

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
    print('bags: ', type(data), data)
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
    out = model(data.x, data.edge_index, data.edge_type)
    #for name, param in model.named_parameters():
     #   print(name, param)
    # Calcolo dei pesi delle classi
    weights = class_weight.compute_class_weight('balanced', classes=np.unique(data.train_y), y=data.train_y.tolist())
    # Conversione dei pesi in un tensore di PyTorch
    weights_tensor = torch.tensor(weights, dtype=torch.float)   
    loss = F.nll_loss(out[data.train_idx].squeeze(-1), data.train_y)#, weight = weights_tensor)
    '''
    # regularization
    l2_reg = None
    for param in model.parameters():
        if l2_reg is None:
            l2_reg = torch.norm(param, p=2)**2  # Calcola la norma L2 al quadrato del parametro
        else:
            l2_reg += torch.norm(param, p=2)**2
    l2_lambda = 0.01
    loss += l2_lambda * l2_reg
    '''
    #loss = F.cross_entropy(out[data.train_idx], data.train_y)
    loss.backward()
    #for name, param in model.named_parameters():
     #   print(name, param)
    optimizer.step()
    return float(loss), weights

@torch.no_grad()
def mpgnn_validation(model, data, class_weight):
    model.eval()
    pred = model(data.x, data.edge_index, data.edge_type)#.argmax(dim=-1)
    loss_val = F.nll_loss(pred[data.val_idx].squeeze(-1), data.val_y)
    
    train_predictions = torch.argmax(pred[data.train_idx], 1).tolist()
    val_predictions = torch.argmax(pred[data.val_idx], 1).tolist()
    
    train_y = data.train_y.tolist()
    val_y = data.val_y.tolist()
    f1_train = f1_score(train_predictions, train_y, average='macro')
    #print('train accuracy: ', f1_train)
    f1_val_macro = f1_score(val_predictions, val_y, average = 'macro')
    #print('val accuracy: ', f1_val_macro)
    f1_val_micro = f1_score(val_predictions, val_y, average = 'macro')
    return f1_train, f1_val_micro, f1_val_macro,loss_val

@torch.no_grad()
def mpgnn_test(model, data, class_weight):
    model.eval()
    pred = model(data.x, data.edge_index, data.edge_type)#.argmax(dim=-1)
    loss_test = F.nll_loss(pred[data.test_idx].squeeze(-1), data.test_y)
    
    test_predictions = torch.argmax(pred[data.test_idx], 1).tolist()
    test_y = data.test_y.tolist()
    f1_test_micro = f1_score(test_predictions, test_y, average = 'macro')
    #print(test_predictions)
    #print(test_y)
    return loss_test, f1_test_micro

# def mpgnn_parallel(data_mpgnn, input_dim, hidden_dim, num_rel, output_dim, ll_output_dim, metapath):
#     metapath=[0, 1, 2]
#     mpgnn_model = MPNet(input_dim, hidden_dim, num_rel, output_dim, ll_output_dim, len(metapath), metapath)
#     print(mpgnn_model)
#     # for name, param in mpgnn_model.named_parameters():
#     #     print(name, param, param.size())
#     mpgnn_optimizer = torch.optim.Adam(mpgnn_model.parameters(), lr=0.01, weight_decay=0.0005)
#     best_macro, best_micro = 0., 0.
#     for epoch in tqdm(range(1, 100)):
#         loss = mpgnn_train(mpgnn_model, mpgnn_optimizer, data_mpgnn)
#         train_acc, f1_test_micro, f1_test_macro = mpgnn_validation(mpgnn_model, data_mpgnn)
        
#         if f1_test_macro > best_macro:
#             best_macro = f1_test_micro
#         if f1_test_micro > best_micro:
#             best_micro = f1_test_micro
#     return best_micro

def mpgnn_parallel_multiple(data_mpgnn, input_dim, hidden_dim, num_rel, output_dim, ll_output_dim, metapaths):
    #metapaths = [[2, 0], [3, 1]] # imdb
    #metapaths = [[1, 4, 2, 0], [1, 0], [1, 5, 3, 0]]
    #metapaths = [[4, 3, 0], [1, 0], [0, 4, 2]]
    #metapaths = [[2, 1, 0]]
    print('METAPATHS: ', metapaths)
    mpgnn_model = MPNetm(input_dim, hidden_dim, num_rel, output_dim, ll_output_dim, len(metapaths), metapaths)
    # for name, param in mpgnn_model.named_parameters():
    #     print(name, param, param.size())
    mpgnn_optimizer = torch.optim.Adam(mpgnn_model.parameters(), lr=0.01, weight_decay=0.0005)
    best_macro, best_micro = 0., 0.
    for epoch in range(1, 2000):
        loss, class_weight = mpgnn_train(mpgnn_model, mpgnn_optimizer, data_mpgnn)
        train_acc, f1_val_micro, f1_valt_macro, loss_val = mpgnn_validation(mpgnn_model, data_mpgnn, class_weight)
        if f1_val_micro > best_micro:
            best_micro = f1_val_micro
            best_model = mpgnn_model
        if epoch % 100 == 0:
            print(epoch, "train loss %0.3f" % loss, "validation loss %0.3f" % loss_val,
                  'train micro: %0.3f'% train_acc, 'validation micro: %0.3f'% f1_val_micro)
            
    test_loss, f1_micro_test = mpgnn_test(best_model, data_mpgnn, class_weight)
    print("test loss %0.3f" % test_loss, "test micro %0.3f" % f1_micro_test)
    return f1_micro_test

def find_smallest_values(accuracies_list):
    # Creazione del modello DBSCAN
    dbscan = DBSCAN(eps=0.1, min_samples=1)
    
    # Addestramento del modello
    dbscan.fit(np.array(accuracies_list).reshape(-1, 1))
    
    # Trovare i cluster unici
    unique_labels = np.unique(dbscan.labels_)
    
    # Trovare il cluster con un solo elemento
    smallest_cluster = None
    smallest_cluster_size = float('inf')
    
    for label in unique_labels:
        cluster_indices = np.where(dbscan.labels_ == label)[0]
        cluster_size = len(cluster_indices)
        
        if cluster_size == 1 and cluster_size < smallest_cluster_size:
            smallest_cluster = cluster_indices
            smallest_cluster_size = cluster_size
    
    # Verificare se è stato trovato un cluster con un solo elemento
    if smallest_cluster is not None:
        smallest_values = [accuracies_list[i] for i in smallest_cluster]
        return smallest_values
    
    return min(accuracies_list) 

def main(node_file_path, link_file_path, label_file_path, embedding_file_path, metapath_length, pickle_filename, input_dim, hidden_dim, num_rel, output_dim, ll_output_dim, dataset, dataset_path):
    # MPI variables
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    if rank == 0:
        print(dataset)
        # Obtain true 0|1 labels for each node, feature matrix (1-hot encoding) and links among nodes
        if dataset == 'complex' or dataset == 'simple':
            sources = []
            true_labels, features, edges, embedding = load_files(node_file_path, link_file_path, label_file_path, embedding_file_path, dataset)
        elif dataset == 'fb15k237':
            true_labels, features, edges, sources, num_rel, labels_multi = load_files_fb15k237(node_file_path, link_file_path, label_file_path, dataset)
        elif dataset == 'dblp' or dataset == 'imdb':
            true_labels, features, edges, sources, num_rel, labels_multi, train_idx, train_y, val_idx, val_y, test_idx, test_y = load_files_dblp(node_file_path, link_file_path)
        else: 
            true_labels, features, edges, sources, labels_multi = load_files(node_file_path, link_file_path, label_file_path, embedding_file_path, dataset)
        # Get features' matrix
        x = get_node_features(features)
        
        if dataset == 'fb15k237' or dataset == 'dblp' or dataset == 'imdb' :
            input_dim = len(x[0])
            ll_output_dim = len(torch.unique(true_labels).tolist())


        # Get edge_index and types
        edge_index, edge_type = get_edge_index_and_type_no_reverse(edges)

        if dataset != 'dblp':
            # Split data into train and test
            node_idx, train_idx, train_y, test_idx, test_y, val_idx, val_y = splitting_node_and_labels(true_labels, features, sources, dataset)
        #node_idx, train_idx, train_y, test_idx, test_y, val_idx, val_y = splitting_node_and_labels(labels_multi, features, sources, dataset)
        #x = mask_features_test_nodes(test_idx, val_idx, x)
  
        # Dataset for MPGNN
        data_mpgnn = Data()
        data_mpgnn.x = x
        data_mpgnn.edge_index = edge_index
        data_mpgnn.edge_type = edge_type
        data_mpgnn.train_idx = train_idx
        data_mpgnn.test_idx = test_idx
        data_mpgnn.train_y = train_y
        data_mpgnn.test_y = test_y
        data_mpgnn.val_idx = val_idx
        data_mpgnn.val_y = val_y
        data_mpgnn.num_nodes = data_mpgnn.x.size(0)
        # Variables
        if sources:
            source_nodes_mask = sources
        else:
            source_nodes_mask = []
        metapath = []

        # generate files for gtn
        gtn_files(node_idx, train_idx, train_y, test_idx, test_y, val_idx, val_y, data_mpgnn, dataset_path)
        print('gtn finished')
    

    if rank == 0:
        # metapaths variables
        current_metapaths_list, current_metapaths_dict = [], {}
        intermediate_metapaths_list = []
        final_metapaths_list, final_metapaths_dict = [], {}

        # Dataset for score function
        data = Data()
        data.x = x
        data.edge_index = edge_index
        data.edge_type = edge_type
        if dataset == 'simple' or dataset == 'complex':
            data.labels = true_labels
        else:
            data.labels = labels_multi
        data.labels = data.labels.unsqueeze(-1)
        data.num_nodes = x.size(0)
        data.bags = torch.empty(1)
        data.bag_labels = torch.empty(1)
        data.source_nodes_mask = source_nodes_mask
        # All possible relations
        relations = torch.unique(data.edge_type).tolist()
        actual_relations = node_types_and_connected_relations(data, BAGS=False)
        print(actual_relations)
        result = []
        intermediate_metapaths_list = []
    else:
        data = None
        relations = None
        actual_relations = None
        current_metapaths_list = None 
        current_metapaths_dict = None
        intermediate_metapaths_list = None

    # Il processo padre invia i dati ai processi figli
    data = comm.bcast(data, root=0)
    relations = comm.bcast(relations, root=0)
    actual_relations = comm.bcast(actual_relations, root=0)

    # Ogni processo figlio riceve solo una parte della lista graph
    local_relations = np.array_split(actual_relations, size)[rank]
    #local_relations = np.array_split(relations, size)[rank]
    
    # Execute the function
    p_result = []
    for rel in local_relations:
        partial_result = score_relation_parallel(data, rel, data.source_nodes_mask)
        p_result.append(partial_result)

    # Ogni processo figlio invia il risultato al processo padre
    result = comm.gather(p_result, root=0)
    if rank == 0:
       print('local: ', local_relations)
       final_result = sum(result, [])

    if rank == 0:
        # Il processo padre raccoglie i risultati dai processi figli e li combina in una singola lista
        #final_result = []
        #for list in final_result:
            #if len(list[2]) != 0:
          #      final_result.append(list)
        #print()
        #print('Relations and scores: ')
        min_a, min_n = 100, 0
        print('first iteration')
        for r in final_result:
            print(r[0], r[1])
            if r[1] < min_a:
                min_a = r[1]
                min_n = r[0]
        #print()
        rels, accs = [], []
        for item in final_result:
            value_0 = item[0]  # Valore in posizione 0
            value_1 = item[1]  # Valore in posizione 1
            rels.append(value_0)
            accs.append(value_1)
        #smallest_values = find_smallest_values(accs)
        #print('smallest : ', smallest_values, min_n, min_a)

        # creo l'array delle differenze per scegliere quali relazioni considerare
        accs.sort()
        array_differenze = np.diff(accs)
        if len(array_differenze) >= 2:
            indice = np.argmax(array_differenze)
            # calculate  a loss-threshold (we want to keep only relations with the smallest losses
            # but may be more than one -> multiple meta-paths
            mean = np.mean([t[1] for t in final_result]) 
            # take only the relations under the mean threshold (best is a list of tuples
            best = [item for item in final_result if item[1] <= accs[indice]]
        else:
            best = [item for item in final_result]

        # final_result[0][0] -> relation
        # final_result[0][1] -> loss 
        # final_result[0][2] -> edge_dictionary
        # final_result[0][3] -> dest_dictionary

        
        #best = [item for item in final_result if item[1] <= min_a]
        # save relations in metapaths list
        for tuple in best:
            current_metapaths_list.append([tuple[0]])
            current_metapaths_dict[str([tuple[0]])] = []
            current_metapaths_dict[str([tuple[0]])].append(tuple[0])
            current_metapaths_dict[str([tuple[0]])].append(tuple[2])
            current_metapaths_dict[str([tuple[0]])].append(tuple[3])
        for i in range(0, len(current_metapaths_list)):
            mpgnn_f1_micro = mpgnn_parallel_multiple(data_mpgnn, input_dim, hidden_dim, num_rel, output_dim, ll_output_dim, [current_metapaths_list[i]])
            current_metapaths_dict[str(current_metapaths_list[i])].insert(1, mpgnn_f1_micro)
        # print(current_metapaths_list)
        # print(current_metapaths_dict)    

    # send current metapaths list and dict to children
    current_metapaths_list = comm.bcast(current_metapaths_list, root=0)
    current_metapaths_dict = comm.bcast(current_metapaths_dict, root=0)
    #'''
    while current_metapaths_list:
        for i in range(0, len(current_metapaths_list)):
            #print('-----------------------NEXT-----------------------', i , len(current_metapaths_list), current_metapaths_list, type(data), data)
            if rank == 0:
                print('current_metapaths: ', current_metapaths_list, ' final_metapaths: ', final_metapaths_list)#, 'meta dict: ', current_metapaths_dict)
                print('metapath used now: ', current_metapaths_list[i])
                create_bags(current_metapaths_dict[str(current_metapaths_list[i])][2], current_metapaths_dict[str(current_metapaths_list[i])][3], data)
                actual_relations = node_types_and_connected_relations(data, BAGS=True)
                print('actual : ', len(actual_relations))
                #print('bags: ', data.bags)
            
            # check whether there are relations
            
            if actual_relations:

                # Il processo padre invia i dati ai processi figli
                data = comm.bcast(data, root=0)
                actual_relations = comm.bcast(actual_relations, root=0)

                # Ogni processo figlio riceve solo una parte della lista graph
                local_relations = np.array_split(actual_relations, size)[rank]
                # Execute the function
                #for rel in local_relations:
                #    partial_result = score_relation_bags_parallel(data, rel)

                p_result = []
                for rel in local_relations:
                    partial_result = score_relation_bags_parallel(data, rel)
                    p_result.append(partial_result)

                # Ogni processo figlio invia il risultato al processo padre
                result = comm.gather(p_result, root=0)

                if rank == 0:
                    at_least_one = False
                    arr = []
                    bool = False
                    # Il processo padre raccoglie i risultati dai processi figli e li combina in una singola lista
                    final_result = sum(result, [])
                    for r in final_result:
                        print(r[0], r[1])
                        arr.append(r[1])
                    # creo l'array delle differenze per scegliere quali relazioni considerare
                    arr.sort()
                    array_differenze = np.diff(arr)
                    indice = np.argmax(array_differenze)
                    print('threshold: ', arr[indice])
                    # relation, loss.item(), model, predictions_for_each_restart
                    for j in range(0, len(final_result)):
                        #print('relation', final_result[j][0], 'score: ', final_result[j][1])
                        if final_result[j][1] <= arr[indice]: #np.mean([t[1] for t in final_result]): #0.01:
                        #if final_result[j][1] <= 0.02:
                            tmp_meta = current_metapaths_list[i].copy()
                            tmp_meta.insert(0, final_result[j][0])
                            mpgnn_f1_micro = mpgnn_parallel_multiple(data_mpgnn, input_dim, hidden_dim, num_rel, output_dim, ll_output_dim, [tmp_meta])
                            print('actual meta: ', tmp_meta, 'micro f1: ', mpgnn_f1_micro, 'previous micro: ', current_metapaths_dict[str(current_metapaths_list[i])][1], 'previous meta: ', current_metapaths_list[i])
                            if mpgnn_f1_micro > 0.99: 
                                final_metapaths_list = [tmp_meta.copy()]
                                current_metapaths_list = []
                                intermediate_metapaths_list = []
                            else:
                                if mpgnn_f1_micro > current_metapaths_dict[str(current_metapaths_list[i])][1] and tmp_meta not in intermediate_metapaths_list:
                                    print('better')
                                    intermediate_metapaths_list.append(tmp_meta)
                                    # retrain bags
                                    predictions_for_each_restart = retrain_bags(data, final_result[j][0], final_result[j][3], BAGS=True)
                                    #for i, e in predictions_for_each_restart.items():
                                    #   print(i, e)
                                    # relabel nodes into the bags
                                    source_nodes_mask = relabel_nodes_inside_bags(predictions_for_each_restart, data, final_result[j][2], embedding)
                                    edg_dictionary, dest_dictionary  = create_edge_dictionary(data, final_result[j][0], source_nodes_mask, BAGS=False)
                                    new_edge_dictionary, new_dest_dictionary = clean_dictionaries(data, edg_dictionary, dest_dictionary, final_result[j][2])
                                    current_metapaths_dict[str(tmp_meta)] = [final_result[j][0], mpgnn_f1_micro, new_edge_dictionary, new_dest_dictionary]
                                    #current_metapaths_dict[str(tmp_meta)] = [final_result[j][0], final_result[j][1], new_edge_dictionary, new_dest_dictionary]
                                    at_least_one = True
                                elif mpgnn_f1_micro < current_metapaths_dict[str(current_metapaths_list[i])][1] and at_least_one==False and j==len(final_result)-1:
                                    print('worst')
                                    if current_metapaths_list[i] not in final_metapaths_list: final_metapaths_list.append(current_metapaths_list[i])
                    print('final_metapath_list: ', final_metapaths_list, 'current meta list: ', current_metapaths_list, 'intermediate: ', intermediate_metapaths_list)
        if rank == 0:
            if not intermediate_metapaths_list:
                for elm in current_metapaths_list:
                    final_metapaths_list.append(elm)
                current_metapaths_list = []
            else:
                current_metapaths_list = intermediate_metapaths_list.copy()
                intermediate_metapaths_list = []
        current_metapaths_list = comm.bcast(current_metapaths_list, root=0)
        # If intermediate metapath list is empty it means that the system didn't find any new metapath
        # intermediate_metapaths_list = comm.bcast(intermediate_metapaths_list, root=0)
        # if rank == 0:
        #     print('ci siamo ', intermediate_metapaths_list)
        #     if not intermediate_metapaths_list:
        #        return current_metapaths_list
        #     current_metapaths_list = intermediate_metapaths_list.copy()
        #     intermediate_metapaths_list = []

        # if not intermediate_metapaths_list:
        #     return

    # final training
    if rank == 0:
        for elm in current_metapaths_list: 
            if elm not in final_metapaths_list: final_metapaths_list.append(elm)
        print('siamo alla fine', current_metapaths_list, final_metapaths_list)
        mpgnn_f1_micro = mpgnn_parallel_multiple(data_mpgnn, input_dim, hidden_dim, num_rel, output_dim, ll_output_dim, final_metapaths_list)
        print("Final accuracy: ", mpgnn_f1_micro)
        return mpgnn_f1_micro
    return
    #'''

if __name__ == '__main__':

    
    EPOCHS = 200
    COMPLEX = True
    RESTARTS = 5
    NEGATIVE_SAMPLING = False

    metapath_length= 4
    metapaths_number = 1
    tot_rel=3
    deterministic = False


    aggregation= 'max'
    epochs_relations = 150
    epochs_train = 150
    if COMPLEX == True:
        input_dim = 3
        ll_output_dim = 2
        dataset = "simple"
        if deterministic:
            folder= "/Users/francescoferrini/VScode/MultirelationalGNN/data/" + dataset + "/deterministic/"+ "tot_rel_" + str(tot_rel) + '_metapaths_number_' + str(metapaths_number) + "/"
        #folder= "/Users/francescoferrini/VScode/MultirelationalGNN/data/" + dataset + "/length_m_" + str(metapath_length) + "__tot_rel_" + str(tot_rel) + "/"
        else:
            folder= "/Users/francescoferrini/VScode/MultirelationalGNN/data/synthetic/"+ "tot_rel_" + str(tot_rel) + '_metapaths_number_' + str(metapaths_number) + '_metapath_length_' + str(metapath_length) + "/"
    elif COMPLEX == False:
        input_dim = 3
        ll_output_dim = 2
        dataset = "simple"
        #folder= "/Users/francescoferrini/VScode/MultirelationalGNN/data/" + dataset + "/length_m_" + str(metapath_length) + "__tot_rel_" + str(tot_rel) + "/"
        folder= "/Users/francescoferrini/VScode/MultirelationalGNN/data/" + dataset + "/"+ "tot_rel_" + str(tot_rel) + "/"
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
    elif COMPLEX == 'fb15k237':
        input_dim=55
        tot_rel= 227
        ll_output_dim= 2
        dataset = 'fb15k237'
        folder="/Users/francescoferrini/VScode/MultirelationalGNN/data/" + dataset + "/"
    elif COMPLEX == 'dblp':
        input_dim=55
        tot_rel= 227
        ll_output_dim= 2
        dataset = 'dblp'
        folder="/Users/francescoferrini/VScode/MultirelationalGNN/data2/DBLP/"
    
    node_file= folder + "node.dat"
    link_file= folder + "link.dat"
    label_file= folder + "label.dat"
    embedding_file = folder + "embedding.dat"
    # Define the filename for saving the variables
    pickle_filename = folder + "iteration_variables.pkl"
    # mpgnn variables
    hidden_dim = 16
    num_rel = tot_rel
    output_dim = 64

    dataset_path = '/Users/francescoferrini/VScode/Graph_Transformer_Networks/data/synthetic/'+ "tot_rel_" + str(tot_rel) + '_metapaths_number_' + str(metapaths_number) + '_metapath_length_' + str(metapath_length) + "/"
    
    meta = main(node_file, link_file, label_file, embedding_file, metapath_length, pickle_filename, input_dim, hidden_dim, num_rel, output_dim, ll_output_dim, dataset, dataset_path)



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
