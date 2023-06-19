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
            #if data.labels[data.edge_index[0][i].item()].item() == 1:
            if data.edge_index[0][i].item() in data.source_nodes_mask:
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
    loss = F.nll_loss(out[data.train_idx].squeeze(-1), data.train_y, weight = weights_tensor)
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
   # print('conv1: ', model.conv1.weight)
    
    
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
    print(test_predictions)
    print(test_y)
    print('param: ')
    for n, p in model.named_parameters():
        print(n, p)
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

def mpgnn_parallel_multiple(data_mpgnn, input_dim, hidden_dim, num_rel, output_dim, ll_output_dim, metapath_length):
    
    mpgnn_model = Net(input_dim, hidden_dim, num_rel, output_dim, ll_output_dim, metapath_length)
    # for name, param in mpgnn_model.named_parameters():
    #     print(name, param, param.size())
    mpgnn_optimizer = torch.optim.Adam(mpgnn_model.parameters(), lr=0.01, weight_decay=0.0005)
    best_macro, best_micro = 0., 0.
    for epoch in range(1, 1000):
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
        
  
        # Dataset for MPGNN
        data_rgcn = Data()
        data_rgcn.x = x
        data_rgcn.edge_index = edge_index
        data_rgcn.edge_type = edge_type
        data_rgcn.train_idx = train_idx
        data_rgcn.test_idx = test_idx
        data_rgcn.train_y = train_y
        data_rgcn.test_y = test_y
        data_rgcn.val_idx = val_idx
        data_rgcn.val_y = val_y
        data_rgcn.num_nodes = data_rgcn.x.size(0)
        # Variables
        if sources:
            source_nodes_mask = sources
        else:
            source_nodes_mask = []
        metapath = []

        mpgnn_f1_micro = mpgnn_parallel_multiple(data_rgcn, input_dim, hidden_dim, num_rel, output_dim, ll_output_dim, metapath_length)


if __name__ == '__main__':

    
    EPOCHS = 200
    COMPLEX = True
    RESTARTS = 5
    NEGATIVE_SAMPLING = False

    metapath_length= 4
    metapaths_number = 1
    tot_rel=3
    deterministic = True
    hidden_dim = 32
    #num_rel = 4
    #num_rel = 8
    #num_rel = 10
    num_rel = 15
    output_dim = 64

    d = 'overlap_3_rels_3/'


    aggregation= 'max'
    epochs_relations = 150
    epochs_train = 150
    if COMPLEX == True:
        input_dim = 2
        ll_output_dim = 2
        dataset = "simple"
        if deterministic:
            folder = '/Users/francescoferrini/VScode/MultirelationalGNN/data/final_datasets/metapath_length_4/' + d
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
    hidden_dim = 32
    num_rel = tot_rel
    output_dim = 32

    dataset_path = '/Users/francescoferrini/VScode/Graph_Transformer_Networks/data/synthetic/'+ "tot_rel_" + str(tot_rel) + '_metapaths_number_' + str(metapaths_number) + '_metapath_length_' + str(metapath_length) + "/"
    
    meta = main(node_file, link_file, label_file, embedding_file, metapath_length, pickle_filename, input_dim, hidden_dim, num_rel, output_dim, ll_output_dim, dataset, dataset_path)


