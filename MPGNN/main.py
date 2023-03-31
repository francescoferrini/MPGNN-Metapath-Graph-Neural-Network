import pandas as pd
import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import HeteroData, Data
from mpgnn_model import *
import random
from sklearn.metrics import f1_score
# input_dim = 2  
# hidden_dim = 32
# num_rel = 10
# output_dim = hidden_dim
# ll_output_dim = num_rel # output dimension of the linear layer. One neuron for each label
# n_layers = 5

# node_file_path = '/Users/francescoferrini/VScode/HNE-pyg/data/dataset_manual/' + '5/node.dat' #+ str(n_layers)+ '/node.dat'
# links_file_path = '/Users/francescoferrini/VScode/HNE-pyg/data/dataset_manual/'+ '5/link.dat' #+ str(n_layers)+ '/link.dat'
# label_file_path = '/Users/francescoferrini/VScode/HNE-pyg/data/dataset_manual/'+ '5/label.dat' #+ str(n_layers)+ '/label.dat'


def load_files(node_file_path, links_file_path, label_file_path):
    colors = pd.read_csv(node_file_path, sep='\t', header = None)
    labels = pd.read_csv(label_file_path, sep='\t', header = None)
    #links = pd.read_csv(links_file_path, sep='\t', header = None, skiprows=1)
    links = pd.read_csv(links_file_path, sep='\t', header = None)

    labels.rename(columns = {0: 'node', 1: 'label'}, inplace = True)
    colors.rename(columns = {0: 'node', 1: 'color'}, inplace = True)
    links.rename(columns = {0: 'node_1', 1: 'relation_type', 2: 'node_2'}, inplace = True)

    return labels, colors, links

def splitting_node_and_labels(labels, colors):
    node_idx = torch.tensor(colors['node'].values)   
    train_split = int(len(node_idx)*0.8)
    test_split = len(node_idx) - train_split
    train_idx = node_idx[:train_split]
    test_idx = node_idx[-test_split:]
    labels_id = torch.tensor(labels['label'].values)
    train_y = labels_id[:train_split]
    test_y = labels_id[-test_split:]
    return node_idx, train_idx, train_y, test_idx, test_y

# def splitting_node_and_labels(labels, colors):
#     node_idx = torch.tensor(colors['node'].values)
#     labels_id = torch.tensor(labels['label'].values)
#     node_idx_list = node_idx.tolist()
#     labels_id_list = labels_id.tolist()
#     # Creazione di una lista di tuple (nodo, label)
#     nodes_labels = list(zip( node_idx_list, labels_id_list))
#     # Separazione dei nodi con label 0 e quelli con label 1
#     nodes_label0 = [node for node in nodes_labels if node[1] == 0]
#     nodes_label1 = [node for node in nodes_labels if node[1] == 1]
#     # Popolazione casuale delle liste di training e test
#     train_label0 = random.sample(nodes_label0, 250)
#     train_label1 = random.sample(nodes_label1, 200)
#     # train_idx = [node[0] for node in train_label0 + train_label1]
#     # train_y = [node[1] for node in train_label0 + train_label1]
#     train_idx, train_y = zip(*random.sample(train_label0 + train_label1, 450))
#     train_idx, train_y = torch.tensor(list(train_idx)), torch.tensor(list(train_y))

#     test_label0 = random.sample([node for node in nodes_label0 if node not in train_label0], 150)
#     test_label1 = random.sample([node for node in nodes_label1 if node not in train_label1], 100)
#     # test_idx = [node[0] for node in test_label0 + test_label1]
#     # test_y = [node[1] for node in test_label0 + test_label1]
#     test_idx, test_y = zip(*random.sample(test_label0 + test_label1, 250))
#     test_idx, test_y = torch.tensor(list(test_idx)), torch.tensor(list(test_y))
#     # shuffling

#     return node_idx, train_idx, train_y, test_idx, test_y

def get_node_features(colors):
    node_features = pd.get_dummies(colors)
    
    node_features.drop(["node"], axis=1, inplace=True)
    
    x = node_features.to_numpy().astype(np.float32)
    x = np.flip(x, 1).copy()
    x = torch.from_numpy(x) 
    return x

def get_edge_index_and_type_no_reverse(links):
    edge_index = links.drop(['relation_type'], axis=1)
    edge_index = torch.tensor([edge_index['node_1'].values, edge_index['node_2'].values])

    edge_type = links['relation_type']
    edge_type = torch.tensor(edge_type)

    return edge_index, edge_type

def get_edge_index_and_type_reverse(links):
    edge_index = links.drop(['relation_type'], axis=1)
    edge_index = torch.tensor([np.array(edge_index['node_1'].values), np.array(edge_index['node_2'].values)])
    edge_index = torch.tensor([np.array(torch.cat((edge_index[0], edge_index[1]), 0)), np.array(torch.cat((edge_index[1], edge_index[0]), 0))])
    
    edge_type = links['relation_type']
    edge_type = torch.tensor(np.concatenate((np.array(edge_type), np.array(edge_type))))
    return edge_index, edge_type

def get_edge_index_and_type_reverse_x_2(links, num_relations):
    edge_index = links.drop(['relation_type'], axis=1)
    edge_index = torch.tensor([np.array(edge_index['node_1'].values), np.array(edge_index['node_2'].values)])
    edge_index = torch.tensor([np.array(torch.cat((edge_index[0], edge_index[1]), 0)), np.array(torch.cat((edge_index[1], edge_index[0]), 0))])
    
    edge_type = links['relation_type']
    new_edge_type = [a+num_relations for a in edge_type]
    edge_type = torch.tensor(np.concatenate((np.array(edge_type), new_edge_type)))
    return edge_index, edge_type

def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, data.edge_type)
    #weight_loss = torch.tensor([1., 10.])
    loss = F.nll_loss(out[data.train_idx], data.train_y)#, weight_loss)
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test(model, data):
    model.eval()
    pred = model(data.x, data.edge_index, data.edge_type).argmax(dim=-1)
    print('train')
    print(pred[data.train_idx].tolist().count(1), pred[data.train_idx].tolist().count(0))
    print(data.train_y.tolist().count(1), data.train_y.tolist().count(0))
    print('test')
    print(pred[data.test_idx].tolist().count(1), pred[data.test_idx].tolist().count(0))
    print(data.test_y.tolist().count(1), data.test_y.tolist().count(0))
    print('f1 score train')
    f1_train = f1_score(data.train_y, pred[data.train_idx])
    print(f1_train)
    print('f1 score test')
    f1_test = f1_score(data.test_y, pred[data.test_idx])
    print(f1_test)
    train_acc = float((pred[data.train_idx] == data.train_y).float().mean())
    test_acc = float((pred[data.test_idx] == data.test_y).float().mean())
    return train_acc, test_acc

def main(args):
    # Load files
    labels, colors, links = load_files(args.node_file_path, args.link_file_path, args.label_file_path)

    # Split data into train and test
    node_idx, train_idx, train_y, test_idx, test_y = splitting_node_and_labels(labels, colors)

    # Get node features
    x = get_node_features(colors)

    #Get index and type of edges
    edge_index, edge_type = get_edge_index_and_type_no_reverse(links)

    # Dataset creation
    data = Data()
    data.x = x
    data.edge_index = edge_index
    data.edge_type = edge_type
    data.train_idx = train_idx
    data.test_idx = test_idx
    data.train_y = train_y
    data.test_y = test_y
    data.num_nodes = node_idx.size(0)

    # Metapath
    metapath = []
    for s in args.metapath:
        metapath.append(int(s))
    # Model
    if args.MP_GNN == 'True':
        print(metapath)
        model = MPNet(args.input_dim, args.hidden_dim, args.num_rel, args.output_dim, args.ll_output_dim, args.n_layers, metapath)
    else:
        model = Net(args.input_dim, args.hidden_dim, args.num_rel, args.output_dim, args.ll_output_dim, args.n_layers, args.metapath_length)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)

    for epoch in range(1, 3000):
        loss = train(model, optimizer, data)
        train_acc, test_acc = test(model, data)
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {train_acc:.4f} '
            f'Test: {test_acc:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RGCN')
    parser.add_argument("--node_file_path", type=str, required=True,
            help="dataset to use")
    parser.add_argument("--link_file_path", type=str, required=True,
            help="dataset to use")
    parser.add_argument("--label_file_path", type=str, required=True,
            help="dataset to use")
    parser.add_argument('--input_dim', required=True, type=int, default=2, 
            help='dimension of feature')
    parser.add_argument("--hidden_dim", type=int, default=32,
            help="hidden dimension")
    parser.add_argument("--output_dim", type=int, default=32,
            help="output dimension for last RGCN layer")
    parser.add_argument("--num_rel", type=int, default=5,
            help="total number of relations")
    parser.add_argument("--ll_output_dim", type=int, default=2,
            help="output dimension for linear layer")
    parser.add_argument("--n_layers", type=int, default=5,
            help="number of RGCN layers")
    parser.add_argument("--metapath_length", type=int, default=2,
            help="length of the metapath")
    parser.add_argument("--metapath", type=str, default='abc',
            help="metapath. Only for experimental purpose")
    parser.add_argument("--MP_GNN", type=str, default="False",
            help="if true use custom MPGNN network")

    args = parser.parse_args()
    print(args, flush=True)
    main(args)
