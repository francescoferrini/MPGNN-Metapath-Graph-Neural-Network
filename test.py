from mpi4py import MPI
import numpy as np
from sklearn.cluster import DBSCAN
import torch
from sklearn.metrics import f1_score
import numpy as np
import torch
import pickle
from scipy.sparse import csr_matrix
import pandas as pd
import torch
from torch_sparse import coalesce
import networkx as nx
import scipy.sparse as sp


def main():
    links = '/Users/francescoferrini/Desktop/data2/IMDB/edges.pkl'
    labels = '/Users/francescoferrini/Desktop/data2/IMDB/labels.pkl'
    features = '/Users/francescoferrini/Desktop/data2/IMDB/node_features.pkl'

    with open(features, 'rb') as file:
        data = pickle.load(file)

    # node output file
    output_file = '/Users/francescoferrini/Desktop/data2/IMDB/node_bow.dat'

    # Apri il file in modalità scrittura
    with open(output_file, 'w') as file:
        # Scorrere le righe della matrice
        for idx, row in enumerate(data):
            # Scrivi l'indice di riga seguito dagli elementi separati da tabulazioni
            row_str = '\t'.join(map(str, row))
            # Write the index and the row content to the file
            file.write('{}\t{}\n'.format(idx, row_str))
    
    # labels
    output_file_train = '/Users/francescoferrini/Desktop/data2/IMDB/labels_train.dat'
    output_file_val = '/Users/francescoferrini/Desktop/data2/IMDB/labels_val.dat'
    output_file_test = '/Users/francescoferrini/Desktop/data2/IMDB/labels_test.dat'

    with open(labels, 'rb') as file:
        data = pickle.load(file)
    with open(output_file_train, 'w') as file:
        for item in data[0]:
            index, label = item
            file.write(f'{index}\t{label}\n')
    with open(output_file_val, 'w') as file:
        for item in data[1]:
            index, label = item
            file.write(f'{index}\t{label}\n')
    with open(output_file_test, 'w') as file:
        for item in data[2]:
            index, label = item
            file.write(f'{index}\t{label}\n')
    '''
    #links
    with open(links, 'rb') as file:
        sparse_matrix = np.load(file, allow_pickle=True)
    output_file = '/Users/francescoferrini/Desktop/data2/IMDB/matrix3.dat'


    with open(output_file, 'w') as file:
        for i in range(sparse_matrix[3].shape[0]):
            row_indices = sparse_matrix[3].indices[sparse_matrix[3].indptr[i]:sparse_matrix[3].indptr[i + 1]]
            if len(row_indices) > 0:
                for j in row_indices:
                    file.write(f"{i}\t3\t{j}\n")
    '''
    file_paths = [
    '/Users/francescoferrini/Desktop/data2/IMDB/matrix0.dat',
    '/Users/francescoferrini/Desktop/data2/IMDB/matrix1.dat',
    '/Users/francescoferrini/Desktop/data2/IMDB/matrix2.dat',
    '/Users/francescoferrini/Desktop/data2/IMDB/matrix3.dat'
]

    # Percorso del file in cui combinare i contenuti
    output_file = '/Users/francescoferrini/Desktop/data2/IMDB/links.dat'

    # Unisci i contenuti dei file .dat in un unico file
    with open(output_file, 'w') as combined_file:
        for file_path in file_paths:
            with open(file_path, 'r') as file:
                combined_file.write(file.read())
if __name__ == '__main__':
    main()