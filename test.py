from mpi4py import MPI
import numpy as np
from sklearn.cluster import DBSCAN
import torch
from sklearn.metrics import f1_score
import numpy as np
import torch
import pickle
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import pandas as pd
import torch
from torch_sparse import coalesce


def main():
    

    # Define the row index, column index, and value
    total = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 2, 2, 3, 3, 4]], dtype=torch.long)
    value = torch.tensor([1, 2, 3, 4, 5, 6])

   
    # Calculate the total number of nodes in the graph
    nu = 5

    # Apply the coalesce function to combine duplicate elements
    coalesced_index, coalesced_value = coalesce(total, value, m=nu, n=nu, op='add')

    # Print the result
    print(coalesced_index)
    print(coalesced_value)

    '''
    ## Edges
    path = '/Users/francescoferrini/VScode/MultirelationalGNN/data2/DBLP/edges.pkl'
    with open(path, "rb") as file:
        # Caricamento dell'oggetto dal file pickle
        data = pickle.load(file)
    triples = []
    for relation, csc_matrix in enumerate(data):
        # Estrazione delle righe e colonne non nulle dalla matrice CSC
        rows, cols = csc_matrix.nonzero()

        # Costruzione delle triple source_node, relation e destination_node
        for row, col in zip(rows, cols):
            source_node = row
            destination_node = col
            triple = (source_node, relation, destination_node)
            triples.append(triple)

    # Creazione del DataFrame dalle triple
    df = pd.DataFrame(triples, columns=['source_node', 'relation', 'destination_node'])

    # Stampa del DataFrame
    print(df)
    with open("/Users/francescoferrini/VScode/MultirelationalGNN/data2/DBLP/link.dat", "w") as f:
        for index, row in df.iterrows():
            f.write(str(row['source_node']) + '\t' + str(row['relation']) + '\t' + str(row['destination_node']) + '\n')
        f.close()
    
    ## Nodes
    path = '/Users/francescoferrini/VScode/MultirelationalGNN/data2/DBLP/node_features.pkl'
    with open(path, "rb") as file:
        # Caricamento dell'oggetto dal file pickle
        data = pickle.load(file)
    with open("/Users/francescoferrini/VScode/MultirelationalGNN/data2/DBLP/node.dat", "w") as f:
        for i in range(0, len(data)):
            f.write(str(i) + '\t')
            for j in range(0, len(data[i])):
                f.write(str(data[i][j]) + '\t')
            f.write('\n')
        f.close()
    
    ## Labels
    path = '/Users/francescoferrini/VScode/MultirelationalGNN/data2/DBLP/labels.pkl'
    with open(path, "rb") as file:
        # Caricamento dell'oggetto dal file pickle
        data = pickle.load(file)
    
    ### train
    with open("/Users/francescoferrini/VScode/MultirelationalGNN/data2/DBLP/labels_train.dat", "w") as f:
        for i in range(0, len(data[0])):
            f.write(str(data[0][i][0]) + '\t' + str(data[0][i][1]) + '\n')
        f.close()
    ### validation
    with open("/Users/francescoferrini/VScode/MultirelationalGNN/data2/DBLP/labels_val.dat", "w") as f:
        for i in range(0, len(data[1])):
            f.write(str(data[1][i][0]) + '\t' + str(data[1][i][1]) + '\n')
        f.close()
    ### test
    with open("/Users/francescoferrini/VScode/MultirelationalGNN/data2/DBLP/labels_test.dat", "w") as f:
        for i in range(0, len(data[2])):
            f.write(str(data[2][i][0]) + '\t' + str(data[2][i][1]) + '\n')
        f.close()'''
    
if __name__ == '__main__':
    main()