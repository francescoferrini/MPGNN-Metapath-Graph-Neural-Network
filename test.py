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
import networkx as nx

def get_best_keys(dictionary):
  best_values = sorted(dictionary.values(), reverse=True)[:1]
  print(best_values)
  return [key for key, value in dictionary.items() if value in best_values]

def main():
    

    dictionary = {
        "a": [1, 26, 3, 4],
        "b": [5, 6, 7, 8],
        "c": [9, 10, 11, 12],
        "d": [13, 45, 15, 16],
        "e": [17, 18, 19, 20],
    }
    sorted_dictionary = sorted(dictionary.items(), key=lambda x: x[1][1])
    print(sorted_dictionary)
    
if __name__ == '__main__':
    main()