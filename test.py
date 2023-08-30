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
    dizionario = {'[219, 112, 156]': 0.5186344857577735, '[204, 219]': 0.47419004131332904, '[129]': 0.4412144702842377, '[176, 219]': 0.42907662082514736, '[214]': 0.42387152777777776, '[161, 129]': 0.396757457846952}

    chiavi_liste = [eval(chiave) for chiave in dizionario.keys()]

    print(chiavi_liste)
if __name__ == '__main__':
    main()