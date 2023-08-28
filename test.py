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
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    num_iterations = 5  # Numero di iterazioni desiderato

    for iteration in range(num_iterations):
        if rank == 0:
            data_to_send = f"Dati per l'iterazione {iteration}"
            for dest_rank in range(1, size):
                comm.send(data_to_send, dest=dest_rank)
            print(f"Processo padre ha inviato: {data_to_send}")
        else:
            received_data = comm.recv(source=0)
            print(f"Processo figlio {rank} ha ricevuto: {received_data}")
        
        # Sincronizza tutti i processi prima della prossima iterazione
        comm.Barrier()

    print(f"Processo {rank} ha completato tutte le iterazioni.")
if __name__ == '__main__':
    main()