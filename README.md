# MPGNN

## Datasets
In this project, we have utilized various datasets for training and evaluating the model. Below are the data sources and relevant information for each dataset used.

### Synthetic
The synthetic dataset was created to simulate specific scenarios and test the model under controlled conditions. 
To generate a synthetic dataset, you should navigate inside directory `data` and run:
```sh
bash run_data.sh
```

In the file you can specify:
- **dataset:** synthetic
- **num_nodes:** total number of nodes in the graph
- **max_rel_for_node:** maximum degree for each node
- **metapath:** the correct meta-path to be predicted
- **overlap:** overlapped relations between node types' pair
- **shared_relations:** shared relations between node types' pair

### FB15K-237
The FB15K-237 dataset is used to test our model in a real scenario where the number of relations is high.
To generate the graph, you should navigate inside directory `data` and run:
```sh
bash run_data.sh
```
In the file you can specify:
- **dataset:** fb15k-237
- **relation:** the specific many-to-one relation to be transformed in a label


## Model
In order to run the model, navigate in the main folder and run:
```sh
bash run.sh
```
where in the file you can specify:
- **dataset:**
- **number of parallel processes:** since the mpi4py library is used to parallelize the workflow




