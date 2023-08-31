#!/bin/bash

dataset="fb15k-237" # 'synthetic' or 'fb15k-237'
folder="/../data/${dataset}/"

if [ "$dataset" == "fb15k-237" ]; then
    node_file="${folder}node_bow.dat"
else
    node_file="${folder}node.dat"
fi

link_file="${folder}link.dat"
label_file="${folder}label.dat"

if [ "$dataset" == "fb15k-237" ]; then
    relation='/people/person/gender'
    python3 "${folder}data_processing.py" --relation "${relation}"
else
    num_nodes=5000
    max_rel_for_node=10 #max node degree
    metapath='red-red-blue' #example of metapath
    overlap=0 #0 or 1 or 2 or 3
    shared_relations=0 #0 or 1 or 2 or 3
    python3 "${folder}create_graph_multi_metapath_deterministic.py" --num_nodes "${num_nodes}" --max_rel_for_node "${max_rel_for_node}" --metapath "${metapath}" --overlap "${overlap}" --shared_relations "${shared_relations}"
fi