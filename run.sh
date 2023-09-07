#!/bin/bash

dataset="synthetic" # 'synthetic' or 'fb15k-237'
hidden_dim=64

if [ "${dataset}" == "fb15k-237" ]; then
    folder="../data/${dataset}/"
    node_file="$folder""node_bow.dat"
    link_file="$folder""link.dat"
    label_file="$folder""label.dat"
    relations_legend_file="$folder""relations_legend.dat"
    pickle_filename="$folder""iteration_variables.pkl"
elif [ "${dataset}" == "DBLP" ] || [ "${dataset}" == "IMDB" ] || [ "${dataset}" == "ACM" ]; then
    folder="../data/${dataset}/"
    node_file="$folder""node_bow.dat"
    link_file="$folder""link.dat"
    label_file=""
    relations_legend_file=""
    pickle_filename=""
elif [ "${dataset}" == "synthetic" ]; then
    metapath_length=3
    overlap=0 #0 or 1 or 2 or 3
    shared_relations=0 #0 or 1 or 2 or 3
    folder="../data/${dataset}/metapath_length_${metapath_length}/overlap_${overlap}rels_${shared_relations}/"
    node_file="$folder""node.dat"
    link_file="$folder""link.dat"
    label_file="$folder""label.dat"
    relations_legend_file=""
    pickle_filename=""
fi

mpiexec -n 10 python main.py --hidden_dim "${hidden_dim}" --dataset "${dataset}" --folder "${folder}" --node_file "${node_file}" --link_file "${link_file}" --label_file "${label_file}" --relations_legend_file "${relations_legend_file}" --pickle_filename "${pickle_filename}" 