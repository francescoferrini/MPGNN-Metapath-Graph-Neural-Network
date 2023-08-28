#!/bin/bash

dataset="fb15k-237" # 'synthetic' or 'fbk15k-237'
hidden_dim=64

if [ "${dataset}" == "fb15k-237" ]; then
    folder="/Users/francescoferrini/VScode/MultirelationalGNN/data/${dataset}/"
    node_file="$folder""node_bow.dat"
    link_file="$folder""link.dat"
    label_file="$folder""label.dat"
    relations_legend_file="$folder""relations_legend.dat"
    pickle_filename="$folder""iteration_variables.pkl"
elif [ "${dataset}" == "DBLP" ] || [ "${dataset}" == "IMDB" ] || [ "${dataset}" == "ACM" ]; then
    folder="/Users/francescoferrini/Desktop/data2/${dataset}/"
    node_file="$folder""node_bow.dat"
    link_file="$folder""link.dat"
    label_file=""
    relations_legend_file=""
    pickle_filename=""
elif [ "${dataset}" == "synthetic" ]; then
    folder="/Users/francescoferrini/VScode/MultirelationalGNN/data/${dataset}/"
    node_file="$folder""node.dat"
    link_file="$folder""link.dat"
    label_file="$folder""label.dat"
    relations_legend_file=""
    pickle_filename=""
fi

mpiexec -n 3 python main.py --hidden_dim "${hidden_dim}" --dataset "${dataset}" --folder "${folder}" --node_file "${node_file}" --link_file "${link_file}" --label_file "${label_file}" --relations_legend_file "${relations_legend_file}" --pickle_filename "${pickle_filename}" 