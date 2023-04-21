#!/bin/bash

num_rel=5
ll_output_dim=2
n_layers=3
metapath_length=3
MP_GNN='False'
metapath='3210'

dataset="complex"
folder= "/Users/francescoferrini/VScode/MultirelationalGNN/data/${dataset}/length_m_${metapath_length}__tot_rel_${num_rel}"
# node_file="${folder}node.dat"
# link_file="${folder}link.dat"
# label_file="${folder}label.dat"
node_file="/Users/francescoferrini/VScode/MultirelationalGNN/data/complex/length_m_3__tot_rel_5/node.dat"
link_file="/Users/francescoferrini/VScode/MultirelationalGNN/data/complex/length_m_3__tot_rel_5/link.dat"
label_file="/Users/francescoferrini/VScode/MultirelationalGNN/data/complex/length_m_3__tot_rel_5/label.dat"
input_dim=6
hidden_dim=32
output_dim=32 # output dimension for the last RGCN layer (same as hidden_dim)

python /Users/francescoferrini/VScode/MultirelationalGNN/MPGNN/main.py --node_file_path ${node_file} --link_file_path ${link_file} --label_file_path ${label_file} --input_dim ${input_dim} --hidden_dim ${hidden_dim} --output_dim ${output_dim} --num_rel ${num_rel} --ll_output_dim ${ll_output_dim} --n_layers ${n_layers} --metapath_length ${metapath_length} --metapath ${metapath} --MP_GNN ${MP_GNN}
