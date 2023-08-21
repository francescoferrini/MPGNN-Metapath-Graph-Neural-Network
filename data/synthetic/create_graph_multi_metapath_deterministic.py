'''
File to create graph.
Running the code, generates 3 files:
- node.dat: node features for each node
- links.dat: edges files. Each line is composed by src edge_type dst
- label.dat: label for each node

Specifications:
- The metapath is manually set passing it as an argument.
  Example:
    --metapath rstuv -> In this case we have this metapath of length 5 where the last edge is v

    metapath = 'a-red-b-blue-c-green'   

    }
'''
import os
import random
from random import choice, shuffle
import argparse
import pandas as pd
import numpy as np

def main(args):
    colors = {}
    node = {}
    links = {}
    triplets = []
    color_distribution = {'red' : 0, 'blue' :0, 'green': 0, 'purple': 0, 'orange': 0, 'yellow': 0}
    label_distribution = {'+': 0, '-':0}
    label_distribution2 = {'+': 0, '-':0}
    label3_distribution = {'+': 0, '-':0}


    # setting the possible colors
    color_list = ['red', 'blue'] # red = 010, blue = 100, green = 001
    
    relations_list = []
    # possible combination of relations
    # Create a directory to save files
    data = 'overlap_' + str(args.overlap) + 'rels_' + str(args.shared_relations)
    path = '/Users/francescoferrini/VScode/MultirelationalGNN/data/synthetic/metapath_length_3/'+data

    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
    node_dir_path = path +'/node.dat'
    links_dir_path = path + '/link.dat'
    label_dir_path = path + '/label.dat'
    embeddings_path = path + '/embedding.dat'
    meta_path = path + '/metapath.dat'


    if args.overlap == 0 and args.shared_relations == 0: diff=0
    elif args.overlap == 0 and args.shared_relations ==1: diff=1
    elif args.overlap == 0 and args.shared_relations ==2: diff=2
    elif args.overlap == 0 and args.shared_relations ==3: diff=3
    elif args.overlap == 1 and args.shared_relations ==0: diff=4
    elif args.overlap == 1 and args.shared_relations ==1: diff=5
    elif args.overlap == 1 and args.shared_relations ==2: diff=6
    elif args.overlap == 1 and args.shared_relations ==3: diff=7
    elif args.overlap == 2 and args.shared_relations ==0: diff=8
    elif args.overlap == 2 and args.shared_relations ==1: diff=9
    elif args.overlap == 2 and args.shared_relations ==2: diff=10
    elif args.overlap == 2 and args.shared_relations ==3: diff=11
    elif args.overlap == 3 and args.shared_relations ==0: diff=12
    elif args.overlap == 3 and args.shared_relations ==1: diff=13
    elif args.overlap == 3 and args.shared_relations ==2: diff=14
    elif args.overlap == 3 and args.shared_relations ==3: diff=15

    if diff == 1:
        relations_dict = {
            'red-red': [0, 1],
            'red-blue' : [2, 3],
            'blue-red': [4, 5], 
            'blue-blue': [6, 7]
        }
    elif diff == 0:
        relations_dict = {
            'red-red': [0],
            'red-blue' : [1],
            'blue-red': [2], 
            'blue-blue': [3]
        }
    elif diff == 2:
        relations_dict = {
            'red-red': [0, 1, 2],
            'red-blue' : [3, 4],
            'blue-red': [5, 6, 7], 
            'blue-blue': [8, 9]
        }
    elif diff == 3:
        relations_dict = {
            'red-red': [0, 1, 2],
            'red-blue' : [3, 4, 5],
            'blue-red': [6, 7, 8, 9], 
            'blue-blue': [10, 11, 12, 13]
        }
    ##################
    elif diff == 4:
        relations_dict = {
            'red-red': [0, 1],
            'red-blue' : [1],
            'blue-red': [2, 3], 
            'blue-blue': [2]
        }
    elif diff == 5:
        relations_dict = {
            'red-red': [0, 7],
            'red-blue' : [1, 2],
            'blue-red': [2, 3, 5], 
            'blue-blue': [3, 4]
        }
    elif diff == 6:
        relations_dict = {
            'red-red': [0, 1, 2],
            'red-blue' : [3, 4, 0],
            'blue-red': [5, 6, 7], 
            'blue-blue': [8, 9, 2]
        }
    elif diff == 7:
        relations_dict = {
            'red-red': [0, 1, 2, 9],
            'red-blue' : [3, 4, 5, 10],
            'blue-red': [6, 7, 8, 9], 
            'blue-blue': [10, 11, 12, 13]
        }
    ##################
    elif diff == 8:
        relations_dict = {
            'red-red': [0, 3],
            'red-blue' : [1, 2],
            'blue-red': [2, 3], 
            'blue-blue': [0, 1]
        }
    elif diff == 9:
        relations_dict = {
            'red-red': [0, 1, 5],
            'red-blue' : [1, 2, 7],
            'blue-red': [4, 6, 5], 
            'blue-blue': [7, 0, 3]
        }
    elif diff == 10:
        relations_dict = {
            'red-red': [0, 1, 2, 7],
            'red-blue' : [3, 4, 0],
            'blue-red': [5, 6, 7], 
            'blue-blue': [8, 9, 2, 3]
        }
    elif diff == 11:
        relations_dict = {
            'red-red': [0, 1, 2, 9, 8],
            'red-blue' : [3, 4, 5, 10],
            'blue-red': [6, 7, 8, 9, 11], 
            'blue-blue': [10, 11, 12, 13]
        }
    ##################
    elif diff == 12:
        relations_dict = {
            'red-red': [0, 1, 2, 3],
            'red-blue' : [0, 1, 2, 3],
            'blue-red': [0, 1, 2, 3], 
            'blue-blue': [0, 1, 2, 3]
        }
    elif diff == 13:
        relations_dict = {
            'red-red': [0, 1, 2, 3, 4, 5, 6, 7],
            'red-blue' : [0, 1, 2, 3, 4, 5, 6, 7],
            'blue-red': [0, 1, 2, 3, 4, 5, 6, 7], 
            'blue-blue': [0, 1, 2, 3, 4, 5, 6, 7]
        }
    elif diff == 14:
        relations_dict = {
            'red-red': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            'red-blue' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            'blue-red': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
            'blue-blue': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        }
    elif diff == 15:
        relations_dict = {
            'red-red': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            'red-blue' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            'blue-red': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], 
            'blue-blue': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        }

    num_nodes = args.num_nodes

    # METAPATH 1 ##################################### 
    metapath = args.metapath
    metapath_split = metapath.split('-')
    meta = []
    order_colors = []
    print(metapath_split)
    for elm in metapath_split:
        order_colors.append(color_list.index(elm))
    for i in range(0, len(metapath_split)-1):
        meta.append(random.choice(relations_dict[metapath_split[i] + '-' + metapath_split[i+1]]))

    order_colors.reverse()
    print('colors:', order_colors)
    metapath_length = len(meta)
    meta.reverse()
    print('metapath1: ', meta)

    # METAPATH 2 ##################################### 
    if args.metapath2:
        metapath2 = args.metapath2
        metapath_split2 = metapath2.split('-')
        meta2 = []
        order_colors2 = []
        for elm in metapath_split2:
            order_colors2.append(color_list.index(elm))
        for i in range(0, len(metapath_split2)-1):
            meta2.append(relations_list.index(metapath_split2[i] + '-' + metapath_split2[i+1]))

        order_colors2.reverse()
        metapath_length2 = len(meta2)
        meta2.reverse()

    #####################################
    for i in range(0, num_nodes):
        col = random.choice(color_list)
        one_hot_vector = np.zeros(len(color_list))
        color_index = color_list.index(col)
        one_hot_vector[color_index] = 1
        # Converti il vettore in una lista
        one_hot_list = list(one_hot_vector)
        int_list = [int(num) for num in one_hot_list]
        
        colors[i] = color_index
        node[i] = int_list
        links[i] = random.randint(1, args.max_rel_for_node)
        copy_links = links.copy()



    # links
    for i in range(0, num_nodes):
        src = i
        num_links = links[i]
        for j in range(0, num_links):
            rel = []
            rel.append(src)
            dst = choice([t for t in range(0, num_nodes-1) if t not in [src]])
            src_color = color_list[colors[src]]
            dst_color = color_list[colors[dst]]
            r = src_color + '-' + dst_color
            rel.append(random.choice(relations_dict[r]))
            rel.append(dst)
            triplets.append(rel)
            #print(copy_links[dst])
            copy_links[dst] = copy_links[dst] - 1
        copy_links[src] = 0
    embeddings_list = []
     
    # meta = [3, 2, 1, 0]
    # METAPATH 1 ############################
    for emb_num in range(0, metapath_length):
        if emb_num == 0:
            #print('primo: ', emb_num)
            #print(colors_dict[order_colors[emb_num+1]], '->', meta[emb_num], '->', colors_dict[order_colors[emb_num]])
            current_embedding = {}
            for i in range(0, num_nodes):
                current_embedding[i] = 0               
            for t in triplets:
                if colors[t[0]] == order_colors[emb_num+1] and t[1] == meta[emb_num] and colors[t[2]] == order_colors[emb_num]: 
                        current_embedding[t[0]] = 1
            embeddings_list.append(current_embedding)
        elif emb_num == metapath_length-1:
            #print('ultimo: ', emb_num)
            #print('+ ->', meta[emb_num], '->', colors_dict[order_colors[emb_num]])
            next_embedding = {}
            for i in range(0, num_nodes):
                next_embedding[i] = 0
            for t in triplets:
                if t[1] == meta[emb_num] and colors[t[2]] == order_colors[emb_num] and current_embedding[t[2]] == 1: 
                    next_embedding[t[0]] = 1
            embeddings_list.append(next_embedding)
        else:
            #print('intermedio: ', emb_num)
            #print(colors_dict[order_colors[emb_num+1]], '->', meta[emb_num], '->', colors_dict[order_colors[emb_num]])
            next_embedding = {}
            for i in range(0, num_nodes):
                next_embedding[i] = 0
            for t in triplets:
                if t[1] == meta[emb_num] and colors[t[2]] == order_colors[emb_num] and current_embedding[t[2]] == 1 and colors[t[0]] == order_colors[emb_num+1]: 
                    next_embedding[t[0]] = 1
            current_embedding = next_embedding
            embeddings_list.append(current_embedding)
    label1 = next_embedding
    metapath1_embeddings = []
    for e in embeddings_list:
        metapath1_embeddings.append(list(e.values()))
    metapath1_embeddings.reverse()
    meta1 = meta[::-1]
    meta_c1 = order_colors[::-1]
   ####################################################################################
   # METAPATH 2 ############################
    if args.metapath2:
        for emb_num in range(0, metapath_length2):
            if emb_num == 0:
                #print('primo: ', emb_num)
                #print(colors_dict[order_colors[emb_num+1]], '->', meta[emb_num], '->', colors_dict[order_colors[emb_num]])
                current_embedding = {}
                for i in range(0, num_nodes):
                    current_embedding[i] = 0               
                for t in triplets:
                    if colors[t[0]] == order_colors2[emb_num+1] and t[1] == meta2[emb_num] and colors[t[2]] == order_colors2[emb_num]: 
                            current_embedding[t[0]] = 1
                embeddings_list.append(current_embedding)
            elif emb_num == metapath_length2-1:
                #print('ultimo: ', emb_num)
                #print('+ ->', meta[emb_num], '->', colors_dict[order_colors[emb_num]])
                next_embedding = {}
                for i in range(0, num_nodes):
                    next_embedding[i] = 0
                for t in triplets:
                    if t[1] == meta2[emb_num] and colors[t[2]] == order_colors2[emb_num] and current_embedding[t[2]] == 1: 
                        next_embedding[t[0]] = 1
                embeddings_list.append(next_embedding)
            else:
                #print('intermedio: ', emb_num)
                #print(colors_dict[order_colors[emb_num+1]], '->', meta[emb_num], '->', colors_dict[order_colors[emb_num]])
                next_embedding = {}
                for i in range(0, num_nodes):
                    next_embedding[i] = 0
                for t in triplets:
                    if t[1] == meta2[emb_num] and colors[t[2]] == order_colors2[emb_num] and current_embedding[t[2]] == 1 and colors[t[0]] == order_colors2[emb_num+1]: 
                        next_embedding[t[0]] = 1
                current_embedding = next_embedding
                embeddings_list.append(current_embedding)
        label2 = next_embedding
   ####################################################################################
  
    for l in label1:
        if label1[l] == 0:
            label_distribution['-'] += 1
        else:
            label_distribution['+'] += 1

    if args.metapath2:
        for l in label2:
            if label2[l] == 0:
                label_distribution2['-'] += 1
            else:
                label_distribution2['+'] += 1

    if args.metapath2:
        for k, v in label2.items():
            if v == 1:
                label1[k] = 1

    final_distribution = {'+': 0, '-': 0}
    for l in label1:
        if label1[l] == 0:
            final_distribution['-'] += 1
        else:
            final_distribution['+'] += 1


    if args.metapath:
        metapaths_number = 1
    if args.metapath2:
        metapaths_number = 2
    if args.metapath3:
        metapaths_number = 3
    
    #### sparsification metapath 1
    pd_triplets = pd.DataFrame(triplets, columns=['source', 'relation', 'destination'])
    pd_triplets['bool'] = ''
    pd_triplets['color'] = ''
    pd_triplets['color'] = pd_triplets['destination'].map(colors)
    # itero sugli embeddings
    for i, c in enumerate(metapath1_embeddings):
        # trovo gli indici dei nodi che hanno embedding a 1
        labeled_nodes = [indice for indice, valore in enumerate(c) if valore == 1]
        # itero usl dataframe
        for index, row in pd_triplets.iterrows():
            if row['relation'] == meta1[i] and row['color'] == meta_c1[i] and row['source'] in labeled_nodes:
                pd_triplets.at[index, 'bool'] = True

    for i, c in enumerate(metapath1_embeddings):
        # trovo gli indici dei nodi che hanno embedding a 1
        labeled_nodes = [indice for indice, valore in enumerate(c) if valore == 1]
        # itero usl dataframe
        for index, row in pd_triplets.iterrows():
            if row['relation'] != meta1[i] and row['color'] == meta_c1[i] and row['source'] in labeled_nodes and row['bool'] != True:
                pd_triplets.at[index, 'bool'] = False
    df = pd_triplets.loc[~(pd_triplets['bool'] == False)]
    df = df.drop('bool', axis=1)
    df = df.drop('color', axis=1)
    triplets = df.values.tolist()
    

    with open(node_dir_path, 'w+') as f:
        for i in range(0, num_nodes):
            line = str(i)
            for j in range(len(node[i])):
                line += '\t' + str(node[i][j])
            line += '\n'
            f.write(line)
    f.close()

    

    with open(links_dir_path, 'w+') as f:
        for t in triplets:
            f.write(str(t[0]) + '\t' + str(t[1]) + '\t' + str(t[2]) + '\n')
    f.close()

    with open(label_dir_path, 'w+') as f:
        for i in range(0, len(label1)):
            f.write(str(i) + '\t' + str(label1[i]) + '\n')
    f.close()

    with open(embeddings_path, 'w+') as f:
        for i in range(0, num_nodes):
            f.write(str(i) + '\t')
            for dict in embeddings_list:
                f.write(str(dict[i]) + '\t')
            f.write('\n')
            #f.write(str(embeddings_list[i][0]) + '\t' + str(embeddings_list[i][1]) + '\t' + str(embeddings_list[i][2]) + '\n' )
    f.close()

    with open(meta_path, 'w+') as f:
        f.write(args.metapath)
        f.write('\n')
        for elm in meta:
            f.write(str(elm))
            f.write(' ')
        f.write('\n')
        for elm in order_colors:
            f.write(str(elm))
            f.write(' ')
    f.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='synthetic graph creation')
    parser.add_argument("--num_nodes", type=int, required=True,
            help="number of nodes")
    parser.add_argument("--max_rel_for_node", type=int, required=True,
            help="maximum number of outgoing edges for node")
    parser.add_argument("--metapath", type=str, required=True,
            help="target metapath")
    parser.add_argument("--overlap", type=int, required=True,
            help="relations overlap")
    parser.add_argument("--shared_relations", type=int, required=True,
            help="shared_relations")
    parser.add_argument("--metapath2", type=str, required=False,
            help="target metapath 2")
    parser.add_argument("--metapath3", type=str, required=False,
            help="target metapath 3")
    


    args = parser.parse_args()
    #print(args, flush=True)
    main(args)