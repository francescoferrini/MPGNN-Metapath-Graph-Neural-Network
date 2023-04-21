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

def main(args):
    colors = {}
    node = {}
    links = {}
    triplets = []
    color_distribution = {'red' : 0, 'blue' :0, 'green': 0, 'purple': 0, 'orange': 0, 'yellow': 0}
    label_distribution = {'+': 0, '-':0}
    label1_distribution = {'+': 0, '-':0}
    label2_distribution = {'+': 0, '-':0}
    label3_distribution = {'+': 0, '-':0}


    # setting the possible colors
    color_list = ['red', 'blue', 'green', 'purple', 'orange', 'yellow'] # red = 010, blue = 100, green = 001


    num_nodes = args.num_nodes

    # METAPATH 1 ##################################### 
    metapath = args.metapath

    metapath_split = metapath.split('-')
    metapath = ''
    order_colors = []
    for elm in metapath_split:
        if len(elm) == 1: # this is a relation
            metapath = metapath + elm
        else:
            if elm == 'red':
                order_colors.append(0)
            elif elm == 'blue':
                order_colors.append(1)
            elif elm == 'green':
                order_colors.append(2)
            elif elm == 'purple':
                order_colors.append(3)
            elif elm == 'orange':
                order_colors.append(4)
            elif elm == 'yellow':
                order_colors.append(5)
    colors_dict = {
        0: 'red',
        1: 'blue',
        2: 'green',
        3: 'purple',
        4: 'orange',
        5: 'yelow'
    }
    print(metapath)
    order_colors.reverse()
    print('colors:', order_colors)
    metapath_length = len(metapath)
    meta = []
    for i in range(0, metapath_length):
        meta.append(int(metapath[i]))
    meta.reverse()
    print('metapath1: ', meta)

    # METAPATH 2 #####################################   
    metapath2 = args.metapath2

    metapath2_split = metapath2.split('-')
    metapath2 = ''
    order_colors2 = []
    for elm in metapath2_split:
        if len(elm) == 1: # this is a relation
            metapath2 = metapath2 + elm
        else:
            if elm == 'red':
                order_colors2.append(0)
            elif elm == 'blue':
                order_colors2.append(1)
            elif elm == 'green':
                order_colors2.append(2)
            elif elm == 'purple':
                order_colors2.append(3)
            elif elm == 'orange':
                order_colors2.append(4)
            elif elm == 'yellow':
                order_colors2.append(5)
    colors_dict = {
        0: 'red',
        1: 'blue',
        2: 'green',
        3: 'purple',
        4: 'orange',
        5: 'yelow'
    }
    print(metapath2)
    order_colors2.reverse()
    print('colors:', order_colors2)
    metapath2_length = len(metapath2)
    meta2 = []

    for i in range(0, metapath2_length):
        meta2.append(int(metapath2[i]))
    meta2.reverse()
    print('metapath2: ', meta2)

    #####################################

    # METAPATH 3 ##################################### 
    metapath3 = args.metapath3

    metapath3_split = metapath3.split('-')
    metapath3 = ''
    order_colors3 = []
    for elm in metapath3_split:
        if len(elm) == 1: # this is a relation
            metapath3 = metapath3 + elm
        else:
            if elm == 'red':
                order_colors3.append(0)
            elif elm == 'blue':
                order_colors3.append(1)
            elif elm == 'green':
                order_colors3.append(2)
            elif elm == 'purple':
                order_colors3.append(3)
            elif elm == 'orange':
                order_colors3.append(4)
            elif elm == 'yellow':
                order_colors3.append(5)
    colors_dict = {
        0: 'red',
        1: 'blue',
        2: 'green',
        3: 'purple',
        4: 'orange',
        5: 'yelow'
    }
    print(metapath3)
    order_colors3.reverse()
    print('colors:', order_colors3)
    metapath3_length = len(metapath3)
    meta3 = []
    for i in range(0, metapath3_length):
        meta3.append(int(metapath3[i]))
    meta3.reverse()
    print('metapath1: ', meta3)

    #####################################

    for i in range(0, num_nodes):
        col = random.choice(color_list)
        col_list = []
        if col == 'red':
            col_list.append(0)
            col_list.append(1)
            col_list.append(0)
            col_list.append(0)
            col_list.append(0)
            col_list.append(0)
            node[i] = col_list
            colors[i] = 0
        elif col == 'blue':
            colors[i] = 1
            col_list.append(1)
            col_list.append(0)
            col_list.append(0)
            col_list.append(0)
            col_list.append(0)
            col_list.append(0)
            node[i] = col_list
        elif col == 'green':
            colors[i] = 2
            col_list.append(0)
            col_list.append(0)
            col_list.append(1)
            col_list.append(0)
            col_list.append(0)
            col_list.append(0)
            node[i] = col_list
        elif col == 'purple':
            col_list.append(0)
            col_list.append(0)
            col_list.append(0)
            col_list.append(1)
            col_list.append(0)
            col_list.append(0)
            node[i] = col_list
            colors[i] = 3
        elif col == 'orange':
            colors[i] = 4
            col_list.append(0)
            col_list.append(0)
            col_list.append(0)
            col_list.append(0)
            col_list.append(1)
            col_list.append(0)
            node[i] = col_list
        elif col == 'yellow':
            colors[i] = 5
            col_list.append(0)
            col_list.append(0)
            col_list.append(0)
            col_list.append(0)
            col_list.append(0)
            col_list.append(1)
            node[i] = col_list

        if colors[i] == 0:
            color_distribution['red'] += 1
        elif colors[i] == 1:
            color_distribution['blue'] += 1
        elif colors[i] == 2:
            color_distribution['green'] += 1
        if colors[i] == 3:
            color_distribution['purple'] += 1
        elif colors[i] == 4:
            color_distribution['orange'] += 1
        elif colors[i] == 5:
            color_distribution['yellow'] += 1
        links[i] = random.randint(10, args.max_rel_for_node)
        copy_links = links.copy()
    print('Color distribution: ', color_distribution)


    # links
    for i in range(0, num_nodes):
        src = i
        num_links = links[i]
        # for p in range(0, 5):
        #     rel = []
        #     rel.append(src)
        #     rel.append(1)
        #     rel.append(choice([t for t in range(0, num_nodes-1) if t not in [src]]))
        #     triplets.append(rel)
        for j in range(0, num_links):
            rel = []
            dst = choice([t for t in range(0, num_nodes-1) if t not in [src]])

            #create the triplet
            rel.append(src)
            rel.append(random.randint(0, args.num_rel_types - 1))
            rel.append(dst)
            triplets.append(rel)

            #print(copy_links[dst])
            copy_links[dst] = copy_links[dst] - 1
        copy_links[src] = 0

    embeddings_list = []
     
    # meta = [3, 2, 1, 0]
    # METAPATH 1 ############################
    print('metapath 1')
    for emb_num in range(0, metapath_length):
        if emb_num == 0:
            #print('primo: ', emb_num)
            print(colors_dict[order_colors[emb_num+1]], '->', meta[emb_num], '->', colors_dict[order_colors[emb_num]])
            current_embedding = {}
            for i in range(0, num_nodes):
                current_embedding[i] = 0               
            for t in triplets:
                if colors[t[0]] == order_colors[emb_num+1] and t[1] == meta[emb_num] and colors[t[2]] == order_colors[emb_num]: 
                        current_embedding[t[0]] = 1
            embeddings_list.append(current_embedding)
        elif emb_num == metapath_length-1:
            #print('ultimo: ', emb_num)
            print('+ ->', meta[emb_num], '->', colors_dict[order_colors[emb_num]])
            next_embedding = {}
            for i in range(0, num_nodes):
                next_embedding[i] = 0
            for t in triplets:
                if t[1] == meta[emb_num] and colors[t[2]] == order_colors[emb_num] and current_embedding[t[2]] == 1: 
                    next_embedding[t[0]] = 1
            embeddings_list.append(next_embedding)
        else:
            #print('intermedio: ', emb_num)
            print(colors_dict[order_colors[emb_num+1]], '->', meta[emb_num], '->', colors_dict[order_colors[emb_num]])
            next_embedding = {}
            for i in range(0, num_nodes):
                next_embedding[i] = 0
            for t in triplets:
                if t[1] == meta[emb_num] and colors[t[2]] == order_colors[emb_num] and current_embedding[t[2]] == 1 and colors[t[0]] == order_colors[emb_num+1]: 
                    next_embedding[t[0]] = 1
            current_embedding = next_embedding
            embeddings_list.append(current_embedding)
    label1 = next_embedding
    # METAPATH 2 ############################
    print('metapath 2')
    for emb_num in range(0, metapath2_length):
        if emb_num == 0:
            #print('primo: ', emb_num)
            print(colors_dict[order_colors2[emb_num+1]], '->', meta2[emb_num], '->', colors_dict[order_colors2[emb_num]])
            current_embedding = {}
            for i in range(0, num_nodes):
                current_embedding[i] = 0               
            for t in triplets:
                if colors[t[0]] == order_colors2[emb_num+1] and t[1] == meta2[emb_num] and colors[t[2]] == order_colors2[emb_num]: 
                        current_embedding[t[0]] = 1
            embeddings_list.append(current_embedding)
        elif emb_num == metapath2_length-1:
            #print('ultimo: ', emb_num)
            print('+ ->', meta2[emb_num], '->', colors_dict[order_colors2[emb_num]])
            next_embedding = {}
            for i in range(0, num_nodes):
                next_embedding[i] = 0
            for t in triplets:
                if t[1] == meta2[emb_num] and colors[t[2]] == order_colors2[emb_num] and current_embedding[t[2]] == 1: 
                    next_embedding[t[0]] = 1
            embeddings_list.append(next_embedding)
        else:
            #print('intermedio: ', emb_num)
            print(colors_dict[order_colors2[emb_num+1]], '->', meta2[emb_num], '->', colors_dict[order_colors2[emb_num]])
            next_embedding = {}
            for i in range(0, num_nodes):
                next_embedding[i] = 0
            for t in triplets:
                if t[1] == meta2[emb_num] and colors[t[2]] == order_colors2[emb_num] and current_embedding[t[2]] == 1 and colors[t[0]] == order_colors2[emb_num+1]: 
                    next_embedding[t[0]] = 1
            current_embedding = next_embedding
            embeddings_list.append(current_embedding)
    label2 = next_embedding

    # METAPATH 3 ############################
    print('metapath 3')
    for emb_num in range(0, metapath3_length):
        if emb_num == 0:
            #print('primo: ', emb_num)
            print(colors_dict[order_colors3[emb_num+1]], '->', meta3[emb_num], '->', colors_dict[order_colors3[emb_num]])
            current_embedding = {}
            for i in range(0, num_nodes):
                current_embedding[i] = 0               
            for t in triplets:
                if colors[t[0]] == order_colors3[emb_num+1] and t[1] == meta3[emb_num] and colors[t[2]] == order_colors3[emb_num]: 
                        current_embedding[t[0]] = 1
            embeddings_list.append(current_embedding)
        elif emb_num == metapath3_length-1:
            #print('ultimo: ', emb_num)
            print('+ ->', meta3[emb_num], '->', colors_dict[order_colors3[emb_num]])
            next_embedding = {}
            for i in range(0, num_nodes):
                next_embedding[i] = 0
            for t in triplets:
                if t[1] == meta3[emb_num] and colors[t[2]] == order_colors3[emb_num] and current_embedding[t[2]] == 1: 
                    next_embedding[t[0]] = 1
            embeddings_list.append(next_embedding)
        else:
            #print('intermedio: ', emb_num)
            print(colors_dict[order_colors3[emb_num+1]], '->', meta3[emb_num], '->', colors_dict[order_colors3[emb_num]])
            next_embedding = {}
            for i in range(0, num_nodes):
                next_embedding[i] = 0
            for t in triplets:
                if t[1] == meta3[emb_num] and colors[t[2]] == order_colors3[emb_num] and current_embedding[t[2]] == 1 and colors[t[0]] == order_colors3[emb_num+1]: 
                    next_embedding[t[0]] = 1
            current_embedding = next_embedding
            embeddings_list.append(current_embedding)
    label3 = next_embedding
    ############################
  
    for l in label1:
        if label1[l] == 0:
            label1_distribution['-'] += 1
        else:
            label1_distribution['+'] += 1
    print(label1_distribution)

    for l in label2:
        if label2[l] == 0:
            label2_distribution['-'] += 1
        else:
            label2_distribution['+'] += 1
    print(label2_distribution)

    for l in label3:
        if label3[l] == 0:
            label3_distribution['-'] += 1
        else:
            label3_distribution['+'] += 1
    print(label3_distribution)

    for k, v in label2.items():
        if v == 1:
            label1[k] = 1

    for k, v in label3.items():
        if v == 1:
            label1[k] = 1 

    for l in label1:
        if label1[l] == 0:
            label_distribution['-'] += 1
        else:
            label_distribution['+'] += 1
    print(label_distribution)

    if args.colors == 'yes':
    # Create a directory to save files
        path = '/Users/francescoferrini/VScode/MultirelationalGNN/data/synthetic_multi/tot_rel_' + str(args.num_rel_types)
    else: 
        path = '/Users/francescoferrini/VScode/MultirelationalGNN/data/synthetic_multi/tot_rel_' + str(args.num_rel_types)
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
    node_dir_path = path +'/node.dat'
    links_dir_path = path + '/link.dat'
    label_dir_path = path + '/label.dat'
    embeddings_path = path + '/embedding.dat'
    meta_path = path + '/metapath.dat'

    with open(node_dir_path, 'w+') as f:
        for i in range(0, num_nodes):
            f.write(str(i) + '\t' + 
                    str(node[i][0]) + ' ' + 
                    str(node[i][1]) + ' ' + 
                    str(node[i][2]) + ' ' +
                    str(node[i][3]) + ' ' + 
                    str(node[i][4]) + ' ' + 
                    str(node[i][5]) + '\n')
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
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='graph creation')
    parser.add_argument("--num_nodes", type=int, required=True,
            help="number of nodes")
    parser.add_argument("--colors", type=str, required=True,
            help="colors")
    parser.add_argument("--num_rel_types", type=int, required=True,
            help="number of relation types")
    parser.add_argument("--max_rel_for_node", type=int, required=True,
            help="maximum number of outgoing edges for node")
    parser.add_argument("--metapath", type=str, required=True,
            help="target metapath")
    parser.add_argument("--metapath2", type=str, required=True,
            help="target metapath 2")
    parser.add_argument("--metapath3", type=str, required=False,
            help="target metapath 3")
    


    args = parser.parse_args()
    #print(args, flush=True)
    main(args)