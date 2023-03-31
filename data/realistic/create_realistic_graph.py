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
from random import choice, shuffle, sample
import argparse

def main(args):
    colors = {}
    node = {}
    links = {}
    triplets = []
    color_distribution = {'red' : 0, 'blue' :0, 'green': 0}
    label_distribution = {'+': 0, '-':0}

    # setting the possible colors
    color_list = ['red', 'blue', 'green'] # red = 010, blue = 100, green = 001


    num_nodes = args.num_nodes

    # Create a list for the metapath a-green-b-blue-c-red
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
    print(metapath)
    order_colors.reverse()
    print('colors:', order_colors)
    metapath_length = len(metapath)
    metapath_dict = {}
    meta = []
    for elm in metapath:
        meta.append(int(elm))
    meta.reverse()
    print('metapath: ', meta)

    for i in range(0, num_nodes):
        col = random.choice(color_list)
        col_list = []
        if col == 'red':
            col_list.append(0)
            col_list.append(1)
            col_list.append(0)
            node[i] = col_list
            colors[i] = 0
        elif col == 'blue':
            colors[i] = 1
            col_list.append(1)
            col_list.append(0)
            col_list.append(0)
            node[i] = col_list
        elif col == 'green':
            colors[i] = 2
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
        links[i] = random.randint(2, args.max_node_degree)
        copy_links = links.copy()
    print('Color distribution: ', color_distribution)

    # set contraint on relation types
    # color_constraint_dict = {
    #     0 : [0, 1, 9, 3, 4, 8],
    #     1 : [5, 6, 4, 0, 1, 2, 8],
    #     2 : [7, 8, 9, 2, 1],
    #     3 : [9, 0, 3, 7, 6, 2, 1],
    #     4 : [5, 2, 4, 7, 1],
    #     5 : [1, 7, 9, 3, 11, 0, 8]
    # }

    color_constraint_dict = {
        '00' : 0,
        '01' : 1,
        '02' : 2,
        '10' : 3,
        '11' : 4,
        '12' : 5,
        '20' : 6,
        '21' : 7,
        '22' : 8
    }
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
            #rel.append(random.randint(0, args.num_rel_types - 1))
            #rel.append(sample(color_constraint_dict[colors[src]], 1)[0])
            rel.append(color_constraint_dict[str(colors[src]) + str(colors[dst])])
            rel.append(dst)
            triplets.append(rel)

            #print(copy_links[dst])
            copy_links[dst] = copy_links[dst] - 1
        copy_links[src] = 0

    embeddings_list = []
     
    if args.colors == 'yes':
        # meta = [3, 2, 1, 0]
        for emb_num in range(0, metapath_length):
            if emb_num == 0:
                print('primo: ', emb_num)
                current_embedding = {}
                for i in range(0, num_nodes):
                    current_embedding[i] = 0               
                for t in triplets:
                    if colors[t[0]] == order_colors[emb_num+1] and t[1] == meta[emb_num] and colors[t[2]] == order_colors[emb_num]: 
                        current_embedding[t[0]] = 1
                embeddings_list.append(current_embedding)
            elif emb_num == metapath_length-1:
                print('ultimo: ', emb_num)
                next_embedding = {}
                for i in range(0, num_nodes):
                    next_embedding[i] = 0
                for t in triplets:
                    if t[1] == meta[emb_num] and colors[t[2]] == order_colors[emb_num] and current_embedding[t[2]] == 1: 
                        next_embedding[t[0]] = 1
                embeddings_list.append(next_embedding)
            else:
                print('intermedio: ', emb_num)
                next_embedding = {}
                for i in range(0, num_nodes):
                    next_embedding[i] = 0
                for t in triplets:
                    if t[1] == meta[emb_num] and colors[t[2]] == order_colors[emb_num] and current_embedding[t[2]] == 1 and colors[t[0]] == order_colors[emb_num+1]: 
                        next_embedding[t[0]] = 1
                current_embedding = next_embedding
                embeddings_list.append(current_embedding)
    else:
        for emb_num in range(0, metapath_length):
            if emb_num == 0:
                print('primo: ', emb_num)
                current_embedding = {}
                for i in range(0, num_nodes):
                    current_embedding[i] = 0               
                for t in triplets:
                    if t[1] == meta[emb_num] and colors[t[2]] == order_colors[emb_num]: 
                        current_embedding[t[0]] = 1
                embeddings_list.append(current_embedding)
            elif emb_num == metapath_length-1:
                print('ultimo: ', emb_num)
                next_embedding = {}
                for i in range(0, num_nodes):
                    next_embedding[i] = 0
                for t in triplets:
                    if t[1] == meta[emb_num] and current_embedding[t[2]] == 1: 
                        next_embedding[t[0]] = 1
                embeddings_list.append(next_embedding)
            else:
                print('intermedio: ', emb_num)
                next_embedding = {}
                for i in range(0, num_nodes):
                    next_embedding[i] = 0
                for t in triplets:
                    if t[1] == meta[emb_num] and current_embedding[t[2]] == 1: 
                        next_embedding[t[0]] = 1
                current_embedding = next_embedding
                embeddings_list.append(current_embedding)



    for l in next_embedding:
        if next_embedding[l] == 0:
            label_distribution['-'] += 1
        else:
            label_distribution['+'] += 1
    print(label_distribution)

    if args.colors == 'yes':
    # Create a directory to save files
        path = '/Users/francescoferrini/VScode/MultirelationalGNN/data/realistic/complex/length_m_' + str(metapath_length) + '__tot_rel_' + str(args.num_rel_types)
    else: 
        path = '/Users/francescoferrini/VScode/MultirelationalGNN/data/simple/length_m_' + str(metapath_length) + '__tot_rel_' + str(args.num_rel_types)
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
                    str(node[i][2]) + '\n')
    f.close()

    with open(links_dir_path, 'w+') as f:
        for t in triplets:
            f.write(str(t[0]) + '\t' + str(t[1]) + '\t' + str(t[2]) + '\n')
    f.close()

    with open(label_dir_path, 'w+') as f:
        for i in range(0, len(next_embedding)):
            f.write(str(i) + '\t' + str(next_embedding[i]) + '\n')
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
    parser.add_argument("--max_node_degree", type=int, required=True,
            help="maximum number of outgoing edges for node")
    parser.add_argument("--metapath", type=str, required=True,
            help="target metapath")


    args = parser.parse_args()
    #print(args, flush=True)
    main(args)