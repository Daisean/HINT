from utils import load_data, preprocess_adj, preprocess_features, load_dpdata, preprocess_dptree
import pickle
import os
import copy
import torch
import random
import argparse
import sys

def load_adj_tree(dataset, height, refresh=False):
    # only window
    root_path = 'trees/window/' + dataset

    if not refresh:
        if os.path.exists(root_path + '_train_' + str(height) + '.pickle'):
            return

    # Load and some preprocessing
    train_adj, train_feature, train_y, test_adj, test_feature, test_y = load_data(dataset)
    print('loading training set')
    train_adj, train_tree, stop_index = preprocess_adj(train_adj, height)
    train_feature = preprocess_features(train_feature, stop_index, False)
    print('loading test set')
    test_adj, test_tree, stop_index = preprocess_adj(test_adj, height)
    test_feature = preprocess_features(test_feature, stop_index, False)

    train_data = {'adj': train_adj, 'tree': train_tree, 'label': train_y, 'feature': train_feature}
    test_data = {'adj': test_adj, 'tree': test_tree, 'label': test_y, 'feature': test_feature}

    f_train = open(root_path + '_train_' + str(height) + '.pickle', 'wb')
    f_test = open(root_path + '_test_' + str(height) + '.pickle', 'wb')
    pickle.dump(train_data, f_train)
    pickle.dump(test_data, f_test)


def load_dep_tree(dataset, height, onehot=False, add=False, key=False, window=-1):
    # load dependency tree , then build train and test
    if key:
        if onehot:
            root_path = 'trees/key/onehot/' + dataset
        else:
            if add:
                root_path = 'trees/key/add/' + dataset
            else:
                root_path = 'trees/key/concat/' + dataset
    else:
        if onehot:
            root_path = 'trees/dependency/onehot/' + dataset
        else:
            if add:
                root_path = 'trees/dependency/add/' + dataset
            else:
                root_path = 'trees/dependency/concat/' + dataset

    # Load and Some preprocessing
    train_dptree, train_feature, train_y, test_dptree, test_feature, test_y, max_length = load_dpdata(dataset)
    print('loading training set')
    train_adj, train_tree, stop_index = preprocess_dptree(train_dptree, height, key, window)
    train_feature = preprocess_features(train_feature, max_length, stop_index, onehot, add, True, cut_stop=key)
    print('loading test set')
    test_adj, test_tree, stop_index = preprocess_dptree(test_dptree, height, key, window)
    test_feature = preprocess_features(test_feature, max_length, stop_index, onehot, add, True, cut_stop=key)

    train_data = {'adj': train_adj, 'tree': train_tree, 'label': train_y, 'feature': train_feature}
    test_data = {'adj': test_adj, 'tree': test_tree, 'label': test_y, 'feature': test_feature}

    f_train = open(root_path + '_train_' + str(height) + '.pickle', 'wb')
    f_test = open(root_path + '_test_' + str(height) + '.pickle', 'wb')
    pickle.dump(train_data, f_train)
    pickle.dump(test_data, f_test)


def update_node(tree):
    # update tree node info
    ids = [v.ID for k, v in tree.items()]
    ids.sort()
    new_tree = {}
    for k, v in tree.items():
        n = copy.deepcopy(v)
        n.ID = ids.index(n.ID)
        if n.parent is not None:
            n.parent = ids.index(n.parent)
        if n.children is not None:
            n.children = [ids.index(c) for c in n.children]
        new_tree[n.ID] = n
    return new_tree


def load_tree(dataset, tree_deepth, input_dim, traindata=True, mode='window'):
    if traindata:
        f_train = open('trees/' + mode + '/' + dataset + '_train_' + str(tree_deepth) + '.pickle', 'rb')
        data = pickle.load(f_train)
    else:
        f_test = open('trees/' + mode + '/' + dataset + '_test_' + str(tree_deepth) + '.pickle', 'rb')
        data = pickle.load(f_test)
    tree_list = []

    for i in range(0, len(data['adj'])):
        tree = {'label': data['label'][i].argmax(),
                'node_size': [0] * (tree_deepth + 1),
                'leaf_size': data['adj'][i].shape[0],
                'edges': [[] for j in range(tree_deepth + 1)],
                'node_features': torch.zeros(data['adj'][i].shape[0], input_dim),
                'local_degree': [0] * data['adj'][i].shape[0],
                }
        new_tree = update_node(data['tree'][i])

        # mask
        layer_idx = [0]
        for layer in range(tree_deepth + 1):
            mask = torch.zeros(len(data['tree'][i]))
            layer_nodes = [i for i, n in new_tree.items() if n.child_h == layer]
            layer_idx.append(layer_nodes[0] + len(layer_nodes))
            mask[range(layer_idx[layer], layer_idx[layer + 1])] = 1
            tree['node_size'][layer] = len(layer_nodes)

        # edge
        for j, n in new_tree.items():
            if n.child_h > 0:
                n_idx = n.ID - layer_idx[n.child_h]
                c_base = layer_idx[n.child_h - 1]
                tree['edges'][n.child_h].extend([(n_idx, c - c_base) for c in n.children])
                continue

        # node_feature
        leaf_feature = torch.from_numpy(data['feature'][i]).float()

        tree['node_features'] = leaf_feature

        # degree:ignore

        if leaf_feature.shape[0] != tree['node_size'][0]:
            print(1)
        tree_list.append(tree)

    if traindata:
        random.shuffle(tree_list)
        train_idx = int(len(tree_list) * 0.9)
        return tree_list[0:train_idx], tree_list[train_idx:]
    else:
        return tree_list


def get_train_and_test(dataset, tree_deepth, input_dim, mode='dependency', pe='concat'):
    if mode != 'window':
        mode = mode + '/' + pe
    train_tree, val_tree = load_tree(dataset, tree_deepth, input_dim, True, mode)
    test_tree = load_tree(dataset, tree_deepth, input_dim, False, mode)
    return train_tree, val_tree, test_tree


if __name__ == '__main__':
    '''
        onehot is True : onehot pe
        onehot is Flase and add is True: add pe
        onehot is Flase and add is False: concat pe
        
        output path: tree/key/xxx or tree/dependency/xxx
    '''
    parser = argparse.ArgumentParser(description='building encoding tree by so')
    parser.add_argument('-d', '--dataset', type=str, default="mr",
                        help='name of dataset (default: MUTAG)')
    parser.add_argument('-k', '--tree_deepth', type=int, default=2,
                        help='the deepth of coding tree (default: 2)')
    parser.add_argument('-o', '--onehot', type=bool, default=True,
                        help='onehot pe (default: True)')
    parser.add_argument('-a', '--add', type=bool, default=False,
                        help='add pe (default: False)')
    parser.add_argument('-s', '--stop', type=bool, default=False,
                        help='')
    args = parser.parse_args()

    load_dep_tree(args.dataset, args.tree_deepth, args.onehot, args.add, args.stop)


