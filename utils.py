import pickle as pkl
import sys
import torch
from tqdm import tqdm
import structure_optimal as so
import numpy as np


def get_position_encoding(seq_len, embed):
    pe = np.array([[pos / (10000.0 ** (i // 2 * 2.0 / embed)) for i in range(embed)] for pos in range(seq_len)])  # 公式实现
    pe[:, 0::2] = np.sin(pe[:, 0::2])
    pe[:, 1::2] = np.cos(pe[:, 1::2])
    return pe


def load_data(dataset_str):
    """
    Loads input data from gcn/data directory
    ind.dataset_str.x => the feature vectors and adjacency matrix of the training instances as list;
    ind.dataset_str.tx => the feature vectors and adjacency matrix of the test instances as list;
    ind.dataset_str.allx => the feature vectors and adjacency matrix of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as list;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    All objects above must be saved using python pickle module.
    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x_adj', 'x_embed', 'y', 'tx_adj', 'tx_embed', 'ty', 'allx_adj', 'allx_embed', 'ally']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x_adj, x_embed, y, tx_adj, tx_embed, ty, allx_adj, allx_embed, ally = tuple(objects)
    # train_idx_ori = parse_index_file("data/{}.train.index".format(dataset_str))
    # train_size = len(train_idx_ori)

    train_adj = []
    train_embed = []
    test_adj = []
    test_embed = []

    for i in range(len(y)):
        adj = x_adj[i].toarray()
        embed = np.array(x_embed[i])
        train_adj.append(adj)
        train_embed.append(embed)

    for i in range(len(y), len(ally)):  # train_size):
        adj = allx_adj[i].toarray()
        embed = np.array(allx_embed[i])
        train_adj.append(adj)
        train_embed.append(embed)

    for i in range(len(ty)):
        adj = tx_adj[i].toarray()
        embed = np.array(tx_embed[i])
        test_adj.append(adj)
        test_embed.append(embed)

    train_adj = np.array(train_adj)
    test_adj = np.array(test_adj)
    train_embed = np.array(train_embed)
    test_embed = np.array(test_embed)
    train_y = np.array(ally)
    test_y = np.array(ty)

    return train_adj, train_embed, train_y, test_adj, test_embed, test_y


def load_dpdata(dataset):
    '''
    data = {'train_tree': train_tree,
        'train_feature': train_feature,
        'train_label': train_label,
        'test_tree': test_tree,
        'test_feature': test_feature,
        'test_label': test_label
        }
    :param dataset: dataset_str: Dataset name
    :return: train and test data
    '''
    path = './trees/' + dataset + '.pickle'
    f = open(path, 'rb')
    data = pkl.load(f)
    max_length = 0
    for i in range(0, len(data['train_feature'])):
        if len(data['train_feature'][i]) > max_length:
            max_length = len(data['train_feature'][i])
    for i in range(0, len(data['test_feature'])):
        if len(data['test_feature'][i]) > max_length:
            max_length = len(data['test_feature'][i])
    train_adj = np.array(data['train_tree'])
    train_embed = np.array(data['train_feature'])
    train_y = np.array(data['train_label'])
    test_adj = np.array(data['test_tree'])
    test_embed = np.array(data['test_feature'])
    test_y = np.array(data['test_label'])
    max_length = max(max_length, 500)
    return train_adj, train_embed, train_y, test_adj, test_embed, test_y, max_length


def preprocess_features(features, max_length, stop_index, onehot=False, add=False, position_embedding=True, cut_stop=False):
    # get features
    if add:
        # add transformer pe
        pe = get_position_encoding(max_length + 1, 300)
    else:
        # concat transformer pe
        pe = get_position_encoding(max_length + 1, 512)
    for i in tqdm(range(features.shape[0])):
        if position_embedding:
            if onehot:
                position_feature = torch.zeros(len(features[i]), max_length + 1)
                for word_idx in range(0, len(features[i])):
                    position_feature[word_idx, word_idx] = 1
                position_feature = position_feature.float()
                word_embed = torch.from_numpy(np.array(features[i])).float()
                whole_feature = torch.cat([word_embed, position_feature], 1)
            else:
                position_feature = pe[list(range(0, len(features[i])))]
                position_feature = torch.from_numpy(position_feature).float()
                word_embed = torch.from_numpy(np.array(features[i])).float()
                if add:
                    whole_feature = torch.add(word_embed, position_feature)
                else:
                    whole_feature = torch.cat([word_embed, position_feature], 1)

        else:
            whole_feature = features[i]
        feature = np.array(whole_feature)
        if cut_stop:
            features[i] = preprocess_cutstop(feature, stop_index[i], False)
        else:
            features[i] = feature

    return np.array(list(features))


def normalize_adj(adj):
    # Symmetrically normalize adjacency matrix.
    rowsum = np.array(adj.sum(1))
    with np.errstate(divide='ignore'):
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)


def check_symmetric(a, tol=1e-8):
    # check symmetric
    return np.all(np.abs(a-a.T) < tol)


def preprocess_tree(adj, height=2):
    # build encoding tree
    adj = np.array(adj)
    y = so.PartitionTreeV2(adj)
    if adj.shape[0] <= 2:
        m = 'v1'
    else:
        m = 'v2'
    x = y.build_encoding_tree(mode=m, k=height)
    y = so.get_child_h(y, k=height)
    y = so.map_id(y, k=height)
    return y.tree_node


def preprocess_dptree(dptrees, height=2, cut_stop=False, window=1):
    """
        dptrees: dependency tree
        height: height of tree
        cut_stop: cut the stopwords after building the graph
    """
    stop_list = []
    if cut_stop:
        with open("stoplist.txt", 'r') as stop_list_file:
            for line in stop_list_file:
                line = line.strip()
                stop_list.append(line)
    adjs = []
    alltree = []
    stopword = []
    for i in tqdm(range(len(dptrees))):
        dptree = dptrees[i]
        stopword_index = []
        adj = np.zeros((len(dptree), len(dptree)))
        for item in dptree:
            word_index = item['id'] - 1
            parent = item['parent']
            if cut_stop:
                if item['text'] in stop_list:
                    stopword_index.append(word_index)
            if parent == 0:
                continue
            if window > 0:
                for j in range(1, window + 1):
                    if word_index + j < adj.shape[0]:
                        adj[word_index][word_index + j] = 1
                        adj[word_index + j][word_index] = 1
                    if word_index - j >= 0:
                        adj[word_index][word_index - j] = 1
                        adj[word_index - j][word_index] = 1
            adj[word_index][parent-1] = 1
            adj[parent - 1][word_index] = 1
        if cut_stop:
            adj = preprocess_cutstop(adj, stopword_index, True)
        adjs.append(adj)
        tree = preprocess_tree(adj, height)
        stopword.append(stopword_index)
        alltree.append(tree)
    return adjs, alltree, stopword


def check(a):
    # check the adj matrix
    if (~a.any(axis=0)).any():
        return np.where(~a.any(axis=0))[0]
    return -1


def preprocess_cutstop(data, stopindex, adj=False):
    # cut the stopwords
    all_row = set(range(0, data.shape[0]))
    stop_row = set(stopindex)
    left_row = list(all_row - stop_row)
    left_row = np.array(left_row).astype(np.int32)
    if adj:
        new_data = data[left_row][:, left_row]
    else:
        new_data = data[left_row]
    return new_data


def preprocess_adj(adj, height=2):
    alltree_list = []

    # build tree and normalize
    for i in tqdm(range(adj.shape[0])):
        tree = preprocess_tree(adj[i], height)
        alltree_list.append(tree)

        adj_normalized = normalize_adj(adj[i])
        adj[i] = adj_normalized

    return np.array(list(adj)), alltree_list



