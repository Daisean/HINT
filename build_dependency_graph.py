import os
import stanza
import pickle
import sys
import random
import numpy as np
from tqdm import tqdm

# change your mkdir path of stanfordnlp model
nlp = stanza.Pipeline(lang='en', model_dir='D:\zhangchong\projects\exp\exp_for_git\OpenNRE\stanza_resources')
if len(sys.argv) < 1:
    sys.exit("Use: python build_dependency_graph.py <dataset>")

dataset = sys.argv[1]

# load pre-trained word embeddings
word_embeddings_dim = 300
word_embeddings = {}

with open('glove.6B.' + str(word_embeddings_dim) + 'd.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        data = line.split()
        word_embeddings[str(data[0])] = list(map(float, data[1:]))

# load label list
doc_name_list = []
doc_train_list = []
doc_test_list = []
with open('data/' + dataset + '.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        doc_name_list.append(line.strip())
        temp = line.split("\t")

        if temp[1].find('test') != -1:
            doc_test_list.append(line.strip())
        elif temp[1].find('train') != -1:
            doc_train_list.append(line.strip())

# load content
doc_content_list = []
with open('data/corpus/' + dataset + '.clean.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        doc_content_list.append(line.strip())

# map and shuffle
train_ids = []
for train_name in doc_train_list:
    train_id = doc_name_list.index(train_name)
    train_ids.append(train_id)
random.shuffle(train_ids)

test_ids = []
for test_name in doc_test_list:
    test_id = doc_name_list.index(test_name)
    test_ids.append(test_id)
random.shuffle(test_ids)

ids = train_ids + test_ids

# train + test
shuffle_doc_name_list = []
shuffle_doc_words_list = []
for i in ids:
    shuffle_doc_name_list.append(doc_name_list[int(i)])
    shuffle_doc_words_list.append(doc_content_list[int(i)])

# build label list
label_set = set()
for doc_meta in shuffle_doc_name_list:
    temp = doc_meta.split('\t')
    label_set.add(temp[2])
label_list = list(label_set)

train_size = len(train_ids)
test_size = len(test_ids)


def build_dptree(start, end):
    # build dptree data
    trees = []
    labels = []
    features = []

    for i in tqdm(range(start, end)):
        content = shuffle_doc_words_list[i]
        doc = nlp(content)
        doc_dt = []
        doc_feature = []
        # trees features
        bias = 0

        for sentence in doc.sentences:
            word_num = 0
            for word in sentence.words:
                word_num += 1
                doc_feature.append(word_embeddings[word.text] if word.text in word_embeddings else np.random.uniform(-0.01, 0.01, word_embeddings_dim))
                if word.head == 0:
                    deptree = {'id': int(word.id) + bias,
                               'text': word.text,
                               'parent': word.head}
                else:
                    deptree = {'id': int(word.id) + bias,
                               'text': word.text,
                               'parent': word.head + bias}
                doc_dt.append(deptree)
            bias += word_num
        trees.append(doc_dt)
        features.append(doc_feature)

    # labels
    for i in range(start, end):
        doc_meta = shuffle_doc_name_list[i]
        temp = doc_meta.split('\t')
        label = temp[2]
        one_hot = [0 for l in range(len(label_list))]
        label_index = label_list.index(label)
        one_hot[label_index] = 1
        labels.append(one_hot)
    labels = np.array(labels)

    return trees, features, labels

print('building graphs for training')
train_tree, train_feature, train_label = build_dptree(0, train_size)
print('building graphs for test')
test_tree, test_feature, test_label = build_dptree(train_size, train_size + test_size)
data = {'train_tree': train_tree,
        'train_feature': train_feature,
        'train_label': train_label,
        'test_tree': test_tree,
        'test_feature': test_feature,
        'test_label': test_label
        }
root_path = 'trees/'
if not os.path.exists(root_path):
    os.makedirs(root_path)
fout = open(root_path + dataset + '.pickle', 'wb')
pickle.dump(data, fout)
