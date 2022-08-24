#!/usr/bin/env python
# encoding: utf-8
import torch
import time
import random
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from build_coding_tree import get_train_and_test
from models.coding_tree_learning import TL

criterion = nn.CrossEntropyLoss()


def train(args, model, device, train_trees, optimizer, epoch):
    model.train()

    total_iters = 0

    indices = np.arange(0, len(train_trees))
    np.random.shuffle(indices)
    loss_accum = 0
    for start in range(0, len(train_trees), args.batch_size):
        total_iters += 1
        end = start + args.batch_size
        if start == len(train_trees) - 1:
            continue
        selected_idx = indices[start:end]
        batch_tree = [train_trees[idx] for idx in selected_idx]
        output = model(batch_tree)
        labels = torch.LongTensor([tree['label'] for tree in batch_tree]).to(device)

        # compute loss
        loss = criterion(output, labels)

        # backprop
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss = loss.detach().cpu().numpy()
        loss_accum += loss

    average_loss = loss_accum / total_iters
    print('epoch: %d' % (epoch), "loss training: %.6f" % (average_loss))

    return average_loss


# pass data to model with minibatch during testing to avoid memory overflow (does not perform backpropagation)
def pass_data_iteratively(model, trees, minibatch_size=64):
    model.eval()
    output = []
    idx = np.arange(len(trees))
    for i in range(0, len(trees), minibatch_size):
        sampled_idx = idx[i:i+minibatch_size]
        if len(sampled_idx) == 0:
            continue
        output.append(model([trees[j] for j in sampled_idx]).detach())
    return torch.cat(output, 0)


def test(args, model, device, train_trees, val_trees, test_trees, epoch):
    model.eval()

    output = pass_data_iteratively(model, train_trees)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([tree['label'] for tree in train_trees]).to(device)
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_train = correct / float(len(train_trees))

    output = pass_data_iteratively(model, val_trees)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([tree['label'] for tree in val_trees]).to(device)
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_val = correct / float(len(val_trees))

    output = pass_data_iteratively(model, test_trees)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([tree['label'] for tree in test_trees]).to(device)
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_test = correct / float(len(test_trees))

    print("accuracy train: %.4f val: %.4f test: %.4f" % (acc_train, acc_val, acc_test))

    return acc_train, acc_val, acc_test


def main():
    # settings
    parser = argparse.ArgumentParser(description='Pytorch HINT for text classification')
    parser.add_argument('-d', '--dataset', type=str, default="mr",
                        help='name of dataset (default: MUTAG)')
    parser.add_argument('-k', '--tree_deepth', type=int, default=2,
                        help='the deepth of coding tree (default: 2)')
    parser.add_argument('-b', '--batch_size', type=int, default=4,
                        help='input batch size for training (default: 32)')
    parser.add_argument('-e', '--epochs', type=int, default=500,
                        help='number of epochs to train (default: 500)')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--iters_per_epoch', type=int, default=50,
                        help='number of iterations per each epoch (default: 50)')
    parser.add_argument('-s', '--seed', type=int, default=1,
                        help='random seed for splitting the dataset into 10 (default: 0)')
    parser.add_argument('-id', '--input_dim', type=int, default=300,
                        help='the dim of the input (default: 300)')
    parser.add_argument('-lm', '--num_mlp_layers', type=int, default=2,
                        help='number of layers for MLP EXCLUDING the input one (default: 2). 1 means linear model.')
    parser.add_argument('-hd', '--hidden_dim', type=int, default=96,
                        help='number of hidden units (default: 96)')
    parser.add_argument('-fd', '--final_dropout', type=float, default=0.5,
                        help='final layer dropout (default: 0.5)')
    parser.add_argument('-l2', '--l2rate', type=float, default=0,
                        help='L2 penalty lambda')
    parser.add_argument('-tp', '--tree_pooling_type', type=str, default="sum", choices=["root", "sum", "average"],
                        help='Pooling for over nodes in a tree: root, sum or average')
    parser.add_argument('-np', '--neighbor_pooling_type', type=str, default="sum", choices=["sum", "average", "max"],
                        help='Pooling for over neighboring nodes: sum, average or max')
    parser.add_argument('-md', '--mode', type=str, default="dependency", choices=["window", "dependency", "key"],
                        help='Mode')
    parser.add_argument('-pe', '--position_embedding', type=str, default="onehot", choices=["add", "concat", "onehot"],
                        help='Mode')
    parser.add_argument('--filename', type=str, default="",
                        help='output file')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    args = parser.parse_args()

    begin_time = time.strftime("%m%d_%H%M", time.localtime())
    if not args.filename == "":
        filename = args.filename + args.dataset + '_' + str(args.tree_deepth) + '_' + begin_time + '.txt'
        with open(filename, 'a+') as f:
            f.write(str(vars(args)))
            f.write("\n")

    # set up seeds and gpu device
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    # get data
    train_trees, val_trees, test_trees = get_train_and_test(args.dataset, args.tree_deepth, args.input_dim, args.mode, args.position_embedding)
    num_classes = len(set([t['label'] for t in train_trees]))
    print('#data:%s\t#classes:%s' % (len(train_trees) + len(test_trees), num_classes))

    model = TL(args.tree_deepth,
                args.num_mlp_layers,
                train_trees[0]['node_features'].shape[1],
                args.hidden_dim,
                num_classes,
                args.final_dropout,
                args.tree_pooling_type,
                args.neighbor_pooling_type,
                device
                ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.l2rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    best_val = 0
    best_epoch = 0

    # train
    for epoch in range(1, args.epochs + 1):
        avg_loss = train(args, model, device, train_trees, optimizer, epoch)
        acc_train, acc_val, acc_test = test(args, model, device, train_trees, val_trees, test_trees, epoch)

        if acc_val >= best_val:
            best_val = max(acc_val, best_val)
            best_epoch = epoch
            best_test = acc_test

        scheduler.step()

        # output the train info to file
        if not args.filename == "":
            filename = args.filename + args.dataset + '_' + str(args.tree_deepth) + '_' + begin_time + '.txt'
            with open(filename, 'a+') as f:
                f.write("%f %.4f %.4f" % (avg_loss, acc_train, acc_test))
                f.write("\n")
        print("")

    print("best epoch: %d" % (best_epoch))
    print("test acc: %.4f" % (best_test))
    # output
    if not args.filename == "":
        filename = args.filename + args.dataset + '_' + str(args.tree_deepth) + '_' + begin_time + '.txt'
        with open(filename, 'a+') as f:
            f.write("best epoch: %d" % (best_epoch))
            f.write("\n")
            f.write("test acc: %.4f" % (best_test))
            f.write("\n")


if __name__ == '__main__':
    main()
