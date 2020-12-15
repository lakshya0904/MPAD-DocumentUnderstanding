import time
import argparse
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from math import ceil

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy
from hutils_test import load_file, preprocessing, get_vocab, load_embeddings, create_gows, accuracy, generate_batches, AverageMeter,def_f1_score,def_precision,def_recall

from models import MPAD

# SEED = 4
# torch.manual_seed(SEED)


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--path-to-dataset', default='datasets/subjectivity.txt',
                    help='Path to the dataset.')
parser.add_argument('--path-to-embeddings', default='../GoogleNews-vectors-negative300.bin',
                    help='Path to the to the word2vec binary file.')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--penultimate', type=int, default=64,
                    help='Size of penultimate layer.')
parser.add_argument('--message-passing-layers', type=int, default=2,
                    help='Number of message passing layers.')
parser.add_argument('--window-size', type=int, default=2,
                    help='Size of window.')
parser.add_argument('--directed', action='store_true', default=True,
                    help='Create directed graph of words.')
parser.add_argument('--use-master-node', action='store_true', default=True,
                    help='Include master node in graph of words.')
parser.add_argument('--normalize', action='store_true', default=True,
                    help='Normalize adjacency matrices.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--batch-size', type=int, default=128,
                    help='Batch size.')
parser.add_argument('--graph-of-sentences', default='sentence_att',
                    help='Graph of sentences (clique, path or sentence_att).')
parser.add_argument('--patience', type=int, default=20,
                    help='Number of epochs to wait if no improvement during training.')

parser.add_argument('--path-to-test-dataset', default='datasets/subjectivity.txt',
                    help='Path to the dataset.')

parser.add_argument('--take-test', action='store_true', default=False,
                    help='test on test data or CV')


parser.add_argument('--name-of-dataset', default='Polarity',
                    help='Name of  dataset.')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
if(args.name_of_dataset!='SST1 basic' or args.name_of_dataset!='SST1 hierarchial atten'):
  SEED = 4
  torch.manual_seed(SEED)
  print("Setting seed 4 \n\n")
if(args.use_master_node):
  print("**USING MASTER NODE ! **")
else:
  print("**NOT USING MASTER NODE ! **")

if(args.directed):
  print("**USING DIRECTED EDGES ! **")
else:
  print("**USING UNDIRECTED EDGES ! **")

if(args.take_test==True):#check on test data

    #get the test data size for dividng features and labels later
    test_docs, test_class_labels = load_file(args.path_to_test_dataset)
    n_test_samples = len(test_class_labels)
    print("Num test samples : ",n_test_samples)


    train_docs, class_labels = load_file(args.path_to_dataset)

    train_class_labels = class_labels[:-n_test_samples]
    test_class_labels = class_labels[-n_test_samples:]

    train_docs = preprocessing(train_docs)
    enc = LabelEncoder()

    # fine ?
    encoder = LabelEncoder()
    encoder.fit(train_class_labels)
    encoder.fit(test_class_labels)
    numpy.save('classes.npy', encoder.classes_)

    train_class_labels = enc.fit_transform(train_class_labels)
    test_class_labels = enc.fit_transform(test_class_labels)
    nclass = np.unique(train_class_labels).size
    train_y = list()
    for i in range(len(train_class_labels)):
      t = np.zeros(1)
      t[0] = train_class_labels[i]
      train_y.append(t)


    test_y = list()
    for i in range(len(test_class_labels)):
        t = np.zeros(1)
        t[0] = test_class_labels[i]
        test_y.append(t)


    train_vocab = get_vocab(train_docs)

    embeddings = load_embeddings(args.path_to_embeddings, train_vocab)
    adj, features, subgraphs,num_train_sents = create_gows(len(train_y),train_docs, train_vocab, args.window_size, args.directed, args.normalize, args.use_master_node)
    print(len(adj),len(features),len(subgraphs),num_train_sents)
    # train_adj=adj[:-n_test_samples]
    # train_features = features[:-n_test_samples]
    # train_subgraphs = subgraphs[:-n_test_samples]
    # test_adj = adj[-n_test_samples:]
    # test_features = features[-n_test_samples:]
    # test_subgraphs = subgraphs[-n_test_samples:]

    train_adj=adj[:num_train_sents]
    train_features = features[:num_train_sents]
    train_subgraphs = subgraphs[:-n_test_samples]
    test_adj = adj[num_train_sents:]
    test_features = features[num_train_sents:]
    test_subgraphs = subgraphs[-n_test_samples:]


    print("Train & Test data sizes ",len(train_adj),len(test_adj))

    train_index = np.arange(0,len(train_y),1)
    idx = np.random.permutation(train_index)
    train_index = idx[:int(idx.size*0.9)].tolist()  # 90% of train_index to training & 10% to validation
    val_index = idx[int(idx.size*0.9):].tolist()

    n_train = len(train_index)
    n_val = len(val_index)

    adj_train = list()
    features_train = list()
    subgraphs_train = list()
    subgraphs_tmp = [train_subgraphs[i] for i in train_index]
    """
    for i in train_index:
        subgraphs_tmp.append(train_subgraphs[i])
    """
    c = 0
    for s in subgraphs_tmp:
        l = list()
        for i in s:
            adj_train.append(train_adj[i])
            features_train.append(train_features[i])
            l.append(c) # PROBLEM
            c += 1
        subgraphs_train.append(l)

    y_train = [train_y[i] for i in train_index]

    adj_val = list()
    features_val = list()
    subgraphs_val = list()
    subgraphs_tmp = [train_subgraphs[i] for i in val_index]
    c = 0
    for s in subgraphs_tmp:
        l = list()
        for i in s:
            adj_val.append(train_adj[i])
            features_val.append(train_features[i])
            l.append(c)
            c += 1
        subgraphs_val.append(l)
    y_val = [train_y[i] for i in val_index]

    adj_test = list()
    features_test = list()
    subgraphs_test = list()
    test_index = np.arange(len(test_y))
    subgraphs_tmp = [test_subgraphs[i] for i in test_index]
    c = 0
    for s in subgraphs_tmp:
        l = list()
        # print('\n')
        for i in s:
            # print(i,'\t',i-num_train_sents)
            adj_test.append(test_adj[i-num_train_sents])
            features_test.append(test_features[i-num_train_sents])
            l.append(c)
            c += 1
        subgraphs_test.append(l)
    y_test = [test_y[i] for i in test_index]

    adj_train, adj_s_train, features_train, shapes_train, y_train = generate_batches(adj_train, features_train, subgraphs_train, y_train, args.batch_size, args.use_master_node, args.graph_of_sentences)
    adj_val, adj_s_val, features_val, shapes_val, y_val = generate_batches(adj_val, features_val, subgraphs_val, y_val, args.batch_size, args.use_master_node, args.graph_of_sentences)
    test_batch_size = len(y_test)
    adj_test, adj_s_test, features_test, shapes_test, y_test = generate_batches(adj_test, features_test, subgraphs_test, y_test, test_batch_size, args.use_master_node, args.graph_of_sentences)

    n_train_batches = ceil(n_train/args.batch_size)
    n_val_batches = ceil(n_val/args.batch_size)
    n_test_batches = ceil(n_test_samples/test_batch_size)

    # Model and optimizer
    model = MPAD(embeddings.shape[1], args.message_passing_layers, args.hidden, args.penultimate, nclass, args.dropout, embeddings, args.use_master_node, args.graph_of_sentences)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters, lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    if args.cuda:
        model.cuda()
        features_train = [x.cuda() for x in features_train]
        adj_train = [x.cuda() for x in adj_train]
        adj_s_train = [x.cuda() for x in adj_s_train]
        shapes_train = [x.cuda() for x in shapes_train]
        y_train = [x.cuda() for x in y_train]
        features_val = [x.cuda() for x in features_val]
        adj_val = [x.cuda() for x in adj_val]
        adj_s_val = [x.cuda() for x in adj_s_val]
        shapes_val = [x.cuda() for x in shapes_val]
        y_val = [x.cuda() for x in y_val]
        features_test = [x.cuda() for x in features_test]
        adj_test = [x.cuda() for x in adj_test]
        adj_s_test = [x.cuda() for x in adj_s_test]
        shapes_test = [x.cuda() for x in shapes_test]
        y_test = [x.cuda() for x in y_test]

    def train(epoch, adj, adj_s, features, shapes, y):
        optimizer.zero_grad()
        output = model(features, adj, adj_s, shapes)
        loss_train = F.cross_entropy(output, y)
        loss_train.backward()
        optimizer.step()
        return output, loss_train

    def test(adj, adj_s, features, shapes, y):
        output = model(features, adj, adj_s, shapes)
        loss_test = F.cross_entropy(output, y)
        return output, loss_test

    best_acc = 0
    list_of_train_losses = list()
    list_of_train_acc = list()
    list_of_train_f1 = list()
    epochs_completed=0
    for epoch in range(args.epochs):
        # scheduler.step()
        epochs_completed+=1
        start = time.time()
        model.train()
        train_loss = AverageMeter()
        train_acc = AverageMeter()
        train_f1 = AverageMeter()
        train_recall = AverageMeter()
        train_precision = AverageMeter()
        # Train for one epoch
        for i in range(n_train_batches):
            if args.graph_of_sentences == 'sentence_att':
                output, loss = train(epoch, adj_train[i], None, features_train[i], shapes_train[i], y_train[i])
            else:
                output, loss = train(epoch, adj_train[i], adj_s_train[i], features_train[i], shapes_train[i], y_train[i])

            train_loss.update(loss.item(), output.size(0))
            train_acc.update(accuracy(output.data, y_train[i].data), output.size(0))
            train_f1.update(def_f1_score(output.data, y_train[i].data), output.size(0))
            # list_of_train_f1.append(loss.item())
            train_precision.update(def_precision(output.data, y_train[i].data), output.size(0))
            train_recall.update(def_recall(output.data, y_train[i].data), output.size(0))
        # Evaluate on validation set
        scheduler.step()
        list_of_train_losses.append(train_loss.avg)
        list_of_train_acc.append(train_acc.avg)
        list_of_train_f1.append(train_f1.avg)

        model.eval()
        val_loss = AverageMeter()
        val_acc = AverageMeter()
        val_f1 = AverageMeter()
        val_recall = AverageMeter()
        val_precision = AverageMeter()

        for i in range(n_val_batches):
            if args.graph_of_sentences == 'sentence_att':
                output, loss = test(adj_val[i], None, features_val[i], shapes_val[i], y_val[i])
            else:
                output, loss = test(adj_val[i], adj_s_val[i], features_val[i], shapes_val[i], y_val[i])

            val_loss.update(loss.item(), output.size(0))
            val_acc.update(accuracy(output.data, y_val[i].data), output.size(0))
            val_f1.update(def_f1_score(output.data, y_val[i].data), output.size(0))
            val_precision.update(def_precision(output.data, y_val[i].data), output.size(0))
            val_recall.update(def_recall(output.data, y_val[i].data), output.size(0))


        # Print results
        print("epoch:", '%03d' % (epoch + 1), "train_loss=", "{:.5f}".format(train_loss.avg),
          "train_acc=", "{:.5f}".format(train_acc.avg), "train_f1=", "{:.5f}".format(train_f1.avg),
           "train_precision=", "{:.5f}".format(train_precision.avg),"train_recall=", "{:.5f}".format(train_recall.avg),'\n\t  ',
           "val_loss=", "{:.5f}".format(val_loss.avg),"val_acc=", "{:.5f}".format(val_acc.avg),
           "val_f1=", "{:.5f}".format(val_f1.avg),
           "val_precision=", "{:.5f}".format(val_precision.avg),"val_recall=", "{:.5f}".format(val_recall.avg),
            "time=", "{:.5f}".format(time.time() - start))

        # Remember best accuracy and save checkpoint
        is_best = val_acc.avg >= best_acc
        best_acc = max(val_acc.avg, best_acc)
        if is_best:
            early_stopping_counter = 0
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, 'model_best.pth.tar')
        else:
            early_stopping_counter += 1
            print("EarlyStopping: %i / %i" % (early_stopping_counter, args.patience))
            if early_stopping_counter == args.patience:
                print("EarlyStopping: Stop training")
                break

    print("Optimization finished!")

    #PLOTTING THE RESULTS

    #PLOTTING
    # if()
    import matplotlib.pyplot as plt
    # PLOT LOSS
    plt.figure(1)
    epochs_enumerated = [i for i in range(1,epochs_completed+1)]
    list_of_train_losses  = np.array(list_of_train_losses).reshape((len(epochs_enumerated),1))
    epochs_enumerated = np.array(epochs_enumerated).reshape((list_of_train_losses.shape[0],1))
    plt.plot(epochs_enumerated,list_of_train_losses)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title(args.name_of_dataset+' training loss')
    # plt.show()
    plt.savefig(args.name_of_dataset+'_loss.png')

    # PLOT ACC
    # import matplotlib.pyplot as plt
    plt.figure(2)
    epochs_enumerated = [i for i in range(1,epochs_completed+1)]
    list_of_train_acc  = np.array(list_of_train_acc).reshape((len(epochs_enumerated),1))
    epochs_enumerated = np.array(epochs_enumerated).reshape((list_of_train_acc.shape[0],1))
    plt.plot(epochs_enumerated,list_of_train_acc)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title(args.name_of_dataset+' training acc')
    # plt.show()
    plt.savefig(args.name_of_dataset+'_acc.png')

    # PLOT F1
    # import matplotlib.pyplot as plt
    plt.figure(3)
    epochs_enumerated = [i for i in range(1,epochs_completed+1)]
    list_of_train_f1  = np.array(list_of_train_f1).reshape((len(epochs_enumerated),1))
    epochs_enumerated = np.array(epochs_enumerated).reshape((list_of_train_f1.shape[0],1))
    plt.plot(epochs_enumerated,list_of_train_f1)
    plt.xlabel('epoch')
    plt.ylabel('f1-score')
    plt.title(args.name_of_dataset+' training f1-score')
    # plt.show()
    plt.savefig(args.name_of_dataset+'_f1 score.png')


    test_loss = AverageMeter()
    test_acc = AverageMeter()
    test_f1 = AverageMeter()
    test_recall = AverageMeter()
    test_precision = AverageMeter()

    print("Loading checkpoint!")
    checkpoint = torch.load('model_best.pth.tar')
    epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    for i in range(n_test_batches):
        if args.graph_of_sentences == 'sentence_att':
            output, loss = test(adj_test[i], None, features_test[i], shapes_test[i], y_test[i])
        else:
            output, loss = test(adj_test[i], adj_s_test[i], features_test[i], shapes_test[i], y_test[i])

        test_loss.update(loss.item(), output.size(0))
        test_acc.update(accuracy(output.data, y_test[i].data), output.size(0))
        test_f1.update(def_f1_score(output.data, y_test[i].data), output.size(0))
        test_precision.update(def_precision(output.data, y_test[i].data), output.size(0))
        test_recall.update(def_recall(output.data, y_test[i].data), output.size(0))

    # accs.append(test_acc.avg.cpu().numpy())

    # Print results
    print("test_loss=", "{:.5f}".format(test_loss.avg), "test_acc=", "{:.5f}".format(test_acc.val), "test_acc=", "{:.5f}".format(test_acc.avg),
  "test_f1=", "{:.5f}".format(test_f1.avg),"test_precision=", "{:.5f}".format(test_precision.avg),"test_recall=", "{:.5f}".format(test_recall.avg))
