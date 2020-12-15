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

from utils import load_file, preprocessing, get_vocab, load_embeddings, create_gows, accuracy, generate_batches, AverageMeter,def_f1_score,def_precision,def_recall
from models import MPAD
SEED = 4
torch.manual_seed(SEED)


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
parser.add_argument('--directed', type=str2bool, default=True,
                    help='Create directed graph of words.')
parser.add_argument('--use-master-node', type=str2bool, default=True,
                    help='Include master node in graph of words.')#made changes here
parser.add_argument('--normalize', action='store_true', default=True,
                    help='Normalize adjacency matrices.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--batch-size', type=int, default=128,
                    help='Batch size.')
parser.add_argument('--patience', type=int, default=20,
                    help='Number of epochs to wait if no improvement during training.')
parser.add_argument('--name-of-dataset', default='Polarity',
                    help='Name of  dataset.')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# Read data
docs, class_labels = load_file(args.path_to_dataset)
docs = preprocessing(docs)

enc = LabelEncoder()
class_labels = enc.fit_transform(class_labels)

nclass = np.unique(class_labels).size
y = list()
for i in range(len(class_labels)):
    t = np.zeros(1)
    t[0] = class_labels[i]
    y.append(t)

vocab = get_vocab(docs)
embeddings = load_embeddings(args.path_to_embeddings, vocab)
adj, features, _ = create_gows(docs, vocab, args.window_size, args.directed, args.normalize, args.use_master_node)

kf = KFold(n_splits=10, shuffle=True)
it = 0
accs = list()
f1_scores = list()
precisions = list()
recalls = list()
for train_index, test_index in kf.split(y):
    it += 1

    idx = np.random.permutation(train_index)
    train_index = idx[:int(idx.size*0.9)].tolist()
    val_index = idx[int(idx.size*0.9):].tolist()

    n_train = len(train_index)
    n_val = len(val_index)
    n_test = len(test_index)

    adj_train = [adj[i] for i in train_index]
    features_train = [features[i] for i in train_index]
    y_train = [y[i] for i in train_index]

    adj_val = [adj[i] for i in val_index]
    features_val = [features[i] for i in val_index]
    y_val = [y[i] for i in val_index]

    adj_test = [adj[i] for i in test_index]
    features_test = [features[i] for i in test_index]
    y_test = [y[i] for i in test_index]

    adj_train, features_train, batch_n_graphs_train, y_train = generate_batches(adj_train, features_train, y_train, args.batch_size, args.use_master_node)
    adj_val, features_val, batch_n_graphs_val, y_val = generate_batches(adj_val, features_val, y_val, args.batch_size, args.use_master_node)
    adj_test, features_test, batch_n_graphs_test, y_test = generate_batches(adj_test, features_test, y_test, args.batch_size, args.use_master_node)

    n_train_batches = ceil(n_train/args.batch_size)
    n_val_batches = ceil(n_val/args.batch_size)
    n_test_batches = ceil(n_test/args.batch_size)

    # Model and optimizer
    model = MPAD(embeddings.shape[1], args.message_passing_layers, args.hidden, args.penultimate, nclass, args.dropout, embeddings, args.use_master_node)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters, lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    if args.cuda:
        model.cuda()
        adj_train = [x.cuda() for x in adj_train]
        features_train = [x.cuda() for x in features_train]
        batch_n_graphs_train = [x.cuda() for x in batch_n_graphs_train]
        y_train = [x.cuda() for x in y_train]
        adj_val = [x.cuda() for x in adj_val]
        features_val = [x.cuda() for x in features_val]
        batch_n_graphs_val = [x.cuda() for x in batch_n_graphs_val]
        y_val = [x.cuda() for x in y_val]
        adj_test = [x.cuda() for x in adj_test]
        features_test = [x.cuda() for x in features_test]
        batch_n_graphs_test = [x.cuda() for x in batch_n_graphs_test]
        y_test = [x.cuda() for x in y_test]

    def train(epoch, adj, features, batch_n_graphs, y):
        optimizer.zero_grad()
        output = model(features, adj, batch_n_graphs)
        loss_train = F.cross_entropy(output, y)
        loss_train.backward()
        optimizer.step()
        return output, loss_train

    def test(adj, features, batch_n_graphs, y):
        output = model(features, adj, batch_n_graphs)
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
            output, loss = train(epoch, adj_train[i], features_train[i], batch_n_graphs_train[i], y_train[i])
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
            output, loss = test(adj_val[i], features_val[i], batch_n_graphs_val[i], y_val[i])
            val_loss.update(loss.item(), output.size(0))
            val_acc.update(accuracy(output.data, y_val[i].data), output.size(0))
            val_f1.update(def_f1_score(output.data, y_val[i].data), output.size(0))
            val_precision.update(def_precision(output.data, y_val[i].data), output.size(0))
            val_recall.update(def_recall(output.data, y_val[i].data), output.size(0))
        # Print results
        print("Cross-val iter:", '%02d' % it,"epoch:", '%03d' % (epoch + 1), "train_loss=", "{:.5f}".format(train_loss.avg),
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





    # Testing
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
        output, loss = test(adj_test[i], features_test[i], batch_n_graphs_test[i], y_test[i])
        test_loss.update(loss.item(), output.size(0))
        test_acc.update(accuracy(output.data, y_test[i].data), output.size(0))
        test_f1.update(def_f1_score(output.data, y_test[i].data), output.size(0))
        test_precision.update(def_precision(output.data, y_test[i].data), output.size(0))
        test_recall.update(def_recall(output.data, y_test[i].data), output.size(0))

    accs.append(test_acc.avg.cpu().numpy())
    # f1_scores.append(test_f1.avg.cpu().numpy())
    # precisions.append(test_precision.avg.cpu().numpy())
    # recalls.append(test_recall.avg.cpu().numpy())
    f1_scores.append(test_f1.avg)
    precisions.append(test_precision.avg)
    recalls.append(test_recall.avg)

    # Print results
    print("test_loss=", "{:.5f}".format(test_loss.avg),  "test_acc=", "{:.5f}".format(test_acc.avg),
  "test_f1=", "{:.5f}".format(test_f1.avg),"test_precision=", "{:.5f}".format(test_precision.avg),"test_recall=", "{:.5f}".format(test_recall.avg))

print("avg_test_acc=", "{:.5f}".format(np.mean(accs)),
       "avg_test_f1=", "{:.5f}".format(np.mean(f1_scores)),
       "avg_test_prec=", "{:.5f}".format(np.mean(precisions)),
       "avg_test_recall=", "{:.5f}".format(np.mean(recalls))
        )
