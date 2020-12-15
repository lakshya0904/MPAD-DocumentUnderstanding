import time
import argparse
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from math import ceil
from sklearn.metrics import classification_report,f1_score
# print("Imported f1-score")
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy
from utils import load_file, preprocessing, get_vocab, load_embeddings, create_gows, accuracy, generate_batches, AverageMeter,def_f1_score,def_precision,def_recall
from models import MPAD
# SEED = 4
# torch.manual_seed(SEED)
# Training settings

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


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
parser.add_argument('--use-master-node',  default=True,type=str2bool,
                    help='Include master node in graph of words.')
parser.add_argument('--normalize', action='store_true', default=True,
                    help='Normalize adjacency matrices.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--batch-size', type=int, default=128,
                    help='Batch size.')
parser.add_argument('--patience', type=int, default=20,
                    help='Number of epochs to wait if no improvement during training.')
#ADDED CODE
parser.add_argument('--path-to-test-dataset', default='datasets/subjectivity.txt',
                    help='Path to the dataset.')

parser.add_argument('--take-test', action='store_true', default=False,
                    help='test on test data or CV')

parser.add_argument('--name-of-dataset', default='Polarity',
                    help='Name of  dataset.')
#The store_true option automatically creates a default value of False.

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if(args.name_of_dataset!='SST1 basic'):
  SEED = 4
  torch.manual_seed(SEED)

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
  numpy.save('classes.npy', encoder.classes_)

  train_class_labels = enc.fit_transform(train_class_labels)

  nclass = np.unique(train_class_labels).size
  train_y = list()
  for i in range(len(train_class_labels)):
      t = np.zeros(1)
      t[0] = train_class_labels[i]
      train_y.append(t)

  train_vocab = get_vocab(train_docs)

  embeddings = load_embeddings(args.path_to_embeddings, train_vocab)
  adj, features, _ = create_gows(train_docs, train_vocab, args.window_size, args.directed, args.normalize, args.use_master_node)

  train_adj=adj[:-n_test_samples]
  train_features = features[:-n_test_samples]

  test_adj = adj[-n_test_samples:]
  test_features = features[-n_test_samples:]

  print("Train & Test data sizes ",len(train_adj),len(test_adj))
  #NEXT STEP

  train_index = np.arange(0,len(train_y),1)
  idx = np.random.permutation(train_index)
  train_index = idx[:int(idx.size*0.9)].tolist()  # 90% of train_index to training & 10% to validation
  val_index = idx[int(idx.size*0.9):].tolist()

  n_train = len(train_index)
  n_val = len(val_index)
  # n_test = len(test_index)

  # getting matrices for a set of sentences
  adj_train = [train_adj[i] for i in train_index]
  features_train = [train_features[i] for i in train_index]
  y_train = [train_y[i] for i in train_index]

  adj_val = [train_adj[i] for i in val_index]
  features_val = [train_features[i] for i in val_index]
  y_val = [train_y[i] for i in val_index]

  # adj_test = [adj[i] for i in test_index]
  # features_test = [features[i] for i in test_index]
  # y_test = [y[i] for i in test_index]

  adj_train, features_train, batch_n_graphs_train, y_train = generate_batches(adj_train, features_train, y_train, args.batch_size, args.use_master_node)
  adj_val, features_val, batch_n_graphs_val, y_val = generate_batches(adj_val, features_val, y_val, args.batch_size, args.use_master_node)
  # adj_test, features_test, batch_n_graphs_test, y_test = generate_batches(adj_test, features_test, y_test, args.batch_size, args.use_master_node)

  n_train_batches = ceil(n_train/args.batch_size)
  n_val_batches = ceil(n_val/args.batch_size)
  # n_test_batches = ceil(n_test/args.batch_size)

  model = MPAD(embeddings.shape[1], args.message_passing_layers, args.hidden, args.penultimate, nclass, args.dropout, embeddings, args.use_master_node)

  # fitering those are needed for back-prop
  parameters = filter(lambda p: p.requires_grad, model.parameters())
  optimizer = optim.Adam(parameters, lr=args.lr)
  scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5) # LR-schedule inplace of a constant LR

  if args.cuda: #just converting to cuda
      model.cuda()

      adj_train = [x.cuda() for x in adj_train]
      features_train = [x.cuda() for x in features_train]
      batch_n_graphs_train = [x.cuda() for x in batch_n_graphs_train]
      y_train = [x.cuda() for x in y_train]


      adj_val = [x.cuda() for x in adj_val]
      features_val = [x.cuda() for x in features_val]
      batch_n_graphs_val = [x.cuda() for x in batch_n_graphs_val]
      y_val = [x.cuda() for x in y_val]
      # adj_test = [x.cuda() for x in adj_test]
      # features_test = [x.cuda() for x in features_test]
      # batch_n_graphs_test = [x.cuda() for x in batch_n_graphs_test]
      # y_test = [x.cuda() for x in y_test]

  def train(epoch, adj, features, batch_n_graphs, y):#normal train
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
  for epoch in range(args.epochs):# THE REAL TRAINING LOOP BEGINS HERE
      # scheduler.step()
      epochs_completed+=1
      start = time.time()
      model.train()



      train_loss = AverageMeter()
      train_acc = AverageMeter()
      train_f1 = AverageMeter()
      train_recall = AverageMeter()
      train_precision = AverageMeter()
      # Train for EACH  epoch
      # loss_all_batches = 0
      # acc_all_
      for i in range(n_train_batches):
          output, loss = train(epoch, adj_train[i], features_train[i], batch_n_graphs_train[i], y_train[i])
          train_loss.update(loss.item(), output.size(0))
          # list_of_train_losses.append(loss.item())
          train_acc.update(accuracy(output.data, y_train[i].data), output.size(0))
          # list_of_train_acc.append(loss.item())
          train_f1.update(def_f1_score(output.data, y_train[i].data), output.size(0))
          # list_of_train_f1.append(loss.item())
          train_precision.update(def_precision(output.data, y_train[i].data), output.size(0))
          train_recall.update(def_recall(output.data, y_train[i].data), output.size(0))
      scheduler.step()
      list_of_train_losses.append(train_loss.avg)
      list_of_train_acc.append(train_acc.avg)
      list_of_train_f1.append(train_f1.avg)
      # Evaluate on validation set
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



  # TESTING

  # TEST REPEAT
  model.eval()  # it was there previusoly
  # test_docs, test_class_labels = load_file(args.path_to_test_dataset)
  # test_docs = preprocessing(test_docs)
  # # Will fit transform give same label to same class every tiem ?
  # # enc = LabelEncoder()
  #
  # encoder = LabelEncoder()
  # encoder.classes_ = numpy.load('classes.npy')
  #
  # # test_class_labels = enc.fit_transform(test_class_labels)
  # test_class_labels = encoder.fit_transform(test_class_labels)
  # # print(test_class_labels)
  # ntestclass = np.unique(test_class_labels).size
  # #
  # # assert(nclass==ntestclass)

  test_y = list()
  for i in range(len(test_class_labels)):
      t = np.zeros(1)
      t[0] = test_class_labels[i]
      test_y.append(t)

  # print(train_y,test_y)
  # test_y = train_y

  # test_vocab = get_vocab(test_docs)
  # embeddings = load_embeddings(args.path_to_embeddings, test_vocab)   # PROBLEM ?
  # test_adj, test_features, _ = create_gows(test_docs, test_vocab, args.window_size, args.directed, args.normalize, args.use_master_node)

  test_index = np.arange(len(test_y))
  n_test = len(test_index)

  adj_test = [test_adj[i] for i in test_index]
  features_test = [test_features[i] for i in test_index]
  y_test = [test_y[i] for i in test_index]

  test_batch_size = len(y_test)
  adj_test, features_test, batch_n_graphs_test, y_test = generate_batches(adj_test, features_test, y_test, test_batch_size, args.use_master_node)

  n_test_batches = ceil(n_test/test_batch_size)

  if args.cuda: #just converting to cuda
    adj_test = [x.cuda() for x in adj_test]
    features_test = [x.cuda() for x in features_test]
    batch_n_graphs_test = [x.cuda() for x in batch_n_graphs_test]
    y_test = [x.cuda() for x in y_test]


  test_loss = AverageMeter()
  test_acc = AverageMeter()
  test_f1 = AverageMeter()
  test_recall = AverageMeter()
  test_precision = AverageMeter()

  print("Loading checkpoint for Testing !")
  checkpoint = torch.load('model_best.pth.tar')
  epoch = checkpoint['epoch']
  model.load_state_dict(checkpoint['state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer'])

  # print("num test bathces ",n_test_batches)

  for i in range(n_test_batches):
      output, loss = test(adj_test[i], features_test[i], batch_n_graphs_test[i], y_test[i])
      test_loss.update(loss.item(), output.size(0))
      test_acc.update(accuracy(output.data, y_test[i].data), output.size(0))
      test_f1.update(def_f1_score(output.data, y_test[i].data), output.size(0))
      test_precision.update(def_precision(output.data, y_test[i].data), output.size(0))
      test_recall.update(def_recall(output.data, y_test[i].data), output.size(0))

  # accs.append(test_acc.avg.cpu().numpy())

  # Print results
  print("test_loss=", "{:.5f}".format(test_loss.avg), "test_acc=", "{:.5f}".format(test_acc.val), "test_acc=", "{:.5f}".format(test_acc.avg),
  "test_f1=", "{:.5f}".format(test_f1.avg),"test_precision=", "{:.5f}".format(test_precision.avg),"test_recall=", "{:.5f}".format(test_recall.avg))
  # print()
  # print("avg_test_acc=", "{:.5f}".format(np.mean(accs)))
