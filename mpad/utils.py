import numpy as np
import scipy.sparse as sp
import re
from math import ceil
from scipy.sparse import csr_matrix, lil_matrix
import torch
from gensim.models.keyedvectors import KeyedVectors
import sklearn
from sklearn.metrics import f1_score, precision_score, recall_score
# take filename & get text and labels
def load_file(filename):
    labels = []
    docs =[]

    with open(filename, encoding='utf8', errors='ignore') as f:
        for line in f:
            content = line.split('\t')
            labels.append(content[0])
            docs.append(content[1][:-1])
    
    return docs,labels  

  
def load_embeddings(fname, vocab):
    word_vecs = np.zeros((len(vocab)+1, 300))
    unknown_words = set()
    model = KeyedVectors.load_word2vec_format(fname, binary=True)
    for word in vocab:
        if word in model:
            word_vecs[vocab[word],:] = model[word]
        else:
            unknown_words.add(word)
            word_vecs[vocab[word],:] = np.random.uniform(-0.25, 0.25, 300)
            #setting random embedding for out of the dictionary words, we can have it zeros also
    print("Existing vectors:", len(vocab)-len(unknown_words))
    return word_vecs


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower().split()
    # returns a list of words


def preprocessing(docs): 
    preprocessed_docs = []
    n_sentences = 0

    for doc in docs:
        preprocessed_docs.append(clean_str(doc))
    
    return preprocessed_docs
    
    
def get_vocab(docs):
    vocab = dict()
    
    for doc in docs:
        for word in doc:
            if word not in vocab:
                vocab[word] = len(vocab)+1

    print("Vocabulary size: ", len(vocab))
    print("Please neglect the below warning, if any.")
        
    return vocab


def create_gows(docs, vocab, window_size, directed, to_normalize, use_master_node):
    adj = list()
    features = list()
    idx2term = list()
    
    for doc in docs:#for each example 
        edges = dict()  #from a pair of words to an integer 
        
        idx = dict()  #which has the unoque words->index dictionary fro a particular doc
        l_terms = list()# ??
        for i in range(len(doc)):#for each word( preprocessing returns list of words )
            if doc[i] not in idx:
                l_terms.append(doc[i])  #adding unique words in the sentence 
                idx[doc[i]] = len(idx)  #again, assignong a unique label for each word 
        idx2term.append(l_terms)#appendung unique words (list of list)
        if use_master_node:
            idx["master_node"] = len(idx) #if master node is present, add it to ir
        X = np.zeros(len(idx), dtype=np.int32)# X is list will contain the REAL INDICES INTO WORD-EMBEDDINGS (for each unique word int he sent3ece)
        for w in idx:
            if w != "master_node":
                X[idx[w]] = vocab[w]
            else:
                X[idx[w]] = len(vocab)

        for i in range(len(doc)):
            for j in range(i+1, i+window_size): #if window_size==2, look at only i+1th
                if j < len(doc):
                    if (doc[i], doc[j]) in edges:
                        edges[(doc[i], doc[j])] += 1.0/(j-i)  #giving more weight to the repeating neighbpurs 
                        if not directed:
                            edges[(doc[j], doc[i])] += 1.0/(j-i)
                    else:
                        edges[(doc[i], doc[j])] = 1.0/(j-i) #with window size as 2 , always 1. The weight is inverse of distance b/w 2 words
                        if not directed:
                            edges[(doc[j], doc[i])] = 1.0/(j-i)
            if use_master_node:
                edges[(doc[i],"master_node")] = 1.0 #master node connectedt o all 
                edges[("master_node",doc[i])] = 1.0

        edge_s = list() # starting edges
        edge_t = list() # terminal edges
        val = list()  #weights  
        for edge in edges:
            edge_s.append(idx[edge[0]]) #assigning unique indices
            edge_t.append(idx[edge[1]])
            val.append(edges[edge])
        #  Sparse matrix is the one which has most of the elements as zeros as opposed to dense which has most of the elements as non-zeros.
        # converting to adjacency matrix 
        # an adjacency matrix for each doc
        A = sp.csr_matrix((val,(edge_s, edge_t)), shape=(len(idx), len(idx)))
        if len(edges) == 0:# 0 words / 1 word
            A = sp.csr_matrix(([0],([0], [0])), shape=(1, 1))
            X = np.zeros(1, dtype=np.int32)

        if directed:#WHY ?
            A = A.transpose()
        if to_normalize and A.size > 1:
            A = normalize(A)
        adj.append(A)
        features.append(X)
        # NOW APPENDNG TO THE FEATURES

    return adj, features, idx2term  # adj is list of matrices, featires is list of lists(bcoz X is list) and it wil contain the indexes of unique words in embeddings
    #   idx2term is a list of lists and each list has the unique wors in it 

def normalize(mx):
    import warnings
    warnings.filterwarnings('ignore')
    
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def def_f1_score(output, labels):
    preds = output.max(1)[1].type_as(labels)
    # correct = preds.eq(labels).double()
    ytrue = labels.cpu().numpy()
    ypred = preds.cpu().numpy()
    return f1_score(ytrue,ypred,average='weighted',zero_division=1)
    # correct = correct.sum()
    # return correct / len(labels)

def def_precision(output, labels):
    preds = output.max(1)[1].type_as(labels)
    # correct = preds.eq(labels).double()
    ytrue = labels.cpu().numpy()
    ypred = preds.cpu().numpy()
    return precision_score(ytrue,ypred,average='weighted',zero_division=1)

def def_recall(output, labels):
    preds = output.max(1)[1].type_as(labels)
    # correct = preds.eq(labels).double()
    ytrue = labels.cpu().numpy()
    ypred = preds.cpu().numpy()
    return recall_score(ytrue,ypred,average='weighted',zero_division=1)
  

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row,
                                          sparse_mx.col))).long()
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

# called with args - (adj_train, features_train, y_train, args.batch_size, args.use_master_node)
def generate_batches(adj, features, y, batch_size, use_master_node, shuffle=False):
    n = len(y)
    if shuffle:
        index = np.random.permutation(n)
    else:
        index = np.array(range(n), dtype=np.int32)

    n_batches = ceil(n/batch_size)  # number of batches 

    adj_l = list()
    features_l = list()
    batch_n_graphs_l = list()
    y_l = list()

    for i in range(0, n, batch_size):
        if n > i + batch_size:  #
            up = i + batch_size
        else: #edge case 
            up = n

        n_graphs = 0
        max_n_nodes = 0

        for j in range(i, up):
            n_graphs += 1
            if adj[index[j]].shape[0] > max_n_nodes:
                max_n_nodes = adj[index[j]].shape[0]  #max number fo words in a sentence 

        n_nodes = n_graphs*max_n_nodes

        # it's like np.zeros 2 dimensional # lil_matrix converted to CSR later 
        adj_batch = lil_matrix((n_nodes, n_nodes))  # seems saem as csr matrx, just implementtional differences  lil_matrix is certainly sparse, it just stores information differently from csr_matrix. A normal 5 by 5 dense matrix containing np.float would take 200 bytes
        features_batch = np.zeros(n_nodes)
        y_batch = np.zeros(n_graphs)

        for j in range(i, up):  # i is lower bound & up is the upper bound of a particualr batch
            idx = (j-i)*max_n_nodes
            if max_n_nodes >= adj[index[j]].shape[0]:# HWN DOES IT HOLD TRUE/FALSE ? # using the lowest of them as index
                if use_master_node:
                    # both lhs & lhs are 2d matrices 
                    # can be done in single statement also. and -1 exclude the last row/column
                    adj_batch[idx:idx+adj[index[j]].shape[0]-1, idx:idx+adj[index[j]].shape[0]-1] = adj[index[j]][:-1,:-1]#assigning a 2d-matrix(its like assigning a sentence)
                    adj_batch[idx:idx+adj[index[j]].shape[0]-1, idx+max_n_nodes-1] = adj[index[j]][:-1,-1]#master node assignment
                    adj_batch[idx+max_n_nodes-1, idx:idx+adj[index[j]].shape[0]-1] = adj[index[j]][-1,:-1]  # master node is in both row & c=olumn
                else:
                    adj_batch[idx:idx+adj[index[j]].shape[0], idx:idx+adj[index[j]].shape[0]] = adj[index[j]]
                    
                features_batch[idx:idx+adj[index[j]].shape[0]-1] = features[index[j]][:-1]# assigning excepth the last value of X Why not taking last even when master node is not used ? 
            else:
                if use_master_node:
                    adj_batch[idx:idx+max_n_nodes-1, idx:idx+max_n_nodes-1] = adj[index[j]][:max_n_nodes-1,:max_n_nodes-1]
                    adj_batch[idx:idx+max_n_nodes-1, idx+max_n_nodes-1] = adj[index[j]][:max_n_nodes-1,-1]
                    adj_batch[idx+max_n_nodes-1, idx:idx+max_n_nodes-1] = adj[index[j]][-1,:max_n_nodes-1]
                else:
                    adj_batch[idx:idx+max_n_nodes, idx:idx+max_n_nodes] = adj[index[j]][:max_n_nodes,:max_n_nodes]
                
                features_batch[idx:idx+max_n_nodes-1] = features[index[j]][:max_n_nodes-1]
            # for each batch, you need adj_batch, y_batch & features_batch
            y_batch[j-i] = y[index[j]]

        adj_batch = adj_batch.tocsr()
        # adj_l contain many adj_batches , simiilarlily others 
        adj_l.append(sparse_mx_to_torch_sparse_tensor(adj_batch))
        features_l.append(torch.LongTensor(features_batch))
        batch_n_graphs_l.append(torch.LongTensor(np.array([n_graphs], dtype=np.int64)))
        y_l.append(torch.LongTensor(y_batch))

    return adj_l, features_l, batch_n_graphs_l, y_l


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count