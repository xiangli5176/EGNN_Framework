import numpy as np
from collections import defaultdict
import itertools
import sklearn.preprocessing as skp
from munkres import Munkres, print_matrix
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn import metrics
from sklearn.metrics import cluster, silhouette_score
from scipy.spatial.distance import pdist
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score


### ======================== classification metrics ==========================
def eva_node_classification(probs, labels):
    """
    Args:
        probs (numpy.array): probabitliy matrix from the log loss regression
        labels (numpy.array):  golden labels 

    Returns:
        [type]: [description]
    """
    preds = np.argmax(probs, 1)

    f1_micro, f1_macro = compute_f1_score(preds, labels)
    accuracy = compute_accuracy(preds, labels)

    res_metric = {"f1_micro" : f1_micro, "f1_macro" : f1_macro, "accuracy": accuracy}
    return res_metric

def compute_accuracy(preds, labels):

    correct = np.equal(preds, labels)
    correct = correct.sum()
    return correct / len(labels)

def compute_f1_score(preds, labels):
    """[summary]

    Args:
        preds (numpy.array): predictions
        labels (numpy.array): golden labels

    Returns:
        [type]: [description]
    """
    micro = f1_score(labels, preds, average='micro')
    macro = f1_score(labels, preds, average='macro')
    return micro, macro

### ======================== link prediction metrics ==========================
def get_roc_score(emb, adj_orig, edges_pos, edges_neg):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    adj_rec = np.dot(emb, emb.T)
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    res_metric = {"auc_score" : roc_score, "ap_score" : ap_score}

    return res_metric

class semantic_emb:
    """ Metrics for embedding from semantic structure """
    def __init__(self, ground_truth):
        self.pos_idx, self.neg_idx = self.get_semantic_idx(ground_truth)
    
    def get_pos_neg_distr(self, emb, neg_bin_idx=-1):
        """ Obtain the stats of emb from semantic structure
            Return: 
                1) mean of positive feature product;
                2) 1/10 fraction of largest feature inner product
                3) distribution of histogram for semantic negative samples
        """
        n_emb = skp.normalize(emb)
        z = np.matmul(n_emb, n_emb.transpose())

        num_bin = 10
        hist, bin_edges = np.histogram(z[self.neg_idx], bins = num_bin, range = (-1.0, 1.0), density=False)
        hist = hist / np.sum(hist)

        idx_total = sum(range(1, num_bin+1))
        hist_importance = sum(val * idx / idx_total for idx, val in enumerate(hist, 1))
        
        return np.mean(z[self.pos_idx]), hist_importance, [hist, bin_edges]

    def get_semantic_idx(self, golden_labels):
        """ Obtain the semantic positive and negative pairs indices
            Return : tuples of lists
        """
        n_nodes = golden_labels.shape[0]
        memo = defaultdict(list)
        for idx, label in enumerate(golden_labels):
            memo[label].append(idx)

        align = set()
        for key, val in memo.items():
            memo[key] = sorted(val)
            align |= set(itertools.combinations(memo[key], 2))

        triu_1 = np.triu_indices(n_nodes, 1)

        neg = ([], [])
        for r, c in zip(*triu_1):
            if (r, c) not in align:
                neg[0].append(r)
                neg[1].append(c)

        pos = list(align)
        pos = ([a for a, _ in pos], [b for _, b in pos])
        return pos, neg


def eval_golden_label(label, adjacency):
    """
        Evaluate the graph structural metrics from classification golden labels
    """
    golden_conductance = conductance(adjacency, label)
    golden_modularity = modularity(adjacency, label)

    return golden_conductance, golden_modularity


def emb_eva(y_true, y_pred, embedding, emb_stat_obj, neg_bin_idx=-1):
    res_metric = {}
    # res_metric["silhouette"] = silhouette_score(embedding, y_pred)
    res_metric["uniformity"] = uniformity(embedding, t = 2, distance = 'euclidean')
    
    semantic_pos_mean, semantic_neg_frac_rank, [hist, bin_edges] = emb_stat_obj.get_pos_neg_distr(embedding, neg_bin_idx = neg_bin_idx)
    res_metric["semantic_pos_mean"] = semantic_pos_mean
    res_metric["semantic_neg_frac_rank"] = semantic_neg_frac_rank
    res_metric["semantic_neg_frac_distr"] = [hist, bin_edges]

    return res_metric


def eva(y_true, y_pred, adjacency, embedding = None):
    """
        adjacency : 
    """
    acc, f1_macro, f1_micro = cluster_acc(y_true, y_pred)
    nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
    ari = ari_score(y_true, y_pred)

    res_metric = {}
    
    res_metric["Accuracy"] = acc
    res_metric["f1_micro"] = f1_micro
    res_metric["f1_macro"] = f1_macro
    res_metric["NMI"] = nmi
    res_metric["ARI"] = ari

    # precision = pairwise_precision(y_true, y_pred)
    # recall = pairwise_recall(y_true, y_pred)
    # res_metric["dmon_f1_score"] = 2 * precision * recall / (precision + recall)

    res_metric["conductance"] = conductance(adjacency, y_pred)
    res_metric["modularity"] = modularity(adjacency, y_pred)

    return res_metric
    

def cluster_acc(y_true, y_pred):
    """
        y_true : is the golden label from the dataset 
        y_pred: is the obtained prediction from clustering
    """
    y_true = y_true - np.min(y_true)  # make the lebel starts from : 0

    l1 = list(set(y_true))
    numclass1 = len(l1)

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    ind = 0
    if numclass1 != numclass2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i   # random assign missing predict labels into pred
                ind += 1

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    assert numclass1 == numclass2, "y_true and y_pred not the same size"

    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]  # keep all the element index of c1 of y_true
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)

    # match two clustering results by Munkres algorithm
    m = Munkres()
    cost = cost.__neg__().tolist()  # fiirst use the negative, we wish to minimize this using Hungarian algorithm
                                    # this is equal to maximize the clustering overall 
    indexes = m.compute(cost)

    # get the match results
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        # correponding label in l2:
        c2 = l2[indexes[i][1]]

        # ai is the index with label==c2 in the pred_label list
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c

    acc = metrics.accuracy_score(y_true, new_predict)
    f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
    precision_macro = metrics.precision_score(y_true, new_predict, average='macro')
    recall_macro = metrics.recall_score(y_true, new_predict, average='macro')
    f1_micro = metrics.f1_score(y_true, new_predict, average='micro')
    precision_micro = metrics.precision_score(y_true, new_predict, average='micro')
    recall_micro = metrics.recall_score(y_true, new_predict, average='micro')
    return acc, f1_macro, f1_micro

    
### ============== Below metrics are all from the DMoN framework


def pairwise_precision(y_true, y_pred):
    """Computes pairwise precision of two clusterings.

    Args:
        y_true: An [n] int ground-truth cluster vector.
        y_pred: An [n] int predicted cluster vector.

    Returns:
        Precision value computed from the true/false positives and negatives.
  """
    true_positives, false_positives, _, _ = _pairwise_confusion(y_true, y_pred)
    return true_positives / (true_positives + false_positives)


def pairwise_recall(y_true, y_pred):
    """Computes pairwise recall of two clusterings.

    Args:
    y_true: An (n,) int ground-truth cluster vector.
    y_pred: An (n,) int predicted cluster vector.

    Returns:
    Recall value computed from the true/false positives and negatives.
    """
    true_positives, _, false_negatives, _ = _pairwise_confusion(y_true, y_pred)
    return true_positives / (true_positives + false_negatives)


def pairwise_accuracy(y_true, y_pred):
    """Computes pairwise accuracy of two clusterings.

    Args:
    y_true: An (n,) int ground-truth cluster vector.
    y_pred: An (n,) int predicted cluster vector.

    Returns:
    Accuracy value computed from the true/false positives and negatives.
    """
    true_pos, false_pos, false_neg, true_neg = _pairwise_confusion(y_true, y_pred)
    return (true_pos + false_pos) / (true_pos + false_pos + false_neg + true_neg)


def _pairwise_confusion(y_true, y_pred):
    """Computes pairwise confusion matrix of two clusterings.
        For each true label (row idx), the one with the largest number of examples is the true cluster predicted, i.e. same_class_true
    Args:
        y_true: An (n,) int ground-truth cluster vector.
        y_pred: An (n,) int predicted cluster vector.

    Returns:
        True positive, false positive, true negative, and false negative values.
    """
    contingency = cluster.contingency_matrix(y_true, y_pred)
    same_class_true = np.max(contingency, 1)
    same_class_pred = np.max(contingency, 0)                     # same cluster, largest number of examples in one true label
    diff_class_true = contingency.sum(axis=1) - same_class_true   # same label but does not belong to the largest cluster
    diff_class_pred = contingency.sum(axis=0) - same_class_pred
    total = contingency.sum()

    true_positives = (same_class_true * (same_class_true - 1)).sum()
    false_positives = (diff_class_true * same_class_true * 2).sum()

    false_negatives = (diff_class_pred * same_class_pred * 2).sum()
    true_negatives = total * (
        total - 1) - true_positives - false_positives - false_negatives

    return true_positives, false_positives, false_negatives, true_negatives


def modularity(adjacency, clusters):
    """Computes graph modularity.

    Args:
    adjacency: Input graph in terms of its sparse adjacency matrix.
    clusters: An (n,) int cluster vector.

    Returns:
    The value of graph modularity.
    https://en.wikipedia.org/wiki/Modularity_(networks)
    """
    degrees = adjacency.sum(axis=0).A1
    n_edges = degrees.sum()  # Note that it's actually 2*n_edges.
    result = 0
    for cluster_id in np.unique(clusters):
        cluster_indices = np.where(clusters == cluster_id)[0]
        adj_submatrix = adjacency[cluster_indices, :][:, cluster_indices]
        degrees_submatrix = degrees[cluster_indices]
        result += np.sum(adj_submatrix) - (np.sum(degrees_submatrix)**2) / n_edges
    return result / n_edges


def conductance(adjacency, clusters):
    """Computes graph conductance as in Yang & Leskovec (2012).

    Args:
    adjacency: Input graph in terms of its sparse adjacency matrix.
    clusters: An (n,) int cluster vector.

    Returns:
    The average conductance value of the graph clusters.
    """
    inter = 0  # Number of inter-cluster edges.
    intra = 0  # Number of intra-cluster edges.
    cluster_indices = np.zeros(adjacency.shape[0], dtype=np.bool)
    for cluster_id in np.unique(clusters):
        cluster_indices[:] = 0
        cluster_indices[np.where(clusters == cluster_id)[0]] = 1
        adj_submatrix = adjacency[cluster_indices, :]
        inter += np.sum(adj_submatrix[:, cluster_indices])
        intra += np.sum(adj_submatrix[:, ~cluster_indices])
    return intra / (inter + intra)
    

def uniformity(x, t = 2, distance = 'euclidean'):
    """ 
        Defined or applided by: 
        distance = 'cosine'
        Input x (np.array): the target matrix
        Output (np.float32): Uniformity = -L_uniformity, a larger value means the features are more uniform
    """
    e = np.exp(np.power( pdist(x, distance) , 2 ) * (-t))
    return -np.log(np.mean(e))


def calc_f1(y_true, y_pred,is_sigmoid):
    if not is_sigmoid:
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
    else:
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0
    return metrics.f1_score(y_true, y_pred, average="micro"), metrics.f1_score(y_true, y_pred, average="macro")


