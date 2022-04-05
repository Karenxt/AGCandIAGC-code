import scipy.io as sio
import time
import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans, DBSCAN
from metrics import clustering_metrics
from sklearn.metrics.pairwise import euclidean_distances
from sklearn_extra.cluster import KMedoids
import sklearn
from kneed import DataGenerator, KneeLocator
import torch
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize

import warnings
warnings.filterwarnings("ignore")
def normalize_adj(adj, type='sym'):
    """Symmetrically normalize adjacency matrix."""
    if type == 'sym':
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        # d_inv_sqrt = np.power(rowsum, -0.5)
        # d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        # return adj*d_inv_sqrt*d_inv_sqrt.flatten()
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()
    elif type == 'rw':
        rowsum = np.array(adj.sum(1))
        d_inv = np.power(rowsum, -1.0).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        adj_normalized = d_mat_inv.dot(adj)
        return adj_normalized


def preprocess_adj(adj, type='sym', loop=True):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    if loop:
        adj = adj + sp.eye(adj.shape[0])
    adj_normalized = normalize_adj(adj, type=type)
    return adj_normalized


def to_onehot(prelabel):
    k = len(np.unique(prelabel))
    label = np.zeros([prelabel.shape[0], k])
    label[range(prelabel.shape[0]), prelabel] = 1
    label = label.T
    return label


def square_dist(prelabel, feature):
    if sp.issparse(feature):
        feature = feature.todense()
    feature = np.array(feature)


    onehot = to_onehot(prelabel)

    m, n = onehot.shape
    count = onehot.sum(1).reshape(m, 1)
    count[count==0] = 1

    mean = onehot.dot(feature)/count
    a2 = (onehot.dot(feature*feature)/count).sum(1)
    pdist2 = np.array(a2 + a2.T - 2*mean.dot(mean.T))

    intra_dist = pdist2.trace()
    inter_dist = pdist2.sum() - intra_dist
    intra_dist /= m
    inter_dist /= m * (m - 1)
    return intra_dist

def trace_dist(prelabel, LA, LS):
    F = prelabel.T  # to_onehot(prelabel)

    struc_dist = np.trace(F.dot(LA).dot(F.T))
    fea_dist = np.trace(F.dot(LS).dot(F.T))

    return struc_dist, fea_dist

def dist(prelabel, feature):
    k = len(np.unique(prelabel))
    intra_dist = 0

    for i in range(k):
        Data_i = feature[np.where(prelabel == i)]

        Dis = euclidean_distances(Data_i, Data_i)
        n_i = Data_i.shape[0]
        if n_i == 0 or n_i == 1:
            intra_dist = intra_dist
        else:
            intra_dist = intra_dist + 1 / k * 1 / (n_i * (n_i - 1)) * sum(sum(Dis))


    return intra_dist


if __name__ == '__main__':

    dataset = 'cora'
    flag = 'spectral' #spectral or kmeans, default is spectral
    method = 'AGC' #AGC or IAGC
    max_iter = 60
    rep = 10

    data = sio.loadmat('{}.mat'.format(dataset))
    feature = data['fea']
    if sp.issparse(feature):
        feature = feature.todense()
    if method == 'AGC':
        PCAflag = False
    else:
        PCAflag = True

    if PCAflag==True:
        pca = PCA(n_components=100)
        pca.fit(feature)
        feature = pca.transform(feature)

    adj = data['W']
    gnd = data['gnd']
    gnd = gnd.T
    gnd = gnd - 1
    gnd = gnd[0, :]
    k = len(np.unique(gnd))
    Dis = euclidean_distances(feature, feature)
    d_avg = (np.max(np.array(adj.sum(1))))
    S = 1 / (Dis + 1)
    S = torch.from_numpy(S)
    m, _ = torch.sort(S, dim=1, descending=True)
    eps = m[:, round(d_avg)]
    eps = eps.reshape(-1, 1)
    tol = 0
    S_m = S - eps
    S[S_m < tol] = 0
    S = (S + S.t()) / 2
    S = S.numpy()

    adj = sp.coo_matrix(adj)
    intra_list = []
    intra_list.append(10000)
    struc_list = []
    fea_list = []
    acc_list = []
    nmi_list = []
    f1_list = []
    ar_list = []
    stdacc_list = []
    stdnmi_list = []
    stdf1_list = []
    stdar_list = []

    t = time.time()
    adj_normalized = preprocess_adj(adj)
    adj_normalized = (sp.eye(adj_normalized.shape[0]) + adj_normalized) / 2
    total_dist = []

    A = adj.astype(np.float32)

    LA = np.eye(A.shape[0]) - normalize_adj(A, type='sym')
    LS = np.eye(S.shape[0]) - normalize_adj(S, type='sym')

    tt = 0
    while 1:
        tt = tt + 1
        power = tt
        intraD = np.zeros(rep)
        strucD = np.zeros(rep)
        feaD = np.zeros(rep)

        ac = np.zeros(rep)
        nm = np.zeros(rep)
        f1 = np.zeros(rep)
        ar = np.zeros(rep)

        feature = adj_normalized.dot(feature)
        oofeature = sklearn.preprocessing.normalize(feature, norm='l2', axis=0)
        if flag == 'spectral':
            if PCAflag == False:
                u, s, v = sp.linalg.svds(feature, k=k, which='LM')
            else:
                feature = torch.from_numpy(feature)
                K = torch.nn.functional.relu(torch.matmul(feature, feature.t()))

                D = torch.sum(K, dim=0)
                sqrt_D = torch.diag(torch.pow(D, -0.5))
                K = torch.matmul(torch.matmul(sqrt_D, K), sqrt_D)
                K[np.isnan(K)] = 0.
                w, u = sp.linalg.eigs(K.numpy(), k=k, which='LM')
                u = np.real(u)
                u = u[:, 0:k]
        else:
            u = sklearn.preprocessing.normalize(feature, norm='l2', axis=1)


        for i in range(rep):
            clustering = KMeans(n_clusters=k).fit(u)
            predict_labels = clustering.labels_
            strucD[i], feaD[i] = trace_dist(oofeature, LA, LS)
            intraD[i] = dist(predict_labels, feature)
            cm = clustering_metrics(gnd, predict_labels)
            ac[i], nm[i], f1[i], ar[i] = cm.evaluationClusterModelFromLabel()

        intramean = np.mean(intraD)
        strucmean = np.mean(strucD)
        feamean = np.mean(feaD)
        acc_means = np.mean(ac)
        acc_stds = np.std(ac)
        nmi_means = np.mean(nm)
        nmi_stds = np.std(nm)
        f1_means = np.mean(f1)
        f1_stds = np.std(f1)
        ar_means = np.mean(ar)
        ar_stds = np.std(ar)

        intra_list.append(intramean)
        struc_list.append(strucmean)
        fea_list.append(feamean)
        acc_list.append(acc_means)
        stdacc_list.append(acc_stds)
        nmi_list.append(nmi_means)
        stdnmi_list.append(nmi_stds)
        f1_list.append(f1_means)
        stdf1_list.append(f1_stds)
        ar_list.append(ar_means)
        stdar_list.append(ar_stds)
        print('power: {}'.format(power),
              'intra_dist: {}'.format(intramean),
              'acc_mean: {}'.format(acc_means),
              'acc_std: {}'.format(acc_stds),
              'nmi_mean: {}'.format(nmi_means),
              'nmi_std: {}'.format(nmi_stds),
              'f1_mean: {}'.format(f1_means),
              'f1_std: {}'.format(f1_stds),
              'ar_mean: {}'.format(ar_means),
              'ar_std: {}'.format(ar_stds)
              )

        if intra_list[tt] > intra_list[tt - 1] or tt > max_iter:#
            print('AGC bestpower t-1 is: {}'.format(tt - 1))
            break

    if method == 'IAGC':
        x = np.arange(len(struc_list))
        y = np.array(struc_list) / np.array(fea_list)
        kneedle = KneeLocator(x, y.reshape(-1), S=1.0, curve="convex", direction="decreasing")
        e = kneedle.knee+1 #count from 1
        print("The elbow point e is:", e)
        print("IAGC power is:", np.ceil(0.5*(e+tt-1)))



