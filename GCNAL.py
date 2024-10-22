import numpy as np
from core_set import *
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.optim as optim
import math
from tqdm import tqdm
from torch.utils.data import Dataset
import gc



device = torch.device("cuda:0")

def UncertainGCN(u_features, l_features,n):

    features = torch.cat([u_features, l_features], dim=0)

    features = nn.functional.normalize(features.to(device))

    adj = aff_to_adj(features)

    gcn_model = GCN(nfeat=features.shape[1],
                     nhid=128,
                     nclass=1,
                     dropout=0.3).to(device)

    optim_gcn = optim.Adam(gcn_model.parameters(), lr=1e-3, weight_decay=5e-4)

    nlbl = np.arange(0, u_features.size(0), 1)
    lbl = np.arange(u_features.size(0), features.size(0), 1)

    print('Learning Graph Convolution Network...')
    gcn_model.train()
    for _ in tqdm(range(200)):
        optim_gcn.zero_grad()
        outputs, _, _ = gcn_model(features, adj)
        loss = BCEAdjLoss(outputs, lbl, nlbl, 1.2)
        loss.backward()
        optim_gcn.step()

    gcn_model.eval()
    with torch.no_grad():
        with torch.cuda.device(device):
            inputs = features.cuda()
            #labels = binary_labels.cuda()
        scores, _, feat = gcn_model(inputs, adj)


        s_margin = 0.1
        scores_median = np.squeeze(torch.abs(scores[nlbl] - s_margin).detach().cpu().numpy())
        chosen = np.argsort(-(scores_median))[-n:]

    del gcn_model, optim_gcn, feat, features, adj
    torch.cuda.empty_cache()
    gc.collect()

    return chosen


def CoreGCN(u_features, l_features, n):

    features = torch.cat([u_features, l_features], dim=0)

    features = nn.functional.normalize(features.to(device))

    adj = aff_to_adj(features)

    gcn_model = GCN(nfeat=features.shape[1],
                     nhid=128,
                     nclass=1,
                     dropout=0.3).to(device)

    optim_gcn = optim.Adam(gcn_model.parameters(), lr=1e-3, weight_decay=5e-4)

    nlbl = np.arange(0, u_features.size(0), 1)
    lbl = np.arange(u_features.size(0), features.size(0), 1)
    labeled_indices = np.arange(0, len(l_features))
    unlabeled_rows = np.arange(0, len(u_features))
    print('Learning Graph Convolution Network...')
    gcn_model.train()
    for _ in tqdm(range(200)):
        optim_gcn.zero_grad()
        outputs, _, _ = gcn_model(features, adj)
        loss = BCEAdjLoss(outputs, lbl, nlbl, 1.2)
        loss.backward()
        optim_gcn.step()

    gcn_model.eval()
    with torch.no_grad():
        with torch.cuda.device(device):
            inputs = features.cuda()
            #labels = binary_labels.cuda()
        scores, _, feat = gcn_model(inputs, adj)

        feat = feat.detach().cpu().numpy()

        coreset = Coreset_Greedy(feat)
        new_batch, _ = coreset.sample(labeled_indices, n)
        new_batch = [i - len(feat) for i in new_batch]
        chosen = unlabeled_rows[new_batch]

    del gcn_model, optim_gcn, feat, features, adj
    torch.cuda.empty_cache()
    gc.collect()

    return chosen

class GCNDataset(Dataset):
    def __init__(self, features, adj, labeled):
        self.features = features
        self.labeled = labeled
        self.adj = adj

    def __getitem__(self, index):
        return self.features[index], self.adj[index], self.labeled[index]

    def __len__(self):
        return len(self.features)


def aff_to_adj(x):
    x = x.detach().cpu().numpy()
    adj = np.matmul(x, x.transpose())
    adj +=  -1.0*np.eye(adj.shape[0])
    adj_diag = np.sum(adj, axis=0) #rowise sum
    adj = np.matmul(adj, np.diag(1/adj_diag))
    adj = adj + np.eye(adj.shape[0])
    adj = torch.Tensor(adj).cuda()

    return adj


def BCEAdjLoss(scores, lbl, nlbl, l_adj):
    lnl = torch.log(scores[lbl])
    lnu = torch.log(1 - scores[nlbl])
    labeled_score = torch.mean(lnl)
    unlabeled_score = torch.mean(lnu)
    bce_adj_loss = -labeled_score - l_adj*unlabeled_score
    return bce_adj_loss


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.linear = nn.Linear(nclass, 1)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        feat = F.dropout(x, self.dropout, training=self.training)
        x = self.gc3(feat, adj)
        #x = self.linear(x)
        # x = F.softmax(x, dim=1)
        return torch.sigmoid(x), feat, torch.cat((feat,x),1)