import os
import csv
import torch
import torch.nn.functional as F
import numpy as np
import scipy.io as sio
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid

def get_data(dataset, redo=False):
    obj_file = f'./datasets/obj/{dataset}.pt'
    if not redo and os.path.exists(obj_file):
        data = torch.load(obj_file)
        return data
    
    if dataset == 'cora':
        data = get_cora()
    elif dataset == 'citeseer':
        data = get_citeseer()
    elif dataset == 'acm':
        data = get_acm()
    elif dataset == 'citation':
        data = get_citation()
    elif dataset == 'dblp':
        data = get_dblp()
    elif dataset == 'wikipedia':
        data = get_wikipedia()
    elif dataset == 'email':
        data = get_email()
    elif dataset == 'facebook':
        data = get_facebook()
    elif dataset == 'polblogs':
        data = get_polblogs()
    elif dataset == 'terror':
        data = get_terror()   
    verify_data(data)

    return data

def verify_data(data):
    ''' verify the property of data.Data() after processing raw data from raw files. '''
    feature = data.x
    labels = data.y
    edge_index = data.edge_index

    # type
    assert isinstance(feature, torch.Tensor)
    assert isinstance(labels, torch.Tensor)
    assert isinstance(edge_index, torch.Tensor)
    assert isinstance(feature[0][0].item(), float)
    if len(labels.numpy().shape) == 1:
        assert isinstance(labels[0].item(), int)
    else: # len = 2
        assert isinstance(labels[0][0].item(), float) # probabilities
    assert isinstance(edge_index[0][0].item(), int)

    # shape
    assert len(feature.numpy().shape) == 2
    assert edge_index.shape[0] == 2
    assert len(edge_index.numpy().shape) == 2

    # labels should be true label or propabilities
    if len(labels.numpy().shape) == 1:
        assert labels.min().item() == 0 # labels index begin with 0
    else:
        assert (labels.sum(dim=-1) != 1.0).sum().item() == 0 # sum of propabilities is 1

    # adj symmetric
    num_edges = edge_index.shape[1]
    num_nodes = labels.numpy().shape[0]
    count1, count2 = 0, 0

    edge_dict = {i:[] for i in range(num_nodes)}
    edge_index = edge_index.T.numpy().tolist()
    for edge in edge_index:
        if edge[1] not in edge_dict[edge[0]]:
            edge_dict[edge[0]].append(edge[1])
    edge_index_reverse = [[edge[0], edge[1]] for edge in edge_index]
    for edge_reverse in edge_index_reverse:
        if edge_reverse[1] in edge_dict[edge_reverse[0]]: # check asymmetric
            count1 += 1
        if edge_reverse[0] == edge_reverse[1]: # check self loop
            count2 += 1
    
    assert count1 == num_edges # all the edges should have it's reverse in the edge_index
    assert count2 == 0         # there are no self-loops

def edge_symmetric(edge_index, num_nodes):
    # if isinstance(edge_index, torch.Tensor):
    #     raise ValueError(f"edge_index should be torch.Tensor, while the input's type is {type(edge_index)}")
    if edge_index.shape[0] != 2 or len(edge_index.numpy().shape) != 2:
        raise ValueError('the shape of edge_index should be [2, len]')
    edge_index = edge_index.T.numpy().tolist()

    # delete self loop
    edge_index = [edge for edge in edge_index if edge[0]!=edge[1]]
    
    # symmetric
    edge_dict = {i:[] for i in range(num_nodes)}
    for edge in edge_index:
        if edge[1] not in edge_dict[edge[0]]:
            edge_dict[edge[0]].append(edge[1])
    edge_index_reverse = [[edge[0], edge[1]] for edge in edge_index]
    for edge_reverse in edge_index_reverse:
        if edge_reverse[1] not in edge_dict[edge_reverse[0]]:
            edge_index.append(edge_reverse)

    edge_index = torch.tensor(edge_index).T
    return edge_index


''' Cora '''


def get_cora():
    dataset = Planetoid(root='./datasets/raw/cora', name='Cora')
    cora = dataset[0]
    del cora.train_mask
    del cora.test_mask
    del cora.val_mask
    torch.save(cora, './datasets/obj/cora.pt')
    return cora


''' Citeseer '''


def get_citeseer():
    dataset = Planetoid(root='./datasets/raw/citeseer', name='Citeseer')
    citeseer = dataset[0]
    del citeseer.train_mask
    del citeseer.test_mask
    del citeseer.val_mask
    torch.save(citeseer, './datasets/obj/citeseer.pt')
    return citeseer

''' Wikipedia '''
def get_wikipedia():
    # label
    labels = np.loadtxt('./datasets/raw/wikipedia/Wiki_labels.txt', skiprows=0, dtype=int)
    labels = torch.tensor(labels[:, 1])
    num_nodes = labels.shape[0]

    # edge_index
    edge_index = np.loadtxt('./datasets/raw/wikipedia/Wiki_edgelist.txt', skiprows=0, dtype=int)
    edge_index = torch.tensor(edge_index).T
    edge_index = edge_symmetric(edge_index, num_nodes)

    # feature
    # It's incorrect that Wiki_category.txt is totally the same as Wiki_labels.txt
    feature = [i*[0.0] + [1.0] + (num_nodes-1-i)*[0.0] for i in range(num_nodes)]
    feature = torch.tensor(feature, dtype=torch.float32)

    wikipedia = Data(x=feature, edge_index=edge_index, y=labels)
    torch.save(wikipedia, './datasets/obj/wikipedia.pt')

    return wikipedia

''' Citationv1 '''
def get_citation():
    # .mat to np
    raw_data = sio.loadmat('./datasets/raw/citation/citationv1.mat')

    # labels
    labels = raw_data['group']
    labels = [np.where(labs==1)[0][0] for labs in labels]
    labels = torch.tensor(labels)

    # feature
    feature = raw_data['attrb']
    feature = torch.tensor(feature, dtype=torch.float32)

    # edge_index
    adj_matrix = raw_data['network'].A # sparse matrix to np.ndarray
    edge_index = []
    for i, node in enumerate(adj_matrix):
        adj_node_idxs = np.where(node==1)[0]
        edges = [[i, adj_node_idx] for adj_node_idx in adj_node_idxs]
        if len(edges) != 0:
            edge_index += edges
    edge_index = torch.tensor(edge_index).T
    num_nodes = labels.numpy().shape[0]
    edge_index = edge_symmetric(edge_index, num_nodes)
    
    citation = Data(x=feature, edge_index=edge_index, y=labels)
    torch.save(citation, './datasets/obj/citation.pt')

    return citation

''' DBLPv7 '''
def get_dblp():
    # .mat to np
    raw_data = sio.loadmat('./datasets/raw/dblp/dblpv7.mat')
    
    # labels
    labels = raw_data['group']
    labels = [np.where(labs==1)[0][0] for labs in labels]
    labels = torch.tensor(labels)

    # feature
    feature = raw_data['attrb']
    feature = torch.tensor(feature, dtype=torch.float32)

    # edge_index
    adj_matrix = raw_data['network'].A # sparse matrix to np.ndarray
    edge_index = []
    for i, node in enumerate(adj_matrix):
        adj_node_idxs = np.where(node==1)[0]
        edges = [[i, adj_node_idx] for adj_node_idx in adj_node_idxs]
        if len(edges) != 0:
            edge_index += edges
    edge_index = torch.tensor(edge_index).T
    num_nodes = labels.numpy().shape[0]
    edge_index = edge_symmetric(edge_index, num_nodes)
    
    dblp = Data(x=feature, edge_index=edge_index, y=labels)
    torch.save(dblp, './datasets/obj/dblp.pt')

    return dblp

'''ACMv9'''
def get_acm():
    # .mat to np
    raw_data = sio.loadmat('./datasets/raw/acm/acmv9.mat')
    
    # labels
    labels = raw_data['group']
    # multi labels
    # labels = torch.tensor(labels, dtype=torch.float32)
    # labels = F.normalize(labels, p=1)
    # single label
    labels = [np.where(labs==1)[0][0] for labs in labels]
    labels = torch.tensor(labels)

    # feature
    feature = raw_data['attrb']
    feature = torch.tensor(feature, dtype=torch.float32)

    # edge_index
    adj_matrix = raw_data['network'].A # sparse matrix to np.ndarray
    edge_index = []
    for i, node in enumerate(adj_matrix):
        adj_node_idxs = np.where(node==1)[0]
        edges = [[i, adj_node_idx] for adj_node_idx in adj_node_idxs]
        if len(edges) != 0:
            edge_index += edges
    edge_index = torch.tensor(edge_index).T
    num_nodes = labels.numpy().shape[0]
    edge_index = edge_symmetric(edge_index, num_nodes)
    
    acm = Data(x=feature, edge_index=edge_index, y=labels)
    torch.save(acm, './datasets/obj/acm.pt')

    return acm

'''Email-Eucore'''
def get_email():
    # label
    labels = np.loadtxt('./datasets/raw/email/email-Eucore_Label.csv', skiprows=1, dtype=int, delimiter=',')
    labels = torch.tensor(labels[:, 1])
    num_nodes = labels.shape[0]

    # feature
    feature = [i*[0.0] + [1.0] + (num_nodes-1-i)*[0.0] for i in range(num_nodes)]
    feature = torch.tensor(feature, dtype=torch.float32)

    # edge_index
    edge_index = np.loadtxt('./datasets/raw/email/email-Eucore_Data.csv', skiprows=0, dtype=int, delimiter=',')
    edge_index = torch.tensor(edge_index).T
    edge_index = edge_symmetric(edge_index, num_nodes)

    email = Data(x=feature, edge_index=edge_index, y=labels)
    torch.save(email, './datasets/obj/email.pt')

    return email

'''Fackbook'''
def get_facebook():
    # label
    labels = np.loadtxt('./datasets/raw/facebook/fb_CMU_Carnegie49_Label.csv', skiprows=1, dtype=int, delimiter=',')
    labels = torch.tensor(labels[:, 1]) - 1 # begin with 0
    num_nodes = labels.shape[0]

    # feature
    feature = [i*[0.0] + [1.0] + (num_nodes-1-i)*[0.0] for i in range(num_nodes)]
    feature = torch.tensor(feature, dtype=torch.float32)

    # edge_index
    edge_index = np.loadtxt('./datasets/raw/facebook/fb_CMU_Carnegie49_Data.csv', skiprows=1, dtype=int, delimiter=',')
    edge_index = torch.tensor(edge_index).T - 1 # begin with 0
    edge_index = edge_symmetric(edge_index, num_nodes)

    facebook = Data(x=feature, edge_index=edge_index, y=labels)
    torch.save(facebook, './datasets/obj/facebook.pt')

    return facebook

'''Ploblogs'''
def get_polblogs():
    # label
    labels = np.loadtxt('./datasets/raw/polblogs/polblogs_Label.csv', skiprows=1, dtype=int, delimiter=',')
    labels = torch.tensor(labels[:, 1]) - 1 # begin with 0
    num_nodes = labels.shape[0]

    # feature
    feature = [i*[0.0] + [1.0] + (num_nodes-1-i)*[0.0] for i in range(num_nodes)]
    feature = torch.tensor(feature, dtype=torch.float32)

    # edge_index
    edge_index = np.loadtxt('./datasets/raw/polblogs/polblogs_Data.csv', skiprows=1, dtype=int, delimiter=',')
    edge_index = torch.tensor(edge_index).T
    edge_index = edge_symmetric(edge_index, num_nodes)

    polblogs = Data(x=feature, edge_index=edge_index, y=labels)
    torch.save(polblogs, './datasets/obj/polblogs.pt')

    return polblogs

'''TerrorAttack'''
def get_terror():
    # label
    labels = np.loadtxt('./datasets/raw/terror/TerrorAttack_Label.csv', skiprows=1, dtype=int, delimiter=',')
    labels = torch.tensor(labels[:, 1]) - 1 # begin with 0

    # feature
    feature = np.loadtxt('./datasets/raw/terror/TerrorAttack_Feature.csv', skiprows=1, dtype=float, delimiter=',')
    feature = torch.tensor(feature[:, 1:], dtype=torch.float32) # delete the first col

    # edge_index
    edge_index = np.loadtxt('./datasets/raw/terror/TerrorAttack_Data.csv', skiprows=1, dtype=int, delimiter=',')
    edge_index = torch.tensor(edge_index).T
    num_nodes = labels.numpy().shape[0]
    edge_index = edge_symmetric(edge_index, num_nodes)

    terror = Data(x=feature, edge_index=edge_index, y=labels)
    torch.save(terror, './datasets/obj/terror.pt')

    return terror

if __name__ == "__main__":
    # ['acm', 'citation', 'citeseer', 'cora', 'dblp', 'email', 'facebook', 'polblogs','terror','wikipedia']
    for name in ['acm', 'citation', 'citeseer', 'cora', 'dblp', 'email', 'facebook', 'polblogs','terror', 'wikipedia']:
        print(name)
        data = get_data(name, redo=True)
        print(data)