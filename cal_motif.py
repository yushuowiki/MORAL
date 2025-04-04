import os
import torch
import numpy as np
import networkx as nx
import scipy.sparse as sp
from rawdata_process import get_data

def caculate_VFlag(Na, Nb, inse, node_num):
    """
    Caculate VFlag tag vector using to 4-order motif counting.
    
    Input:
        Na:         (list) Node a neighbor 
        Nb:         (list) Node b neighbor
        inse:       (list) The intersection of Na and Nb
        node_num:   (int) Graph node number
    Output:
        VFlag:      (array) VFLAG vector
    """
    
    VFlag = np.zeros(node_num, dtype = int)
    for node in Na:
        VFlag[node] = 1
    for node in Nb:
        VFlag[node] = 2
    for node in inse:
        VFlag[node] = 3

    return VFlag

def matrix2coo(m):
    """
    Change normal matrix to coo sparse matrix

    Input:
        m:      (np.array 2dim) input normal matrix
    Output:
        coo_m:  (sp.coo_matrix) output coo sparse matrix
    """

    rows, cols, values = [], [], []
    for i in range(0, m.shape[0]):
        for j in range(0, m.shape[1]):
            if m[i,j] != 0:
                rows.append(i)
                cols.append(j)
                values.append(m[i,j])
    coo_m = sp.coo_matrix((values, (rows, cols)), shape = m.shape, dtype = float)

    return coo_m

def count_motif(g, dataset,Sparse=False):
    """
    Calculate how many different kinds of motifs each node is in

    Input:
        g:              (networkx graph) input graph data
        Sparse:         (bool) Output matrix Sparse or not
    Output:
        motif_feature:  (matrix / coo_matix) motif feature matrix
    """

    # Initialize the motif feature dictionary
    node_num, node_list = g.number_of_nodes(), g.nodes()
    nm_dict = {}
    for node in node_list:
        nm_dict[node] = np.zeros(5, float)
    degree = dict(nx.degree(g))

    for node_a in node_list:
        Na = list(g.neighbors(node_a))
        for node_b in Na:
            if node_b < node_a:
                continue
            Nb = list(g.neighbors(node_b))
            inse = list(set(Na).intersection(set(Nb)))
            for node_c in inse:
                nm_dict[node_a][0] += 1/3
                nm_dict[node_b][0] += 1/3
                nm_dict[node_c][0] += 1/3
            VFlag = caculate_VFlag(Na, Nb, inse, node_num)
            VFlag[node_a] = 0
            VFlag[node_b] = 0
            for i in range(0, len(VFlag)):
                if VFlag[i] == 1 or VFlag[i] == 2:
                    nm_dict[node_a][1] += 1/2
                    nm_dict[node_b][1] += 1/2
                    nm_dict[i][1]      += 1/2
            for node_c in inse:
                Nc = list(g.neighbors(node_c))
                for node_d in Nc:
                    if VFlag[node_d] == 3:
                        nm_dict[node_a][2] += 1/12
                        nm_dict[node_b][2] += 1/12
                        nm_dict[node_c][2] += 1/12
                        nm_dict[node_d][2] += 1/12
                    elif VFlag[node_d] == 2 or VFlag[node_d] == 1:
                        nm_dict[node_a][3] += 1/4
                        nm_dict[node_b][3] += 1/4
                        nm_dict[node_c][3] += 1/4
                        nm_dict[node_d][3] += 1/4
            for node_c in Na:
                if VFlag[node_c] != 1 or node_c == node_b:
                    continue
                Nc = list(g.neighbors(node_c))
                for node_d in Nc:
                    if VFlag[node_d] == 2 and node_d != node_a:
                        nm_dict[node_a][4] += 1/4
                        nm_dict[node_b][4] += 1/4
                        nm_dict[node_c][4] += 1/4
                        nm_dict[node_d][4] += 1/4
    motif_feature = []
    for node in node_list:
        temp = [degree[node]]
        temp.extend(list(nm_dict[node]))
        motif_feature.append(temp)

    if Sparse == True:
        motif_feature = matrix2coo(np.matrix(motif_feature))
    else:
        motif_feature = np.matrix(motif_feature)

    motif_feature = torch.tensor(motif_feature, dtype=torch.float32)
    torch.save(motif_feature, f'./datasets/motif/{dataset}.pt')
    return motif_feature

def get_motif(dataset, graph, redo=False):
    obj_file = f'./datasets/motif/{dataset}.pt'
    if not redo and os.path.exists(obj_file):
        motif = torch.load(obj_file)
        return motif
    
    motif = count_motif(graph,dataset)

    return motif

if __name__ == '__main__':
    from torch_geometric.utils import to_networkx
    # ['acm', 'citation', 'citeseer', 'cora', 'dblp', 'email', 'facebook', 'polblogs', 'terror', 'wikipedia']
    for name in ['acm', 'citation', 'citeseer', 'cora', 'dblp', 'email', 'facebook', 'polblogs', 'terror', 'wikipedia']:
        data = get_data(name)
        print(name, data)
        graph = to_networkx(data)
        motif = get_motif(name, graph, redo=True)