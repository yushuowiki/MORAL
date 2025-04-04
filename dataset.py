import torch
import torch.nn.functional as F
import networkx as nx
from torch_geometric.data import Dataset
from torch_geometric.utils import to_networkx
from rawdata_process import get_data
from cal_motif import get_motif
import powerlaw
import numpy as np

class MyDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.feature = data.x
        self.structure= data.structure_info
        self.labels = data.y
        self.edge_index = data.edge_index

    def get(self, idx):
        return self.data

    def len(self):
        return 1

def get_dataset(args):
    name = args['dataset']
    ratio = args['label_ratio']
    assert name in ['cora', 'acm', 'citation', 'dblp', 'wikipedia', 'citeseer', 'email', 'facebook', 'terror', 'polblogs']
    assert ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    data = get_data(name)
    data.x = F.normalize(data.x)
    remask_by_ratio(data, ratio)

    graph = to_networkx(data)
    motif = get_motif(name, graph)
    
    model=args['model_name']
    if model in ['MORAL']:
        a=[]
        motif_t=motif.T
        for i in range(6):
            if torch.max(motif_t[i])>0:
                fit = powerlaw.Fit(motif_t[i], bool = True,estimate_discrete=False,xmin_distance='D')
                a.append(fit.power_law.alpha)
            else:
                a.append(0)
        for j in range(6):
            if a[j]>0:
                motif_t[j]=motif_t[j]**(1/np.sqrt(a[j]))
        motif=motif_t.T
        print(a)
    motif = F.normalize(motif)
    data.structure_info = motif
    dataset = MyDataset(data)
    return dataset

def remask_by_ratio(single_metadata, label_rate): # randomly recalculate the masks of the Data() by a specific label ratio
    return simple(single_metadata, label_rate)

def simple(single_metadata, label_rate):
    # statistic
    data_len = single_metadata.x.shape[0]

    # recalculate the train_mask with label_rate
    train_mask = torch.tensor([False] * data_len)
    train_len = round(data_len * label_rate)
    idx = (train_mask != True).nonzero(as_tuple=False).view(-1)
    idx = idx[torch.randperm(idx.size(0))]
    train_mask[idx[:train_len]] = True

    # recalculate the val_mask by rate 20% and test_mask by supplyment
    val_mask = torch.tensor([False] * data_len)
    test_mask = torch.tensor([False] * data_len)
    val_len = round(data_len * 0.2)
    test_len = round(data_len * (1.0 - label_rate - 0.2))
    idx = (train_mask != True).nonzero(as_tuple=False).view(-1)
    idx = idx[torch.randperm(idx.size(0))]
    val_mask[idx[:val_len]] = True
    test_mask[idx[test_len:]] = True

    # assign masks
    single_metadata.train_mask = train_mask
    single_metadata.test_mask = test_mask
    single_metadata.val_mask = val_mask

def statistic(name, adjust=False): 
    data = torch.load(f'./datasets/obj/{name}.pt')
    motif = torch.load(f'./datasets/motif/{name}.pt').sum(dim=0).numpy().tolist()
    nodes = data.x.shape[0]
    edges = data.edge_index.shape[1]
    features = data.x.numpy().shape[1]
    classes = get_num_class(data)
    
    graph = to_networkx(data)
    degree = torch.tensor(list(dict(graph.degree).values()), dtype=torch.float32).mean().item()
    cluster = nx.average_clustering(graph)
    if adjust:
        motif = [motif[0] / 3, motif[1] / 3, motif[2] / 4, motif[3] / 4, motif[4] / 4]
    else:
        motif = [motif[0], motif[1], motif[2], motif[3], motif[4]]
    
    print(nodes, edges, features, classes, degree, cluster)
    print(motif)

    s = [nodes, edges, features, classes, degree, cluster] + motif
    return s

def get_num_class(data):
    if len(data.y.shape) > 1:
        return data.y.shape[1]

    dict = {}
    for c in data.y:
        c = c.item()
        dict[c] = dict.get(c, 0) + 1
    num_class = len(dict)

    return num_class

def get_data_info(data):
    num_feature = data.x.numpy().shape[1]
    num_class = get_num_class(data)
    struc_input = 6

    dic = {
        'num_feature': num_feature,
        'num_class': num_class,
        'struc_input': struc_input,
    }
    return dic

if __name__ == "__main__":
    import csv
    names = ['cora', 'citation', 'dblp', 'wikipedia', 'citeseer', 'email', 'facebook', 'polblogs', 'terror']
    f = open('./statistic.csv', 'w')
    writer = csv.writer(f)
    writer.writerow(['', 'nodes', 'edges', 'features', 'classes', 'degree', 'cluster', 'M31', 'M32', 'M41', 'M42', 'M43'])
    for name in names:
        print(name)
        s = statistic(name, adjust=True)
        s = [name] + s
        writer.writerow(s)
    f.close()