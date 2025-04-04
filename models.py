import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphSAGE
from torch_geometric.nn.conv import GCNConv, AGNNConv, GATConv, SGConv

class MORAL(nn.Module):
    def __init__(self, num_feature, struc_input, hidden_size, num_class, dropout):
        super().__init__()

        self.dropout = dropout
        self.alpha_s = nn.Parameter(torch.tensor(0.4), requires_grad=True)
        self.alpha_a = nn.Parameter(torch.tensor(0.6), requires_grad=True)
        self.embedding = nn.Embedding(6,hidden_size)
        self.embedding_W = nn.Embedding(hidden_size,1)
        self.feature_layer = GCNConv(num_feature, hidden_size)
        self.strucure_layer = GCNConv(struc_input, hidden_size)
        self.representation_layer = GCNConv((hidden_size*2), num_class)
        
    def forward(self, feature, structure, edge_index):
        feature = self.feature_layer(feature, edge_index)
        attr = F.dropout(feature, p=self.dropout, training=self.training)
        device = torch.device("cuda")
        embeddings = self.embedding(torch.arange(6).to(device))
        embeddings_W=self.embedding_W(torch.arange(64).to(device))
        embedding=embeddings@embeddings_W
        embedding=torch.sigmoid(embedding)
        embedding_total=embedding.clone()
        for i in range(0,6):
            if embedding_total[i]<0.5:
                embedding_total[i]=0.0
            else:
                embedding_total[i]=1.0
        structure=structure*embedding_total.t()
        structure = self.strucure_layer(structure, edge_index)
        struc = F.dropout(structure, p=self.dropout, training=self.training)
        alpha = F.normalize(torch.tensor([[self.alpha_s.data, self.alpha_a.data]]), p=1)
        self.alpha_s.data, self.alpha_a.data = alpha[0][0], alpha[0][1]
        x = torch.cat((attr* self.alpha_a*2,struc* self.alpha_s*2),1)
        y = self.representation_layer(x, edge_index)
        return feature, structure, y

class MORE(torch.nn.Module):
    def __init__(self, num_feature, struc_input, hidden_size, out_channels, dropout):
        super().__init__()
        self.feat_conv = GCNConv(num_feature, hidden_size, cached=True)
        self.struc_conv = GCNConv(struc_input, hidden_size, cached=True)
        self.hidden_conv = GCNConv(hidden_size, out_channels, cached=True)
        self.dropout = dropout

    def forward(self, feat_info, struc_info, edge_index):
        x1 = self.feat_conv(feat_info, edge_index).relu()
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        x2 = self.struc_conv(struc_info, edge_index).relu()
        x2 = F.dropout(x2, p=self.dropout, training=self.training)
        x = x1 + x2
        x = self.hidden_conv(x, edge_index)
        return x

class SGC(torch.nn.Module):
    def __init__(self, num_features, num_classes, dropout):
        super().__init__()
        self.conv1 = SGConv(num_features, num_classes, K=2, cached=True)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x, edge_index = x, edge_index
        x = self.conv1(x, edge_index)
        return x

class AGNN(torch.nn.Module):
    def __init__(self, num_features, num_classes, dropout):
        super().__init__()
        self.lin1 = torch.nn.Linear(num_features, 16)
        self.prop1 = AGNNConv(requires_grad=False)
        self.prop2 = AGNNConv(requires_grad=True)
        self.lin2 = torch.nn.Linear(16, num_classes)
        self.dropout = dropout 

    def forward(self, x, edge_index):
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = F.relu(self.lin1(x))
        x = self.prop1(x, edge_index)
        x = self.prop2(x, edge_index)
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.lin2(x)
        return x
    
class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads, dropout):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads, dropout=0.6)
        # On the Pubmed dataset, use `heads` output heads in `conv2`.
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1,
                             concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=True)
        self.conv2 = GCNConv(hidden_channels, out_channels, cached=True)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x

class GraphSage(torch.nn.Module):
    def __init__(self, num_features, hidden_size, num_class):
        super().__init__()
        self.graphsage = GraphSAGE(
                            in_channels=num_features,
                            hidden_channels=hidden_size,
                            out_channels=num_class,
                            num_layers=2,
                        )
    def forward(self, x, edge_index):
        x = self.graphsage(x, edge_index)
        return x

class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, dropout):
        super().__init__()
        self.lin1 = nn.Linear(input_size, hidden_size1)
        self.lin2 = nn.Linear(hidden_size1, hidden_size2)
        self.lin3 = nn.Linear(hidden_size2, output_size)
        self.dropout = dropout
    
    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin3(x)
        return x