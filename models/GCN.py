from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from models.common_blocks import batch_norm
from regularization import *
from infomax import Infomax
from torch_geometric.utils import to_dense_adj, dropout_adj

def correlation_coefficient(x, y):
    # 입력 벡터가 2D인지 확인하고 1D로 변환
    x = x.view(-1)
    y = y.view(-1)
    
    # 각 벡터의 평균 구하기
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)

    # 분산 및 공분산 계산
    cov_xy = torch.mean((x - mean_x) * (y - mean_y))
    std_x = torch.std(x)
    std_y = torch.std(y)
    
    # 상관계수 계산
    corr = cov_xy / (std_x * std_y)
    
    # 절대값으로 변환
    corr = torch.abs(corr)
    
    return corr

class GCN(nn.Module):
    def __init__(self, args):
        super(GCN, self).__init__()
        self.dataset = args.dataset
        self.num_layers = args.num_layers
        self.num_feats = args.num_feats
        self.num_classes = args.num_classes
        self.dim_hidden = args.dim_hidden

        self.dropout = args.dropout
        self.dropedge = args.dropedge
        self.cached = True if not args.dropedge else False # only could be used in transductive learning
        self.layers_GCN = nn.ModuleList([])
        self.layers_bn = nn.ModuleList([])
        self.type_norm = args.type_norm
        self.skip_weight = args.skip_weight
        self.num_groups = args.num_groups

        ###
        self.alpha = args.alpha
        self.beta = args.beta
        self.gamma = args.gamma
        self.loss_corr = 0
        # self.infomax = Infomax(args.dim_hidden, args.dim_hidden)
        self.infomax = Infomax(self.num_feats, args.dim_hidden)
        self.infomax2 = Infomax(1, 1)
        self.infomax3 = Infomax(1, 1)
        


        # build up the convolutional layers
        if self.num_layers == 1:
            self.layers_GCN.append(GCNConv(self.num_feats, self.num_classes, cached=self.cached, bias=False))
        elif self.num_layers == 2:
            self.layers_GCN.append(GCNConv(self.num_feats, self.dim_hidden, cached=self.cached, bias=False))
            self.layers_GCN.append(GCNConv(self.dim_hidden, self.num_classes, cached=self.cached, bias=False))
        else:
            self.layers_GCN.append(GCNConv(self.num_feats, self.dim_hidden, cached=self.cached, bias=False))
            for _ in range(self.num_layers-2):
                self.layers_GCN.append(GCNConv(self.dim_hidden, self.dim_hidden, cached=self.cached, bias=False))
            self.layers_GCN.append(GCNConv(self.dim_hidden, self.num_classes, cached=self.cached, bias=False))

        # build up the normalization layers
        for i in range(self.num_layers):
            dim_out = self.layers_GCN[i].out_channels
            if self.type_norm in ['None', 'batch', 'pair']:
                skip_connect = False
            else:
                skip_connect = True
            self.layers_bn.append(batch_norm(dim_out, self.type_norm, skip_connect, self.num_groups, self.skip_weight))

    def reset_parameters(self):
        def weight_reset(m):
            if isinstance(m, GCNConv):
                m.reset_parameters()
        self.apply(weight_reset)

    def forward(self, x, edge_index, y, report_metrics=False):

        if not isinstance(y, bool):
            y = y.unsqueeze(1).float()      # [2708] -> [2708, 1]

        if self.dropedge:
            edge_index = dropout_adj(edge_index, p=1-self.dropedge, training=self.training)[0]

        for i in range(self.num_layers):

            if i == 0 or i == self.num_layers-1:
                x_1 = x
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.layers_GCN[i](x, edge_index)
            x = self.layers_bn[i](x)

            if self.training:

                ####input x <-> hidden representation x 간의 infomax######
                if self.beta > 0 and i!=self.num_layers-1 and i%5==4:
                    print(x_1.shape, x.shape)

                    self.loss_corr += self.beta * self.infomax.get_loss(x_1, x)

                ######My Method######
                if self.beta > 0 and i!=self.num_layers-1 and i%2==1:
                    _, f_i, f_j = get_random_dimension_pair(x)
                    f_i = f_i.unsqueeze(1).float()  # [2708] -> [2708, 1]
                    f_j = f_j.unsqueeze(1).float()  # [2708] -> [2708, 1]
                    cc3 = correlation_coefficient(f_j, f_i)

                    self.loss_corr += (self.infomax2.get_loss(f_i, y) + self.infomax2.get_loss(f_j, y) + cc3)/3
                
                
                ######Sparse######
                # if  i == self.num_layers // 2:
                #     mask = torch.rand(x.shape, device=x.device) > 0.1  # 10% chance of masking
                #     x = x * mask.float()  # Apply the mask to make x sparse


            if i == self.num_layers-2:
                corr = torch_corr(x.t())
                corr = torch.triu(corr, 1).abs()
                n = corr.shape[0]
                self.corr_2 = corr.sum().item() / (n *(n-1) /2)
                self.sim_2 = -1

                if report_metrics:
                    self.sim_2 = get_pairwise_sim(x)

            if i == self.num_layers-1:
                corr = torch_corr(x.t())
                corr = torch.triu(corr, 1).abs()
                n = corr.shape[0]
                self.corr = corr.sum().item() / (n *(n-1) /2)
                self.sim = -1
                if report_metrics:
                    self.sim = get_pairwise_sim(x)

            if i != self.num_layers-1:
                x = F.relu(x)

        return x
