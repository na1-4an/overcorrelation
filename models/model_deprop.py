import torch
import torch.nn.functional as F
from models.prop_deprop import DeProp_Prop
from torch_geometric.nn.conv.gcn_conv import GCNConv
from regularization import  get_pairwise_sim, torch_corr
from infomax import Infomax
from regularization import *

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

class DeProp(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, args):
        super(DeProp, self).__init__()

        self.num_layers = args.num_layers
        self.orth = args.orth
        self.lambda1 = args.lambda1
        self.lambda2 = args.lambda2
        self.gamma = args.gamma2
        self.with_bn = args.with_bn
        self.F_norm = args.F_norm
        
        self.corr = 0
        self.corr_2 = 0
        self.sim = -1
        self.sim_2 = -1

        self.num_feats = args.num_feats
        self.dim_hidden = args.dim_hidden


        self.infomax = Infomax(self.num_feats, args.dim_hidden)
        self.infomax2 = Infomax(1, 1)

        self.loss_corr = 0

        self.conv_in = DeProp_Prop(in_channels, hidden_channels, self.lambda1, self.lambda2, self.gamma, args.orth)
        if self.with_bn:
            self.bn_in = (torch.nn.BatchNorm1d(hidden_channels))
            self.bns = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()

        for i in range(self.num_layers-2):
            self.convs.append(DeProp_Prop(hidden_channels, hidden_channels, self.lambda1, self.lambda2, self.gamma, args.orth))
            if self.with_bn:
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        self.conv_out = DeProp_Prop(hidden_channels, out_channels, self.lambda1, self.lambda2, self.gamma, args.orth)
        self.dropout = dropout
        self.smooth = args.smooth
        print(f"DeProp model")

    def reset_parameters(self):
        self.conv_in.reset_parameters()
        for i, conv in enumerate(self.convs):
            conv.reset_parameters()
        self.conv_out.reset_parameters()
        if self.with_bn:
            self.bn_in.reset_parameters()
            for bn in self.bns:
                bn.reset_parameters()

    def forward(self,  x, y, adj_t, test_true=False, report_metrics=False):
        x_1 = x
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv_in(x, adj_t)
        if self.with_bn:
            x = self.bn_in(x)
        if self.F_norm:
            x = F.normalize(x, p=2, dim=0)
        x = F.relu(x)
        if not isinstance(y, bool):
            y = y.unsqueeze(1).float()      # [2708] -> [2708, 1]

        for i, conv in enumerate(self.convs):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, adj_t)
            if self.with_bn:
                x = self.bns[i](x)
            if self.F_norm:
                x = F.normalize(x, p=2, dim=0)


            ####input x <-> hidden representation x 간의 infomax######
            if i!=self.num_layers-3 and i%5==4:
                self.loss_corr += 10*self.infomax.get_loss(x_1, x)

            ######My Method######
            if i!=self.num_layers-3 and i%2==1:
                _, f_i, f_j = get_random_dimension_pair(x)
                f_i = f_i.unsqueeze(1).float()  # [2708] -> [2708, 1]
                f_j = f_j.unsqueeze(1).float()  # [2708] -> [2708, 1]
                cc3 = correlation_coefficient(f_j, f_i)
                self.loss_corr += (self.infomax2.get_loss(f_i, y) + self.infomax2.get_loss(f_j, y) + cc3)/3
            
            if i == self.num_layers-4:
                corr = torch_corr(x.t())
                corr = torch.triu(corr, 1).abs()
                n = corr.shape[0]
                self.corr_2 = corr.sum().item() / (n *(n-1) /2)
                self.sim_2 = -1
                if report_metrics:
                    self.sim_2 = get_pairwise_sim(x)

            if i == self.num_layers-3:
                corr = torch_corr(x.t())
                corr = torch.triu(corr, 1).abs()
                n = corr.shape[0]
                self.corr = corr.sum().item() / (n *(n-1) /2)
                self.sim = -1
                if report_metrics:
                    self.sim = get_pairwise_sim(x)

            x = F.relu(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv_out(x, adj_t)

        return F.log_softmax(x, dim=1)



class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, args):
        super(GCN, self).__init__()

        self.num_layers = args.num_layers
        self.conv_in = GCNConv(in_channels, hidden_channels)
        self.convs = torch.nn.ModuleList()
        for i in range(self.num_layers-2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        self.conv_out = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout
        self.smooth = args.smooth
        print(f"GCN model")

    def reset_parameters(self):
        self.conv_in.reset_parameters()
        for i, conv in enumerate(self.convs):
            conv.reset_parameters()
        self.conv_out.reset_parameters()

    def forward(self, x, adj_t, train_adj=None, test_true=False):

        x, adj_t, train_adj= x, adj_t, train_adj
        x = self.conv_in(x, adj_t)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        for i, conv in enumerate(self.convs):
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv_out(x, adj_t)
        if self.smooth and test_true:
            corr = torch_corr(x.t())
            corr = torch.triu(corr, 1).abs()
            n = corr.shape[0]
            corr = corr.sum().item() / (n * (n - 1) / 2)
            sim = get_pairwise_sim(x)
            return F.log_softmax(x, dim=1), sim, corr

        return F.log_softmax(x, dim=1)


def get_model(args, dataset):
    data = dataset[0]

    # if args.model_type == "DeProp":
    model = DeProp(in_channels=args.num_feats,
                    hidden_channels=args.dim_hidden,
                    out_channels=args.num_classes,
                    dropout=args.dropout,
                    args=args
                    ).cuda()

    # elif args.model_type == "Ori":
    #     print(f"getmodel_GCN")
    #     model = GCN(in_channels=data.num_features,
    #                 hidden_channels=args.hidden_channels,
    #                 out_channels=dataset.num_classes,
    #                 dropout=args.dropout,
    #                 args=args,
    #                 ).cuda()

    # else:
    #     raise Exception(f'Wrong model_type')

    return model


