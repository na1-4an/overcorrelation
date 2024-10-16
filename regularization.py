import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import scipy.sparse as sp
import random

def loss_corr(x, nnodes=None):
    if nnodes is None:
        nnodes = x.shape[0]
    idx = np.random.choice(x.shape[0], int(np.sqrt(nnodes)))
    x = x[idx]
    x = x - x.mean(0)
    cov = x.t() @ x
    I_k = torch.eye(x.shape[1]).cuda() / np.sqrt(x.shape[1])
    loss = torch.norm(cov/torch.norm(cov) - I_k)
    return loss


def torch_corr(x):
    mean_x = torch.mean(x, 1)
    # xm = x.sub(mean_x.expand_as(x))
    xm = x - mean_x.view(-1, 1)
    c = xm.mm(xm.t())
    # c = c / (x.size(1) - 1)

    # normalize covariance matrix
    d = torch.diag(c)
    stddev = torch.pow(d, 0.5)
    c = c.div(stddev.expand_as(c))
    c = c.div(stddev.expand_as(c).t())

    # clamp between -1 and 1
    # probably not necessary but numpy does it
    c = torch.clamp(c, -1.0, 1.0)
    return c


def get_pairwise_sim(x):
    try:
        x = x.detach().cpu().numpy()
    except:
        pass

    if sp.issparse(x):
        x = x.todense()
        x = x / (np.sqrt(np.square(x).sum(1))).reshape(-1,1)
        x = sp.csr_matrix(x)
    else:
        x = x / (np.sqrt(np.square(x).sum(1))+1e-10).reshape(-1,1)
    # x = x / x.sum(1).reshape(-1,1)
    try:
        dis = euclidean_distances(x)
        return 0.5 * (dis.sum(1)/(dis.shape[1]-1)).mean()
    except:
        return -1

def get_random_dimension_pair(x):
    selected_columns = torch.randperm(x.shape[1])[:2]  # 무작위로 두 열 선택
    f_i, f_j = x[:, selected_columns[0]], x[:, selected_columns[1]]
    return selected_columns, f_i, f_j

def get_random_view_pair(x):
    # x의 열 인덱스를 섞기
    num_columns = x.shape[1]  # x의 열 개수
    
    # 홀수 개의 열일 경우 마지막 열을 제외하여 짝수로 만듦
    if num_columns % 2 != 0:
        num_columns -= 1  # 마지막 열 무시
    
    shuffled_indices = np.random.permutation(num_columns)  # 열을 섞은 인덱스 배열 생성
    
    # 열을 반으로 나눔
    half_point = num_columns // 2
    first_half_indices = shuffled_indices[:half_point]
    second_half_indices = shuffled_indices[half_point:]
    
    # 두 개의 2차원 배열 생성
    x_first_half = x[:, first_half_indices]  # 첫 번째 배열
    x_second_half = x[:, second_half_indices]  # 두 번째 배열
    
    return x_first_half, x_second_half

def info_nce_loss(f_i, f_j, temperature=0.5):
    """
    f_i: [batch_size, dim]
    f_j: [batch_size, dim]
    InfoNCE 손실을 계산하는 함수
    """
    # 양성 샘플 간의 유사도 (f_i와 f_j)
    pos_similarity = torch.sum(f_i * f_j, dim=-1) / temperature  # 코사인 유사도 기반 양성 샘플

    # 모든 샘플 (f_i, f_j 포함)과의 유사도 (음성 샘플 포함)
    batch_size = f_i.shape[0]
    
    # 음성 샘플과의 유사도 계산
    # f_i와 모든 f_j의 쌍을 고려한 음성 샘플들 간 유사도
    negative_similarity_i = torch.mm(f_i, f_j.t()) / temperature  # [batch_size, batch_size]
    negative_similarity_j = torch.mm(f_j, f_i.t()) / temperature  # [batch_size, batch_size]
    
    # 양성 유사도를 negative matrix에 추가
    logits = torch.cat([pos_similarity.unsqueeze(1), negative_similarity_i], dim=1)
    
    # 라벨 생성: 양성 샘플의 위치는 첫 번째 열이므로 0으로 설정
    labels = torch.zeros(batch_size, dtype=torch.long).to(f_i.device)
    
    # InfoNCE 손실 계산 (cross entropy)
    loss_i = F.cross_entropy(logits, labels)
    
    return loss_i