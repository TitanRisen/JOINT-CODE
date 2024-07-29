# -*- coding: utf-8 -*-
"""
@author: 

"""
import torch

def pairwise_distances(x, y, power=2, sum_dim=2):
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    x = x.unsqueeze(1).expand(n,m,d)
    y = y.unsqueeze(0).expand(n,m,d)
    dist = torch.pow(x-y, power).sum(sum_dim)
    return dist

def StandardScaler(x,with_std=False):
    mean = x.mean(0, keepdim=True)
    std = x.std(0, unbiased=False, keepdim=True)
    x -= mean
    if with_std:
        x /= (std + 1e-10)
    return x



def JSdiv_loss(Xs,ys,Xt,yt0, DEVICE, lamda=1e-2, sigma=None,truncated_param=1e-8,sigma_y=None,task='classification', alpha=0.5, beta=0.5):
    if alpha > 1.0 or alpha < 0.0 or beta > 1.0 or beta < 0.0:
        raise ValueError("0<=alpha<=1, 0<=beta<=1") 
    if alpha > 0.9 and beta > 0:
        return KL_loss(Xs,ys,Xt,yt0, DEVICE, lamda, sigma, truncated_param, beta)
    ns,dim = Xs.shape
    nt = Xt.shape[0]
    X = StandardScaler(torch.cat((Xs,Xt), dim=0))

    if sigma is None:
        pairwise_dist = pairwise_distances(Xs,Xt)
        sigma = torch.median(pairwise_dist[pairwise_dist!=0]).to(DEVICE)
    if task == 'regression' and sigma_y is None:
        pairwise_dist = pairwise_distances(ys,yt0)
        sigma_y = torch.median(pairwise_dist[pairwise_dist!=0]).to(DEVICE)
    if task == 'classification':
        source_label = ys
        target_label = yt0
        n = ns + nt
        y = torch.cat((source_label, target_label), dim = 0)
        Ky = torch.tensor(y[:,None]==y, dtype=torch.float64).to(DEVICE)
    else:
        return 0
    X_norm = torch.sum(X ** 2, axis=-1).to(DEVICE)
    K_W = torch.exp(-( X_norm[:, None] + X_norm[None,:] - 2 * torch.mm(X, X.T)) / sigma) * Ky

    H = 1.0 / nt * torch.mm(K_W[ns:].T,K_W[ns:])
    theta = torch.mm((H + lamda * torch.eye(ns+nt).to(DEVICE)).inverse(), (torch.mean(K_W[:ns],axis=0)[:,None]))
    PS_over_PT = torch.clamp(torch.mm(K_W[ns:],theta), min=truncated_param)
    H_2 = 1.0 / ns * torch.mm(K_W[:ns].T,K_W[:ns])
    theta_2 = torch.mm((H_2 + lamda * torch.eye(ns+nt).to(DEVICE)).inverse(), (torch.mean(K_W[ns:],axis=0)[:,None]))
    PT_over_PS = torch.clamp(torch.mm(K_W[:ns],theta_2), min=truncated_param)

    js_div = -alpha * torch.mean(torch.log(beta + PT_over_PS  * (1.0-beta) )) - (1.0-alpha) * torch.mean(torch.log( (1.0-beta) + PS_over_PT * beta))
    return js_div


def KL_loss(Xs,ys,Xt,yt0, DEVICE, lamda=1e-2, sigma=None,truncated_param=1e-10,alpha=0.5):
    if alpha > 1.0 or alpha < 0.0:
        raise ValueError("alpha domain: 0.0<=alpha<=1.0")
    ns,dim = Xs.shape
    nt = Xt.shape[0]
    X = StandardScaler(torch.cat((Xs,Xt), dim=0))
    if sigma is None:
        pairwise_dist = pairwise_distances(Xs,Xt)
        sigma = torch.median(pairwise_dist[pairwise_dist!=0]).to(DEVICE)
    source_label = ys
    target_label = yt0
    n = ns + nt
    y = torch.cat((source_label, target_label), dim = 0)
    Ky = torch.tensor(y[:,None]==y, dtype=torch.float64).to(DEVICE)
    X_norm = torch.sum(X ** 2, axis=-1).to(DEVICE)
    K_W = torch.exp(-( X_norm[:, None] + X_norm[None,:] - 2 * torch.mm(X, X.T)) / sigma) * Ky
    H = float(alpha)/ns * torch.mm(K_W[:ns].T,K_W[:ns]) + float(1.0-alpha)/nt * torch.mm(K_W[ns:].T,K_W[ns:])
    b = (torch.mean(K_W[:ns],axis=0)[:,None])
    theta = torch.mm((H + lamda * torch.eye(ns+nt).to(DEVICE)).inverse(), b)
    PS_over_PM = torch.clamp( torch.mm(K_W[:ns],theta)  , min=truncated_param)
    D = torch.mean(torch.log(PS_over_PM))
    return D