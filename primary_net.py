import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from hypernetwork_modules import HyperNetwork
from resnet_blocks import ResNetBlock


class Embedding(nn.Module):

    def __init__(self, z_num, z_dim, device = 'cpu', mode = 'mean', init_log_sigma = -4.0):
        super(Embedding, self).__init__()

        self.z_list = nn.ParameterList()
        self.z_num = z_num
        self.z_dim = z_dim
        self.init_log_sigma = init_log_sigma

        h,k = self.z_num

        if mode == 'mean':
            for i in range(h):
                for j in range(k):
                    self.z_list.append(Parameter(t.fmod(t.randn(self.z_dim).to(device), 4)))
        else:
            for i in range(h):
                for j in range(k):
                    self.z_list.append(Parameter(t.ones(self.z_dim).to(device)*self.init_log_sigma))

    def forward(self, hyper_net, lam = None):
        ww = []
        h, k = self.z_num
        for i in range(h):
            w = []
            for j in range(k):
                w.append(hyper_net(self.z_list[i*k + j], lam))
            ww.append(t.cat(w, dim=1))
        return t.cat(ww, dim=0)
           
    


class PrimaryNetwork(nn.Module):

    def __init__(self, z_dim=64, device = 'cpu', prior_sigma = 1.0):
        super(PrimaryNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.z_dim = z_dim
        self.hope1 = HyperNetwork(z_dim=self.z_dim)
        self.hope2 = HyperNetwork(z_dim=self.z_dim)
        self.prior_sigma = prior_sigma
        self.device = device

        self.zs_size = [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1],
                        [2, 1], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2],
                        [4, 2], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4]]

        self.filter_size = [[16,16], [16,16], [16,16], [16,16], [16,16], [16,16], [16,32], [32,32], [32,32], [32,32],
                            [32,32], [32,32], [32,64], [64,64], [64,64], [64,64], [64,64], [64,64]]

        self.res_net = nn.ModuleList()

        for i in range(18):
            down_sample = False
            if i > 5 and i % 6 == 0:
                down_sample = True
            self.res_net.append(ResNetBlock(self.filter_size[i][0], self.filter_size[i][1], downsample=down_sample))

        self.zs_mean = nn.ModuleList()
        self.zs_sigma = nn.ModuleList()

        for i in range(36):
            self.zs_mean.append(Embedding(self.zs_size[i], self.z_dim, self.device, 'mean'))
            self.zs_sigma.append(Embedding(self.zs_size[i], self.z_dim, self.device, 'sigma'))

        self.global_avg = nn.AvgPool2d(8)
        self.final = nn.Linear(64,10)

    def forward(self, x, lam = None):

        x = F.relu(self.bn1(self.conv1(x)))
        for i in range(18):
            # if i != 15 and i != 17:
            w1_mean = self.zs_mean[2*i](self.hope1, lam)
            w2_mean = self.zs_mean[2*i+1](self.hope1, lam)
            w1_sigma = self.zs_sigma[2*i](self.hope2, lam)
            w2_sigma = self.zs_sigma[2*i+1](self.hope2, lam)
            if self.training:
                self.w1_eps = t.distributions.Normal(w1_mean, t.exp(w1_sigma+0.2))
                self.w2_eps = t.distributions.Normal(w2_mean, t.exp(w2_sigma+0.2))
                w1 = self.w1_eps.rsample()
                w2 = self.w2_eps.rsample()
            else:
                w1 = w1_mean
                w2 = w2_mean
            x = self.res_net[i](x, w1, w2)

        x = self.global_avg(x)
        x = self.final(x.view(-1,64))

        return x
    
    def KLD(self, l):
        # подсчет дивергенции
        device = self.device
        k = 0
        for i in range(18):
            # if i != 15 and i != 17:
            w1_mean = self.zs_mean[2*i](self.hope1, l)
            w2_mean = self.zs_mean[2*i+1](self.hope1, l)
            w1_sigma = self.zs_sigma[2*i](self.hope2, l)
            w2_sigma = self.zs_sigma[2*i+1](self.hope2, l)
            self.h_w1 = t.distributions.Normal(t.zeros_like((w1_mean), device=device),
                                               t.ones_like(t.exp(w1_sigma+0.2), device=device)*self.prior_sigma)
            self.h_w2 = t.distributions.Normal(t.zeros_like((w2_mean), device=device),
                                               t.ones_like(t.exp(w2_sigma+0.2), device=device)*self.prior_sigma)
            self.w1_eps = t.distributions.Normal(w1_mean, t.exp(w1_sigma+0.2))
            self.w2_eps = t.distributions.Normal(w2_mean, t.exp(w2_sigma+0.2))
            k += t.distributions.kl_divergence(self.w1_eps, self.h_w1).sum()
            k += t.distributions.kl_divergence(self.w2_eps, self.h_w2).sum()
        return k