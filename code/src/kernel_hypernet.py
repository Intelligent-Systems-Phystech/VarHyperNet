import torch as t
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class KernelNet(nn.Module):
    def __init__(self, size, kernel_num, init_sigma = 0.001, init_pivot_sigma = 4.0, init_r = 1.0):    
        nn.Module.__init__(self)
        
        if not isinstance(size, tuple): # check if size is 1d
            size = (size,)
            
        self.size = size
        
        self.kernel_num = kernel_num
        
        self.const = nn.Parameter(t.randn(*size) * init_sigma)
        
        
        if len(size)>1:
        
            self.kernels_a = nn.Linear(kernel_num, kernel_num*size[0], bias=False)

        
            self.kernels_b = nn.Linear(kernel_num, kernel_num*size[1], bias=False)
        else:
            self.kernels_a = nn.Linear(kernel_num, size[0], bias=False)
        
        
        
        self.r = nn.Parameter(t.ones(kernel_num)*init_r)
        self.pivots = nn.Parameter(t.rand(kernel_num)*init_pivot_sigma*2 - init_pivot_sigma)
          
            
    def forward(self, lam):
        dist = t.exp(-0.5*self.r * (lam - self.pivots)**2)
        dist = dist / dist.sum()
        if len(self.size)>1:
            a = self.kernels_a(dist).view(-1, self.kernel_num)
            b = self.kernels_b(dist).view(self.kernel_num, -1)
            return self.const + t.matmul(a, b)
        else:
            return self.const + self.kernels_a(dist) 
       
    
class VarKernelLayer(nn.Module):  # вариационная однослойная сеть
    def __init__(self, in_,  out_,  kernel_num, prior_sigma=1.0, init_log_sigma=-3.0,  act=F.relu):
        nn.Module.__init__(self)
        self.mean = KernelNet((in_, out_), kernel_num)  # параметры средних
        self.log_sigma = KernelNet(
            (in_, out_), kernel_num)  # логарифм дисперсии
        # то же самое для свободного коэффициента
        self.mean_b = KernelNet(out_, kernel_num)
        self.log_sigma_b = KernelNet(out_, kernel_num)
    
        self.log_sigma.const.data *= 0
        self.log_sigma.const.data += init_log_sigma
        
        self.log_sigma_b.const.data *= 0
        self.log_sigma_b.const.data += init_log_sigma
        
        
        self.log_sigma.kernels_a.weight.data *= 0
        self.log_sigma.kernels_a.weight.data += t.randn(self.log_sigma.kernels_a.weight.data.shape)*0.001
        
        self.log_sigma.kernels_b.weight.data *= 0
        self.log_sigma.kernels_b.weight.data += t.randn(self.log_sigma.kernels_b.weight.data.shape)*0.001
        
        self.log_sigma_b.kernels_a.weight.data *= 0
        self.log_sigma_b.kernels_a.weight.data += t.randn(self.log_sigma_b.kernels_a.weight.data.shape)*0.001
        
        
        self.in_ = in_
        self.out_ = out_
        self.act = act
        self.prior_sigma = prior_sigma

    def forward(self, x, l):
        if self.training:  # во время обучения - сэмплируем из нормального распределения
            self.eps_w = t.distributions.Normal(
                self.mean(l), t.exp(self.log_sigma(l)))
            self.eps_b = t.distributions.Normal(
                self.mean_b(l), t.exp(self.log_sigma_b(l)))

            w = self.eps_w.rsample()
            b = self.eps_b.rsample()

        else:  # во время контроля - смотрим средние значения параметра
            w = self.mean(l)
            b = self.mean_b(l)
       
        # функция активации
        return self.act(t.matmul(x, w)+b)

    def KLD(self, l, prior_scale=1.0):
        # подсчет дивергенции
        size = self.in_, self.out_
        out = self.out_
        device = self.mean.const.device
        self.eps_w = t.distributions.Normal(
            self.mean(l),t.exp(self.log_sigma(l)))
        self.eps_b = t.distributions.Normal(
            self.mean_b(l), t.exp(self.log_sigma_b(l)))
        self.h_w = t.distributions.Normal(
            t.zeros(size, device=device), t.ones(size, device=device)*self.prior_sigma * prior_scale)
        self.h_b = t.distributions.Normal(
            t.zeros(out, device=device), t.ones(out, device=device)*self.prior_sigma * prior_scale)
        k1 = t.distributions.kl_divergence(self.eps_w, self.h_w).sum()
        k2 = t.distributions.kl_divergence(self.eps_b, self.h_b).sum()
        return k1+k2

    
