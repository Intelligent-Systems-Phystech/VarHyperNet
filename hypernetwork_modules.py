import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class VarHyperNetwork(nn.Module):

    def __init__(self, f_size = 3, z_dim = 64, out_size=16, in_size=16, prior_sigma = 1.0, init_log_sigma=-3.0):
        super(HyperNetwork, self).__init__()
        self.z_dim = z_dim
        self.f_size = f_size
        self.out_size = out_size
        self.in_size = in_size
        self.prior_sigma = prior_sigma

        '''self.w1 = Parameter(torch.fmod(torch.randn((self.z_dim, self.out_size*self.f_size*self.f_size)).cuda(),2))
        self.b1 = Parameter(torch.fmod(torch.randn((self.out_size*self.f_size*self.f_size)).cuda(),2))

        self.w2 = Parameter(torch.fmod(torch.randn((self.z_dim, self.in_size*self.z_dim)).cuda(),2))
        self.b2 = Parameter(torch.fmod(torch.randn((self.in_size*self.z_dim)).cuda(),2))'''
        
                 
        self.mean = LinearApprNet((self.z_dim, self.out_size*self.f_size*self.f_size)) # параметры средних            
        self.log_sigma = LinearApprNet((self.z_dim, self.out_size*self.f_size*self.f_size))) # логарифм дисперсии
        self.mean_b = LinearApprNet( self.out_size*self.f_size*self.f_size) # то же самое для свободного коэффициента
        self.log_sigma_b = LinearApprNet( self.out_size*self.f_size*self.f_size)
        
        self.mean2 = LinearApprNet((self.z_dim, self.in_size*self.z_dim)) # параметры средних            
        self.log_sigma2 = LinearApprNet((self.z_dim, self.in_size*self.z_dim)) # логарифм дисперсии
        self.mean_b2 = LinearApprNet(self.in_size*self.z_dim) # то же самое для свободного коэффициента
        self.log_sigma_b2 = LinearApprNet( self.in_size*self.z_dim)
     
        self.log_sigma.const.data*= 0 # забьем константу нужными нам значениями
        self.log_sigma.const.data+= init_log_sigma
     
        self.log_sigma_b.const.data*= 0 # забьем константу нужными нам значениями
        self.log_sigma_b.const.data+= init_log_sigma    
        
        self.log_sigma2.const.data*= 0 # забьем константу нужными нам значениями
        self.log_sigma2.const.data+= init_log_sigma
     
        self.log_sigma_b2.const.data*= 0 # забьем константу нужными нам значениями
        self.log_sigma_b2.const.data+= init_log_sigma    
        
       
        

    def forward(self, z, l):
        
         if self.training: # во время обучения - сэмплируем из нормального распределения
            self.eps_w = t.distributions.Normal(self.mean(l), t.exp(self.log_sigma(l)))
            self.eps_b = t.distributions.Normal(self.mean_b(l), t.exp(self.log_sigma_b(l)))
            self.eps_w2 = t.distributions.Normal(self.mean(l), t.exp(self.log_sigma(l)))
            self.eps_b2 = t.distributions.Normal(self.mean_b(l), t.exp(self.log_sigma_b(l)))
        
            w = self.eps_w.rsample()
            b = self.eps_b.rsample()
            w2 = self.eps_w2.rsample()
            b2 = self.eps_b2.rsample()
            
        else:  # во время контроля - смотрим средние значения параметра        
            w = self.mean(l) 
            b = self.mean_b(l)
            w2 = self.mean(l) 
            b2 = self.mean_b(l)

        h_in = torch.matmul(z, w2) + b2
        h_in = h_in.view(self.in_size, self.z_dim)

        h_final = torch.matmul(h_in, w1) + b1
        kernel = h_final.view(self.out_size, self.in_size, self.f_size, self.f_size)

        return kernel
    
    def KLD(self, l):        
        # подсчет дивергенции
        size = self.in_size, self.out_size
        out = self.out_size
        device = self.mean.const.device
        self.eps_w = t.distributions.Normal(self.mean(l), t.exp(self.log_sigma(l)))
        self.eps_b = t.distributions.Normal(self.mean_b(l),  t.exp(self.log_sigma_b(l)))
        self.h_w = t.distributions.Normal(t.zeros(size, device=device), t.ones(size, device=device)*self.prior_sigma*l)
        self.h_b = t.distributions.Normal(t.zeros(out, device=device), t.ones(out, device=device)*self.prior_sigma*l)
        self.eps_w2 = t.distributions.Normal(self.mean2(l), t.exp(self.log_sigma2(l)))
        self.eps_b2 = t.distributions.Normal(self.mean_b2(l),  t.exp(self.log_sigma_b2(l)))
        k1 = t.distributions.kl_divergence(self.eps_w,self.h_w).sum()        
        k2 = t.distributions.kl_divergence(self.eps_b,self.h_b).sum()   
        k3 = t.distributions.kl_divergence(self.eps_w2,self.h_w).sum()        
        k4 = t.distributions.kl_divergence(self.eps_b2,self.h_b).sum()   
        return k1+k2+k3+k4

class LinearApprNet(nn.Module):
    def __init__(self, size,  init_const = 1.0, init_const2 = 1.0):    
        nn.Module.__init__(self)        
        if isinstance(size, tuple) and len(size) == 2:
            self.in_, self.out_ = size
            self.diagonal = False
        else:
            self.out_ = size
            self.diagonal = True
                           
        
        if self.diagonal:
            # независимая от параметра lambda часть
            self.const = nn.Parameter(t.randn(self.out_) * init_const) 
            self.const2 = nn.Parameter(t.ones(self.out_) * init_const2) 
            
            
        else:
            self.const = nn.Parameter(t.randn(self.in_, self.out_)) 
            t.nn.init.xavier_uniform(self.const,  init_const)
            self.const2 = nn.Parameter(t.randn(self.in_, self.out_)) 
            t.nn.init.xavier_uniform(self.const2,  init_const2)
            
            
    def forward(self, lam):        
        if self.diagonal:
            return self.const + self.const2 * lam
        else:
            return self.const + self.const2 * lam 
