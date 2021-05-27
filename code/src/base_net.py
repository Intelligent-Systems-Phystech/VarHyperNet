import torch as t 
import torch.nn as nn
import torch.nn.functional as F
from torch.linalg import norm
from linear_var_hypernet import LinearApprNet
from kernel_hypernet import KernelNet



class BaseLayer(nn.Module): #однослойная сеть
    def __init__(self, in_,  out_, device,  act=F.relu, prior_sigma=1.0):         
        nn.Module.__init__(self)                    
        self.mean = nn.Parameter(t.randn(in_, out_, device=device)) # параметры средних
        t.nn.init.xavier_uniform(self.mean) 
        self.mean_b = nn.Parameter(t.randn(out_, device=device)) # то же самое для свободного коэффициента
                
        self.in_ = in_
        self.out_ = out_
        self.act = act
        
        self.prior_sigma = prior_sigma
        
        
    def forward(self,x):    
        w = self.mean 
        b = self.mean_b
            
        # функция активации 
        return self.act(t.matmul(x, w)+b)

    def KLD(self, l = None, prior_scale=1.0):        
        # подсчет hyperloss
        return ((self.mean**2).sum() + (self.mean_b**2).sum())/self.prior_sigma/prior_scale
    
class BaseLayerLinear(nn.Module): #однослойная сеть
    def __init__(self, in_,  out_,   act=F.relu, prior_sigma=1.0):         
        nn.Module.__init__(self)                    
        self.mean = LinearApprNet((in_, out_)) # параметры средних 
        self.mean_b = LinearApprNet( out_) # то же самое для свободного коэффициента
                    
        self.in_ = in_
        self.out_ = out_
        self.act = act
        
        self.prior_sigma = prior_sigma
        
        
    def forward(self,x, l):    
        w = self.mean(l) 
        b = self.mean_b(l)
            
        # функция активации 
        return self.act(t.matmul(x, w)+b)

    def KLD(self, l, prior_scale=1.0):
        
        # подсчет hyperloss
        return  ((self.mean(l)**2).sum() + (self.mean_b(l)**2).sum())/self.prior_sigma/prior_scale
    

class BaseKernelLayer(nn.Module):  # вариационная однослойная сеть
    def __init__(self, in_,  out_,  kernel_num,  prior_sigma = 1.0,  act=F.relu):
        nn.Module.__init__(self)
        self.mean = KernelNet((in_, out_), kernel_num)  # параметры средних
        self.mean_b = KernelNet(out_, kernel_num)
      
        
        self.in_ = in_
        self.out_ = out_
        self.act = act
        
        self.prior_sigma = prior_sigma

    def forward(self, x, l):
        
        w = self.mean(l)
        b = self.mean_b(l)
       
        # функция активации
        return self.act(t.matmul(x, w)+b)

    def KLD(self, l, prior_scale=1.0):
        # подсчет hyperloss
        return  ((self.mean(l)**2).sum() + (self.mean_b(l)**2).sum())/self.prior_sigma/prior_scale
    
    