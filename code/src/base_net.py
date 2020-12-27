import torch as t 
import torch.nn as nn
import torch.nn.functional as F
from torch.linalg import norm
from .linear_var_hypernet import LinearApprNet

class BaseLayer(nn.Module): #однослойная сеть
    def __init__(self, in_,  out_, device,  act=F.relu):         
        nn.Module.__init__(self)                    
        self.mean = nn.Parameter(t.randn(in_, out_, device=device)) # параметры средних
        t.nn.init.xavier_uniform(self.mean) 
        self.mean_b = nn.Parameter(t.randn(out_, device=device)) # то же самое для свободного коэффициента
                
        self.in_ = in_
        self.out_ = out_
        self.act = act
        
    def forward(self,x):    
        w = self.mean 
        b = self.mean_b
            
        # функция активации 
        return self.act(t.matmul(x, w)+b)

    def KLD(self, l = None):        
        # подсчет hyperloss
        return (self.mean**2).sum() + (self.mean_b**2).sum()
    
class BaseLayerLinear(nn.Module): #однослойная сеть
    def __init__(self, in_,  out_,   act=F.relu):         
        nn.Module.__init__(self)                    
        self.mean = LinearApprNet((in_, out_)) # параметры средних 
        self.mean_b = LinearApprNet( out_) # то же самое для свободного коэффициента
                    
        self.in_ = in_
        self.out_ = out_
        self.act = act
        
    def forward(self,x, l):    
        w = self.mean(l) 
        b = self.mean_b(l)
            
        # функция активации 
        return self.act(t.matmul(x, w)+b)

    def KLD(self, l):        
        # подсчет hyperloss
        return  (self.mean(l)**2).sum() + (self.mean_b(l)**2).sum() 
    