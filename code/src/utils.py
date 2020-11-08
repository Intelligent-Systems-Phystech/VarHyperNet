import numpy as np 
import torch as t 

# будем удалять по 10% от модели и смотреть качество
def delete_10(net, device, callback):
    acc_delete = []
    mu = net[0].mean
    sigma = t.exp(2*net[0].log_sigma)
    prune_coef = (mu**2/sigma).cpu().detach().numpy()    
    sorted_coefs = np.sort(prune_coef.flatten())
    mu2 = net[1].mean
    sigma2 = t.exp(2*net[1].log_sigma)
    prune_coef2 = (mu2**2/sigma2).cpu().detach().numpy()    
    sorted_coefs2 = np.sort(prune_coef2.flatten())
    
    
    for j in range(10):
        
        ids = (prune_coef <= sorted_coefs[round(j/10*len(sorted_coefs))]) 
        net[0].mean.data*=(1-t.tensor(ids*1.0, device=device, dtype=t.float))
        
        ids2 = (prune_coef2 <= sorted_coefs2[round(j/10*len(sorted_coefs2))]) 
        net[1].mean.data*=(1-t.tensor(ids2*1.0, device=device, dtype=t.float))
        
        
        acc_delete.append(callback())
    return acc_delete    


def net_copy(net, new_net, lam):    
    for j in range(0, 2): # бежим по слоям        
        new_net[j].mean.data*=0
        new_net[j].mean.data+=net[j].mean(lam)
        new_net[j].mean_b.data*=0
        new_net[j].mean_b.data+=net[j].mean_b(lam)
        new_net[j].log_sigma.data*=0
        new_net[j].log_sigma.data+=net[j].log_sigma(lam)
        new_net[j].log_sigma_b.data*=0
        new_net[j].log_sigma_b.data+=net[j].log_sigma_b(lam)
    