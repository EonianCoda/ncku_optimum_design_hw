from .shampoo import Shampoo
import torch.optim as optim

def get_optimizer(model, 
                  opt_name: str, 
                  lr: float, 
                  **kwargs):
    if opt_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, **kwargs)
    elif opt_name == 'shampoo':
        optimizer = Shampoo(model.parameters(), lr=lr, **kwargs)
        
    return optimizer
        

