import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.transformer import GatedTransformerXL

class PPOTransformerModel(nn.Module):
    def __init__(self,config,state_size,action_size):
        super(PPOTransformerModel,self).__init__()
        self.fc = self._layer_init(nn.Linear(state_size,config['embed_dim']),std=np.sqrt(2))

        self.transformer = GatedTransformerXL(config,input_dim=config['embed_dim'])

        self.policy = nn.Sequential(
            nn.ReLU(),
            self._layer_init(nn.Linear(config['embed_dim'],config['hidden_size']),std=np.sqrt(2)),
            nn.ReLU(),
            self._layer_init(nn.Linear(config['hidden_size'],action_size),std=0.01)
        )

        self.value = nn.Sequential(
            nn.ReLU(),
            self._layer_init(nn.Linear(config['embed_dim'],config['hidden_size']),std=np.sqrt(2)),
            nn.ReLU(),
            self._layer_init(nn.Linear(config['hidden_size'],1),std=1)
        )

    @staticmethod
    def _layer_init(layer, std=np.sqrt(2), bias_const=0.0):
        """Init Weight and Bias with Constraint"""
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer
    
    def forward(self,state,memories, mask, memory_indices):
        
        out        = self.fc(state)
        out,memory = self.transformer(out,memories, mask, memory_indices)
        policy     = self.policy(out)
        value      = self.value(out)

        return policy,value,memory
    
    def get_policy(self,state,memories, mask, memory_indices):
        out    = self.fc(state)
        out,_  = self.transformer(out,memories, mask, memory_indices)
        policy = self.policy(out)
        return policy
    
