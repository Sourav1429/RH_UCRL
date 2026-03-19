# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 18:08:01 2026

@author: Sourav
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor_network(nn.Module):
    def __init__(self, state_dim, hidden,action_dim):
        super().__init__()
        self.layer1 = nn.Linear(state_dim,hidden)
        self.layer2 = nn.Linear(hidden, action_dim)
        self.relu = nn.ReLU()
    def forward(self,x):
        return self.layer2(self.relu(self.layer1(x)))
    
class Critic_Network(nn.Module):
    def __init__(self, state_dim, hidden):
        super().__init__()
        self.layer1 = nn.Linear(state_dim,hidden)
        self.layer2 = nn.Linear(hidden,1)
        self.relu = nn.ReLU()
    def forward(self,x):
        return self.layer2(self.relu(self.layer1(x)))

class Eta(nn.Module): 
    def __init__(self, s_dim, a_dim): 
        super().__init__() 
        self.net = Actor_network(s_dim + a_dim + s_dim, s_dim) 
    def forward(self, s, a, a_bar): 
        x = torch.cat([s, a, a_bar], dim=-1) 
        return torch.tanh(self.net(x))