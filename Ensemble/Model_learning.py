# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 19:47:06 2026

@author: Sourav
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DynamicsModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        input_dim = state_dim + action_dim * 2  # a and a_bar

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self.mu_head = nn.Linear(128, state_dim)
        self.logvar_head = nn.Linear(128, state_dim)

    def forward(self, s, a, a_bar):
        x = torch.cat([s, a, a_bar], dim=-1)
        h = self.net(x)

        mu = self.mu_head(h)
        logvar = self.logvar_head(h)

        logvar = torch.clamp(logvar, -5, 2)  # stability

        return mu, logvar

class EnsembleModel:
    def __init__(self, num_models, state_dim, action_dim, lr=1e-3):
        self.models = [
            DynamicsModel(state_dim, action_dim) for _ in range(num_models)
        ]

        self.optimizers = [
            torch.optim.Adam(m.parameters(), lr=lr) for m in self.models
        ]

        self.num_models = num_models
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def predict(self, s, a, a_bar):
        mus, vars_ = [], []

        for model in self.models:
            mu, logvar = model(s, a, a_bar)
            mus.append(mu)
            vars_.append(torch.exp(logvar))

        mu_stack = torch.stack(mus)   # (N, B, state_dim)
        var_stack = torch.stack(vars_)

        # mean
        mu = mu_stack.mean(dim=0)

        # epistemic uncertainty
        epistemic = mu_stack.var(dim=0)

        # aleatoric uncertainty
        aleatoric = var_stack.mean(dim=0)

        sigma = torch.sqrt(epistemic + aleatoric + 1e-6)

        return mu, sigma

    def sample(self, s, a, a_bar):
        mu, sigma = self.predict(s, a, a_bar)

        epsilon = torch.randn_like(mu)

        delta = mu + epsilon * sigma
        s_next = s + delta

        return s_next
    
    def train_step(self, s, a, a_bar, s_next):
        s = s.to(self.device)
        a = a.to(self.device)
        a_bar = a_bar.to(self.device)
        s_next = s_next.to(self.device)
    
        delta = s_next - s
        batch_size = s.shape[0]
    
        losses = []
    
        for model, opt in zip(self.models, self.optimizers):
    
            idx = torch.randint(0, batch_size, (batch_size,), device=self.device)
    
            s_i = s[idx]
            a_i = a[idx]
            a_bar_i = a_bar[idx]
            delta_i = delta[idx]
    
            mu, logvar = model(s_i, a_i, a_bar_i)
    
            inv_var = torch.exp(-logvar)
            loss = ((mu - delta_i) ** 2 * inv_var + logvar).mean()
    
            opt.zero_grad()
            loss.backward()
            opt.step()
    
            losses.append(loss.item())
    
        return sum(losses) / len(losses)