# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 12:33:18 2026

@author: Sourav
"""

import torch
import numpy as np
from Ensemble.Model_learning import EnsembleModel
from env import AdversarialPendulum
from Replay_buffer import ReplayBuffer
from ActorCritic import Actor_network,Critic_Network,Eta
from tqdm import tqdm
import pandas as pd


# def model_learning(em_obj,rb):
#     #find loss
#     s,a,a_bar,s_next,r,done = rb.get_batch(is_tensor=True)
#     loss = em_obj.train_step(s,a,a_bar,s_next)
    
#     mu,sigma = em_obj.predict(s,a,a_bar)
    
#     s_next_pred = em_obj.sample(s,a,a_bar)
    
device = "cuda" if torch.cuda.is_available() else "cpu"
num_models = 5
env = AdversarialPendulum()
hidden_ac = 128
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

s = env.reset()
gamma = 0.99
rb = ReplayBuffer()
em_obj = EnsembleModel(num_models, state_dim, action_dim)
Actor_ag = Actor_network(state_dim, hidden_ac, action_dim)
Actor_adv = Actor_network(state_dim, hidden_ac, action_dim)
done = False
eta_opt = Eta(state_dim,action_dim).to(device)
eta_pess = Eta(state_dim,action_dim).to(device)
Q_opt_rew = Critic_Network(state_dim,action_dim).to(device) 
Q_pess_rew = Critic_Network(state_dim,action_dim).to(device)
Q_opt_cost = Critic_Network(state_dim,action_dim).to(device) 
Q_pess_cost = Critic_Network(state_dim,action_dim).to(device)

#optimizers
opt_pi = torch.optim(Actor_ag.parameters(),lr=1e-3)
opt_bar = torch.optim(Actor_adv.parameters(),lr=1e-3)
opt_eta_opt = torch.optim(eta_opt.parameters(),lr=1e-2)
opt_eta_pess = torch.optim(eta_pess.parameters(),lr=1e-2)
opt_Q_opt_rew = torch.optim(Q_opt_rew.parameters(),lr=1e-3)
opt_Q_pess_rew = torch.optim(Q_pess_rew.parameters(),lr=1e-3)
opt_Q_opt_cost = torch.optim(Q_opt_cost.parameters(),lr=1e-3)
opt_Q_pess_cost = torch.optim(Q_pess_cost.parameters(),lr=1e-3)
Vr=[]

for t in range(10000):
    s = env.reset()
    while not done:
        a = Actor_ag(s)
        a_bar = Actor_adv(s)
        s_next,r,done,trunc,_ = env.step(a,a_bar)
        rb.add(s, a, a_bar, s_next, r, done)
    #model_learning(em_obj,rb)
    s_t,a_t,a_bar_t,s_next_t,r_t,done_t = rb.get_batch(is_tensor=True)
    Vr.append(r.mean().item())
    
    #training
    for _ in range(50):
        mean_tr_loss = em_obj.train_step(s_t, a_t, a_bar_t, s_next_t)
        
        mu,sigma = em_obj.predict(s_t,a_t,a_bar_t)
        
        #eta shiting or getting the \eta^{(o)} and \eta^{(p)} sampled state
        s_opt = s_t + mu + eta_opt(s_t,a_t,a_bar_t)*sigma
        s_pess = s_t + mu + eta_pess(s_t,a_t,a_bar_t)*sigma
        
        #To train the critics we need the target values
        target_opt_rew = r + gamma*Q_opt_rew(s_opt,Actor_ag(s_opt),Actor_adv(s_opt))
        target_pess_rew = r + gamma*Q_pess_rew(s_opt,Actor_ag(s_opt),Actor_adv(s_opt))
        
        loss_Q_opt_rew = ((Q_opt_rew(s_t,a_t,a_bar_t)-target_opt_rew.detach())**2).mean()
        loss_Q_pess_rew = ((Q_pess_rew(s_t,a_t,a_bar_t)-target_pess_rew.detach())**2).mean()
        #Update Critics
        opt_Q_opt_rew.zero_grad();loss_Q_opt_rew.backward();opt_Q_opt_rew.step();
        opt_Q_pess_rew.zero_grad();loss_Q_pess_rew.backward();opt_Q_pess_rew.step();
        
        #Update eta's
        loss_eta_opt = -Q_opt_rew(s_t,a_t,a_bar_t).mean()
        loss_eta_pess = Q_pess_rew(s_t,a_t,a_bar_t).mean()
        
        opt_eta_opt.zero_grad();loss_eta_opt.backwar();opt_eta_opt.step()
        opt_eta_pess.zero_grad();loss_eta_pess.backwar();opt_eta_pess.step()
        
        #Update agent actor or our protagonistic network
        loss_pi = -Q_opt_rew(s_t,Actor_ag(s_t),Actor_adv(s_t).detach()).mean()
        opt_pi.zero_grad();
        loss_pi.backward();
        opt_pi.step()
        
        #Update the adversary or the antagonist policy network
        loss_pi_bar = Q_opt_rew(s_t,Actor_ag(s_t).detach(),Actor_adv(s_t)).mean() + Q_pess_rew(s_t,Actor_ag(s_t).detach(),Actor_adv(s_t)).mean()
        opt_bar.zeros_grad();
        loss_pi_bar.backward()
        opt_bar.step()

df = {'vf':Vr}
df = pd.DataFrame(df)
df.to_excel('RHUCRL_rewards.xlsx')
        
        
        

    

    