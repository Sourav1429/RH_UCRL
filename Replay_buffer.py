# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 14:18:18 2026

@author: gangu
"""

import torch
import numpy as np

class ReplayBuffer:
    def __init__(self,count):
        self.count=0
        self.s = []
        self.a = []
        self.a_bar = []
        self.s_next = []
        self.r = []
        self.done = []
    def add(self,s,a,a_bar,s_next,r,done,trunc=False):
        self.s.append(s)
        self.a.append(a)
        self.a_bar.append(a_bar)
        self.s_next.append(s_next)
        self.r.append(r)
        self.done.append(int(done and trunc))
        self.count+=1
    def get_batch(self,is_tensor=False):
        if(is_tensor):
            s_tensor = torch.Tensor(self.s)
            a_tensor = torch.Tensor(self.a)
            a_bar_tensor = torch.Tensor(self.a_bar)
            s_next_tensor = torch.Tensor(self.s_next)
            r_tensor = torch.Tensor(self.r)
            done_tensor = torch.Tensor(self.done)
            self.__init__()
            return s_tensor,a_tensor,a_bar_tensor,s_next_tensor,r_tensor,done_tensor
        else:
            return np.array(self.s),np.array(self.a),np.array(self.a_bar),np.array(self.s_next),np.array(self.r),np.array(self.done)
            