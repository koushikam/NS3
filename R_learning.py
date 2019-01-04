#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 13:10:56 2018

@author: Koushik A Manjunatha
"""
import numpy as np
from random import randint
from numpy import random,argmax
from math import exp
from numpy import random
from itertools import combinations,permutations
import pandas as pd
import matplotlib.pyplot as plt


''' RL based station mapping '''
class R_learning():
    '''
    nLabels: total number of station classes 
    nAPs: number of agents in the learning 
    
    '''
    def __init__(self,area,nAPs,nLabels,loc_AP):
        print('Reinforcement learning based station mapping ');
        self.nAPs = nAPs;
        self.nLabels = nLabels;
        self.loc =  loc_AP;
        self.loc =  self.loc[0:nAPs];
        self.area = area;
        self.nstates =  10;        
        
    def action_set(self):
        ''' generating all possible combinations of station leaving and connecting '''
        a =  permutations(np.arange(0,self.nLabels,1),2) 
        m =  list(a)
        for i in range(self.nLabels):
            m.append((i,i))
        actions_i =dict(enumerate(m));
        return actions_i
    
    def get_action(self):
        actions = self.action_set()
        n_actions =  len(actions)
        a = permutations(np.arange(0,n_actions,1),2)
        m=list(a)
        for i in range(n_actions):
            m.append((i,i));
        actions = dict(enumerate(m));
        return actions;
         
    def updateQ(self,Q,s,a,r,n_s):
        gamma=0.6;
        alpha=0.3;
        ''' Update Qvalues by QLearning '''
        maxQ = max(Q[n_s])
        Q[s][a] = Q[s][a] + alpha * ( r + gamma*maxQ - Q[s][a] )
        return Q
    
    def e_greedy_selection(self,Q,c_s,num_actions):
        #global epsilon
        epsilon = 0.5;
        #selects an action using Epsilon-greedy strategy
        # Q: the Qtable
        # s: the current state               
        if (random.rand()> epsilon):
            a = argmax(Q[c_s]) # GetBestAction(c_s)    
        else:
            # selects a random action based on a uniform distribution
            a = randint(0,num_actions-1)
        return a
    
    def learn(self,Q,c_s,n_s,r,num_actions):
        a  = self.e_greedy_selection(Q,c_s,num_actions)
        # convert the index of the action into an action value
        #Update the Qtable, that is,  learn from the experience
        Q= self.updateQ(Q,c_s,a,r,n_s)
        return Q,a    
        
        