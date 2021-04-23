#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 14:26:01 2021

@author: samuelheczko
"""
import netket as nk
import numpy as np
import matplotlib.pyplot as plt
a = 2
l=20
h = 1
j = 1
g = nk.graph.Hypercube(length=l, n_dim=1, pbc=True)
    
    # Hilbert space of spins on the graph
hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)
    
    # Ising spin hamiltonian
ha = nk.operator.Ising(h=h, hilbert=hi, graph=g, J = j)
    
    # RBM Spin Machine
ma = nk.machine.RbmSpin(alpha=a, hilbert=hi,use_visible_bias=False,use_hidden_bias=False)
    
ma.init_random_parameters(seed=1234, sigma=0.01)
Random_Params = ma.parameters.real
paramsReshaped = Random_Params.reshape(20,40)
u,k,vh = np.linalg.svd(paramsReshaped,full_matrices=False)
plt.plot(k)

