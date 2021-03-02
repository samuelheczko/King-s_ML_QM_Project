#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 22:43:36 2021

@author: samuelheczko
"""
# Import netket library
import netket as nk

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
def ising(h1,L1):
    h = h1
    L = L1
    
    edges = []
    for i in range(L):
        edges.append([i, (i+1)%L])
    
    g = nk.graph.CustomGraph(edges)
    # Spin based Hilbert Space
    hi = nk.hilbert.Spin(s=0.5, graph=g)
    # Pauli Matrices
    sigmaz = np.array([[1, 0], [0, -1]])
    sigmax = np.array([[0, 1], [1, 0]])
    
    operators = []
    sites = []
    
    # Local Field term
    for i in range(L):
        operators.append((h*sigmax).tolist())
        sites.append([i])
    
    # Ising iteraction
    for i in range(L):
        operators.append((-np.kron(sigmaz,sigmaz)).tolist())
        sites.append([i, (i+1)%L])
    
    op = nk.operator.LocalOperator(hi, operators=operators, acting_on=sites)
    
    res = nk.exact.lanczos_ed(op, first_n=1, compute_eigenvectors=False)
    eigenvalue = res.eigenvalues[0]
    return eigenvalue
 

es = []
ks = []       
for s in range(10):
    k = 0.9 + s*0.0001
    ks.append(k)
    es.append(ising(k,8))

plt.plot(ks,es)
plt.show

    
    
    
    
    
    