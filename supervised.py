#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 10:36:45 2021

@author: samuelheczko
"""

import netket as nk
import numpy as np
import matplotlib.pyplot as plt

def load_ed_data(L, J2):
    # Sigma^z*Sigma^z interactions
    sigmaz = np.array([[1, 0], [0, -1]])
    mszsz = (np.kron(sigmaz, sigmaz))

    # Exchange interactions
    exchange = np.asarray(
        [[0, 0, 0, 0], [0, 0, 2, 0], [0, 2, 0, 0], [0, 0, 0, 0]])

    # Couplings J1 and J2
    J = [1., J2]

    mats = []
    sites = []

    for i in range(L):
        for d in [0, 1]:
            # \sum_i J*sigma^z(i)*sigma^z(i+d)
            mats.append((J[d] * mszsz).tolist())
            sites.append([i, (i + d + 1) % L])

            # \sum_i J*(sigma^x(i)*sigma^x(i+d) + sigma^y(i)*sigma^y(i+d))
            mats.append(((-1.)**(d + 1) * J[d] * exchange).tolist())
            sites.append([i, (i + d + 1) % L])

    # 1D Lattice
    g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)

    # Spin based Hilbert Space
    hi = nk.hilbert.Spin(s=0.5, graph=g)

    # Custom Hamiltonian operator
    ha = nk.operator.LocalOperator(hi)
    for mat, site in zip(mats, sites):
        ha += nk.operator.LocalOperator(hi, mat, site)

    # Perform Lanczos Exact Diagonalization to get lowest three eigenvalues
    res = nk.exact.lanczos_ed(ha, first_n=3, compute_eigenvectors=True)

    # Eigenvector
    ttargets = []

    tsamples = []

    for i, visible in enumerate(hi.states()):
        # only pick zero-magnetization states
        mag = np.sum(visible)
        if(np.abs(mag) < 1.0e-4):
            tsamples.append(visible.tolist())
            ttargets.append([np.log(res.eigenvectors[0][i])])

    return hi, tsamples, ttargets


def main(number_of_atoms, j2, batch_s, number_of_iterations):

    L = number_of_atoms
    J2 = j2
    
    # Load the Hilbert space info and data
    hi, training_samples, training_targets = load_ed_data(L, J2)
    
    
    # Machine
    ma = nk.machine.RbmSpin(hilbert=hi, alpha=1)
    ma.init_random_parameters(seed=1234, sigma=0.01)
    # Optimizer
    #op = nk.optimizer.Sgd(learning_rate=0.01)
    op = nk.optimizer.AdaDelta()
    
    # Supervised learning object
    spvsd = nk.supervised.Supervised(
        machine=ma,
        optimizer=op,
        batch_size=batch_s,
        samples=training_samples,
        targets=training_targets)
    
    
    # Number of iteration
    n_iter = number_of_iterations
    
    # Run with "Overlap_phi" loss. Also available currently is "MSE, Overlap_uni"
    spvsd.run(n_iter=n_iter, loss_function="Overlap_phi",
              output_prefix='output', save_params_every=50)
    
    # Load the data from the .log file
    import json
    
    data=json.load(open("output.log"))
    
    # Extract the relevant information
    iters=[]
    log_overlap=[]
    mse=[]
    mse_log=[]
    
    data=json.load(open('output.log'))
    for iteration in data["Output"]:
        iters.append(iteration["Iteration"])
        log_overlap.append(iteration["log_overlap"])
        mse.append(iteration["mse"])
        mse_log.append(iteration["mse_log"])
    
    overlap = np.exp(-np.array(log_overlap))
    
    
    plt.subplot(2, 1, 1)
    plt.title(r'$J_1 J_2$ model, $J_2=' + str(J2) + '$')
    plt.ylabel('Overlap = F')
    plt.xlabel('Iteration #')
    
    plt.plot(iters, overlap)
    plt.axhline(y=1, xmin=0, xmax=iters[-1], linewidth=2, color='k',label='max accuracy = 1')
    
    plt.legend(frameon=False)
    
    plt.subplot(2, 1, 2)
    plt.ylabel('Overlap Error = 1-F')
    plt.xlabel('Iteration #')
    plt.semilogy(iters, 1.-overlap)
    plt.show()
    
main(12,0.4,50,1000)






