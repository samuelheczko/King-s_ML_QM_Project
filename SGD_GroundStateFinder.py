#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 10:11:52 2021

@author: samuelheczko
"""
import netket as nk
    # Helper libraries
import numpy as np
import matplotlib.pyplot as plt
def main(noi): ## bonding1 = bonding to the nearest neighbour, bonding2 second nearest
    #Couplings J1 and J2
    
    L = 14
    
    # Define custom graph
    edge_colors = []
    for i in range(L):
        edge_colors.append([i, (i+1)%L, 1])
        edge_colors.append([i, (i+2)%L, 2])
    
    # Define the netket graph object
    g = nk.graph.CustomGraph(edge_colors)
    
    #Sigma^z*Sigma^z interactions
    sigmaz = [[1, 0], [0, -1]]
    mszsz = (np.kron(sigmaz, sigmaz))
    
    #Exchange interactions
    exchange = np.asarray([[0, 0, 0, 0], [0, 0, 2, 0], [0, 2, 0, 0], [0, 0, 0, 0]])
    
    bond_operator = [
        (J[0] * mszsz).tolist(),
        (J[1] * mszsz).tolist(),
        (-J[0] * exchange).tolist(),
        (J[1] * exchange).tolist(),
    ]
    
    bond_color = [1, 2, 1, 2]
    
    # Spin based Hilbert Space
    hi = nk.hilbert.Spin(s=0.5, total_sz=0.0, graph=g)
    global sf
    global op
    
    # Custom Hamiltonian operator
    op = nk.operator.GraphOperator(hi, bondops=bond_operator, bondops_colors=bond_color)
    
    # Define the fully-connected FFNN
    layers = (nk.layer.FullyConnected(input_size=L,output_size=2*L,use_bias=True),
              nk.layer.Lncosh(input_size=2*L),
              nk.layer.SumOutput(input_size=2*L))
    for layer in layers:
        layer.init_random_parameters(seed=1234, sigma=0.01)
    
    ffnn = nk.machine.FFNN(hi, layers)
    
    # We shall use an exchange Sampler which preserves the global magnetization (as this is a conserved quantity in the model)
    sa = nk.sampler.MetropolisExchange(machine=ffnn, d_max = 2)
    
    # We choose a basic, albeit important, Optimizer: the Stochastic Gradient Descent
    opt = nk.optimizer.Sgd(learning_rate=0.01)
    
    # We can then specify a Variational Monte Carlo object, using the Hamiltonian, sampler and optimizers chosen.
    # Note that we also specify the method to learn the parameters of the wave-function: here we choose the efficient
    # Stochastic reconfiguration (Sr), here in an iterative setup
    gs = nk.variational.Vmc(hamiltonian=op,
                            sampler=sa,
                            optimizer=opt,
                            n_samples=1000,
                            use_iterative=True,
                            method='Sr')
    
    
    # We need to specify the local operators as a matrix acting on a local Hilbert space
    sf = []
    sites = []
    for i in range(0, L):
        for j in range(0, L):
            sf.append(((-1)**(i-j)*mszsz/L).tolist())
            sites.append([i,j])
    global structure_factor
    structure_factor = nk.operator.LocalOperator(hi, sf, sites)
    
    # Add the observable to the VMC object
    gs.add_observable(structure_factor, "Structure Factor")
    # Run the optimization protocol
    gs.run(output_prefix='test', n_iter=noi)
    
    
    
def plot(bonding1,bonding2):
     # Load the data from the .log file
    import json
    data=json.load(open("test.log"))
    
    # Extract the relevant information
    
    iters=[]
    energy=[]
    sf=[]
    
    for iteration in data["Output"]:
        iters.append(iteration["Iteration"])
        energy.append(iteration["Energy"]["Mean"])
        sf.append(iteration["Structure Factor"]["Mean"])
    
    fig, ax1 = plt.subplots()
    fig.suptitle('J1= '+str(bonding1)+', J2 = '+str(bonding2))
    ax1.plot(iters, energy, color='blue', label='Energy')
    ax1.set_ylabel('Energy')
    ax1.set_xlabel('Iteration')
    ax2 = ax1.twinx()
    ax2.plot(iters, np.array(sf), color='green', label='Structure Factor')
    ax2.set_ylabel('Structure Factor')
    ax1.legend(loc=2)
    ax2.legend(loc=1)
    plt.show()
    print(r"Structure factor on Sgd = {0:.3f}({1:.3f})".format(np.mean(sf[-50:]),
                                              np.std(np.array(sf[-50:]))/np.sqrt(50)))
    print(r"Energy on Sgd = {0:.3f}({1:.3f})".format(np.mean(energy[-50:]), np.std(energy[-50:])/(np.sqrt(50))))
    res = nk.exact.lanczos_ed(op, first_n=1, compute_eigenvectors=True)
    print("Exact Ground-state Structure Factor: {0:.3f}".format(np.real(res.mean(structure_factor, 0))))
    print("Exact ground state energy = {0:.3f}".format(res.eigenvalues[0]))


J = [1, 0.2] #set the bonding strength J0 =nearest neighbour J1 = one next to it
main(100) #takes in the number of iterations
plot(J[0],J[1])




