#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 22:04:31 2021

@author: samuelheczko
"""

# Copyright 2018 The Simons Foundation, Inc. - All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import netket as nk
import matplotlib as m
import matplotlib.pyplot as plt
import numpy as np

cdict = {
  'red'  :  ( (0.0, 0.25, .25), (0.02, .59, .59), (1., 1., 1.)),
  'green':  ( (0.0, 0.0, 0.0), (0.02, .45, .45), (1., .97, .97)),
  'blue' :  ( (0.0, 1.0, 1.0), (0.02, .75, .75), (1., 0.45, 0.45))
}

cm = m.colors.LinearSegmentedColormap('my_colormap', cdict, 1024)

#x = np.arange(0, 10, .1)
#y = np.arange(0, 10, .1)
#X, Y = np.meshgrid(x,y)

#data = 2*( np.sin(X) + np.sin(3*Y) )
def run(h,j,l, n_iterations,a,n_samp):
    # 1D Lattice
    g = nk.graph.Hypercube(length=l, n_dim=1, pbc=True)
    
    # Hilbert space of spins on the graph
    hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)
    
    # Ising spin hamiltonian
    ha = nk.operator.Ising(h=h, hilbert=hi, graph=g, J = j)
    
    # RBM Spin Machine
    ma = nk.machine.RbmSpin(alpha=a, hilbert=hi,use_visible_bias=False,use_hidden_bias=False)
    
    ma.init_random_parameters(seed=1234, sigma=0.01)
    
    
    # Metropolis Local Sampling
    sa = nk.sampler.MetropolisLocal(ma, n_chains=32)
    
    # Optimizer
    op = nk.optimizer.Sgd(ma, learning_rate=0.1)
    
    # Stochastic Reconfiguration
    sr = nk.optimizer.SR(ma, diag_shift=0.05)
    
    # Create the optimization driver
    gs = nk.Vmc(hamiltonian=ha, sampler=sa, optimizer=op, n_samples=n_samp,n_discard=20)

    # Run the optimization for 300 iterations
    gs.run(n_iter=n_iterations, out="DataMATRIX5_J1_1D_H_{:.1f}".format(h))
    s = ma.to_array()
    normalised = s.real/(sum(s.real))
    #normalised = np.sqrt(s.real)

    #normalised = 1
    #plt.plot(normalised)
    params = ma.parameters.imag
    #weights = params[:-24]
    print(np.shape(params))
    return normalised, params
#print (params)



def showTensor(aTensor,h,vmax,vmin):
    plt.figure()
    plt.imshow(aTensor,cmap=cm, vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title("H = {:.1f}".format(h))
    #print "1 + 1 = %i" % num #
    plt.show()
#amount of max/min value /amount of points
gap = 0.25
maxValue = 3.25
Minvalue =0
H_list1D = np.arange(Minvalue,maxValue,gap)
#print(len(H_list1D))
parameters_list = np.zeros((len(H_list1D),800))
normalised_list=np.zeros((len(H_list1D),1048576))

#p, w = run(1,1,20,1,2,300) #h,j,l, n_iterations,a,n_samp
#print(len(w))
#len(p)

for n in range (len(H_list1D)):
    normalised_list[n], parameters_list[n] = run(H_list1D[n],1,20,1000,2,1000) #h,j,l, n_iterations,a,n_samp

np.savetxt('NormalisedListSqaureRoot.csv', normalised_list, delimiter=',')    
np.savetxt('parametersImag.csv', parameters_list, delimiter=',')    

AverageWListImag=[] ##the weight matrix values
AverageSListImag = [] ##the eigenavalues average per itreation
for s in range(len(H_list1D)):
    params = parameters_list[s]
    fig2, ax2 = plt.subplots()
    ax2.plot(normalised_list[s])
    plt.title("H = {:.1f}".format(H_list1D[s]))
    plt.ylabel("Probability of a state")
    plt.xlabel("State number")
    plt.show()
    fig3, ax3 = plt.subplots()
    params1 = params[-800:]
    paramsReshaped = params1.reshape(20,40)
    average_w = np.mean(np.abs(params1.reshape(800,1)))
    AverageWListImag.append(average_w)
    u,k,vh = np.linalg.svd(paramsReshaped,full_matrices=False)
    AverageSListImag.append(np.mean(k))
    ax3.plot(k,label="the value of the eigenvectors in the weight matrix")
    plt.ylabel("Value of Eigenvalue (imaginary)")
    plt.title("H = {:.1f}".format(H_list1D[s]))
    plt.show()
    showTensor(params1.reshape(40,20), H_list1D[s],0.4,-0.4)
fig4, ax4 = plt.subplots()
ax4.scatter(H_list1D,AverageWListImag)
plt.ylabel("Average W value(imaginary)")
plt.xlabel("H")
fig5, ax5 = plt.subplots()
ax5.scatter(H_list1D,AverageSListImag)
plt.ylabel("Average Eigenvalue(imaginary)")
plt.xlabel("H")



    
    
    
    