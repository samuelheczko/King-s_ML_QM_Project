#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 15:16:24 2021

@author: samuelheczko
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import matplotlib as m
cdict = {
  'red'  :  ( (0.0, 0.25, .25), (0.02, .59, .59), (1., 1., 1.)),
  'green':  ( (0.0, 0.0, 0.0), (0.02, .45, .45), (1., .97, .97)),
  'blue' :  ( (0.0, 1.0, 1.0), (0.02, .75, .75), (1., 0.45, 0.45))
}

cm = m.colors.LinearSegmentedColormap('my_colormap', cdict, 1024)   
## PLOTTER FOR MATRIX ANALYSIS

##Plot: average S + w, Full wavefunciton, spectrum for H = 0,1,2

#parameters_list = np.zeros((len(H_list1D),800))
#normalised_list
params = parameters_list
def showTensor(aTensor,h):
    plt.figure()
    plt.imshow(aTensor,cmap=cm, vmin=-0.5, vmax=0.5)
    plt.colorbar()
    plt.title("H = {:.1f}".format(h))
    #print "1 + 1 = %i" % num #
    plt.show()
def probabiltiyPlotter(): 
    for n in range(5):
        fig, ax = plt.subplots()
        H = n*0.5
        index = 2*n
        ax.plot(normalised_list[index],label="H = {:.1f}".format(H), color = "grey")
        leg = ax.legend(prop={'size': 8})
        plt.ylabel("Probabiltiy of the state")
        plt.xlabel("Number of state")
        plt.show()
def EigenPlotter(ParameterList,average=False):
    averageList = []
    H_list3=[]
    fig2, ax2 = plt.subplots()
    for n in range(7):
        H = n*0.5
        index=n*2
        H_list3.append(H)
        paramsReshaped = ParameterList[index].reshape(20,40)
        
        u,k,vh = np.linalg.svd(paramsReshaped,full_matrices=False)
        k2 = k[:-7]/max(k)
        if average == False:
            ax2.plot(k2,label="H = {:.1f}".format(H))
            leg = ax2.legend(prop={'size': 10})
        else:
            averageList.append(np.mean(k))
    if average == False:
        plt.ylabel("Scaled eigenvalue (imaginary)")
        plt.xlabel("Index")
        plt.show
    else:
        ax2.scatter(H_list3,averageList,marker="x",color = "k",label = "Eigenvalue (imaginary)")
        plt.ylabel("Average Eigenvalue")
        plt.xlabel("H")
        plt.axvline(1,color='grey', linestyle='--',label="Critical Point")
        leg = ax2.legend(prop={'size': 10})
        plt.show
       
#run the funciton,for plots of normalised spectrum or average value of eigenvalue vs H (average = True)

def WeightPlotter(ParameterList):
    average_list = []
    H_list3 = []
    fig2, ax2 = plt.subplots()
    for n in range(len(ParameterList)):
        index = n
        H = H_list1D[index]
        H_list3.append(H)
        #paramsReshaped = ParameterList[index].reshape(40,20)
        s = np.mean(np.abs(ParameterList[index]))
        average_list.append(s)
        #showTensor(paramsReshaped, H)
    ax2.scatter(H_list3,average_list,label="Average absolute weight value (imaginary)")
    plt.ylabel("Value")
    plt.xlabel("H")
    plt.axvline(1,color='grey', linestyle='--',label="Critical Point")
    leg = ax2.legend(prop={'size': 8})
    
    
#WeightPlotter(params)
EigenPlotter(params)

 
        
    
