#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 14:18:34 2018

@author: jcdazeredo
"""

import pandas as pd
import numpy as np


def sigmoid(x):
    return 1.0/(1+ np.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - x)

num_input_nodes = 1 + 1
num_output_nodes = 1

num_hidden_layers = 1
num_layers = num_hidden_layers + 2
num_nodes_per_layer = [1+1]*(num_hidden_layers + 2)
num_nodes_per_layer[0] = num_input_nodes
num_nodes_per_layer[-1] = num_output_nodes

matrix_activation_list = []
matrix_weights_list = []

# Input Layer
input_activation = np.random.rand(num_nodes_per_layer[0], 1)
input_activation[0,0] = 1
matrix_activation_list.append(input_activation)
matrix_weights_list.append(np.random.rand(num_nodes_per_layer[1], num_nodes_per_layer[0]))

# Hidden Layers
for i in (range(num_layers))[1:-1]:
    new_matrix_activation = np.random.rand(num_nodes_per_layer[i], 1)
    new_matrix_activation[0, 0] = 1
    matrix_activation_list.append(new_matrix_activation)
    
    new_matrix_weight = np.random.rand(num_nodes_per_layer[i + 1], num_nodes_per_layer[i])
    matrix_weights_list.append(new_matrix_weight)

# Output Layer
matrix_activation_list.append(np.random.rand(num_nodes_per_layer[-1], 1))

activations = np.array(matrix_activation_list)
weights = np.array(matrix_weights_list)

data = pd.DataFrame([[5, 10, 20], [6, 12, 22], [7, 14, 24], [8, 16, 26]], columns = list('ABY'))
data2 = pd.DataFrame(data.loc[:, ['A']])
x = np.array(data2.iloc[0, :])
    
# Input Layer - Coloca o vetor de entrada "i" como sendo a matriz de ativação da camada 0
input_vector = np.array(pd.DataFrame(data2.iloc[0, :]))
activations[0][1:] = input_vector
    
for layer_i in (range(num_layers))[0:-1]:
    activations[layer_i+1] = sigmoid(np.dot(weights[layer_i], activations[layer_i]))
    
activations[1]
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
