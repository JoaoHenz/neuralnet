#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 14:18:34 2018

@author: jcdazeredo
"""

# =============================================================================
# Arquivo com algumas implementações
#
# Próximos Passos:
#    -> Backpropagation
#    -> Verificar o Feedforward com os slides (fiz algo diferente do que pediu acho)
#    -> Ajeitar o BIAS, deve tá errado
#
#
#
# =============================================================================

import pandas as pd
import numpy as np


# =============================================================================
# K-fold Estratificado
# Retorna uma lista, cada item sendo um dataframe (fold)
# =============================================================================
def stratified_k_fold(k_folds, y_column, dataframe):
    """
    1) Calcula quantas instâncias de cada classe devem ser inseridos
    em cada fold;
    
    2) Cria um fold por vez, inserindo N instâncias de cada classe no fold;
    
    3) No último fold, insere as instâncias que sobraram. 
    
    *Qualquer instância do dataframe original deve estar somente em um único fold. 
    """
    
    y = dataframe.iloc[:,y_column]
    classes = np.unique(y.iloc[:])
    num_per_fold = {}
    k_fold_dataframes = []
    classes_index = {}
    counter = {}

    for c in classes:
        total_rows_class = np.sum(y.iloc[:] == c)
        num = int(round(total_rows_class/k_folds))
        num_per_fold[c] = num
        index = y[y.iloc[:] == c].index
        classes_index[c] = index
        counter[c] = 0    
    
    for k in range(k_folds-1):
        index = np.array([], dtype = "int64")
        for c in classes:
            num = num_per_fold[c]
            limit = counter[c] + num
            index = np.concatenate((index, classes_index[c][counter[c]:limit].values))
            counter[c] += num
        k_fold_dataframes.append(dataframe.iloc[index])
    
    index = np.array([], dtype = "int64")
    for c in classes:
        limit = classes_index[c].shape[0]
        index = np.concatenate((index, classes_index[c][counter[c]:limit].values))
        counter[c] += num
    k_fold_dataframes.append(dataframe.iloc[index])

    return k_fold_dataframes

#data = pd.read_csv("wine.csv", header = None)
#
#x = stratified_k_fold(10, 0, data)
    

class Neural(object):
    
    def __init__(self):
        self.num_input_nodes = 2
        self.num_output_nodes = 1
        
        self.num_hidden_layers = 1
        self.num_layers = self.num_hidden_layers + 2
        self.num_nodes_per_layer = [2]*(self.num_layers)
        self.num_nodes_per_layer[0] = self.num_input_nodes
        self.num_nodes_per_layer[-1] = self.num_output_nodes
                
        data = pd.DataFrame([[5, 10], [6, 12], [7, 14], [8, 16]], columns = list('XY'))
        self.y = pd.DataFrame(data.loc[:, 'Y'])
        self.data = pd.DataFrame(data.loc[:, 'X'])
        self.learning_rate = 0.1
        
    def initiliaze_structure(self):
        matrix_activation_list = []
        matrix_weight_list = []
        matrix_gradient_list = []
        matrix_error_list = []

        # Input Layer
        input_activation = np.random.rand(self.num_nodes_per_layer[0], 1)
        input_activation[0,0] = 1
        matrix_activation_list.append(input_activation)
        matrix_weight_list.append(np.random.rand(self.num_nodes_per_layer[1], self.num_nodes_per_layer[0]))
        matrix_gradient_list.append(np.random.rand(self.num_nodes_per_layer[1], self.num_nodes_per_layer[0]))
        matrix_error_list.append(np.array([[np.nan]]))
        
        # Hidden Layers
        for i in (range(self.num_layers))[1:-1]:
            new_matrix_activation = np.random.rand(self.num_nodes_per_layer[i], 1)
            new_matrix_activation[0, 0] = 1
            matrix_activation_list.append(new_matrix_activation)
            
            new_matrix_error = np.empty((self.num_nodes_per_layer[i], 1))
            matrix_error_list.append(new_matrix_error)
            
            new_matrix_weight = np.random.rand(self.num_nodes_per_layer[i + 1], self.num_nodes_per_layer[i])
            matrix_weight_list.append(new_matrix_weight)
            
            new_matrix_gradient = np.random.rand(self.num_nodes_per_layer[i + 1], self.num_nodes_per_layer[i])
            matrix_gradient_list.append(new_matrix_gradient)
        
        # Output Layer
        matrix_activation_list.append(np.random.rand(self.num_nodes_per_layer[-1], 1))
        matrix_weight_list.append(np.array([[np.nan]]))
        matrix_gradient_list.append(np.array([[np.nan]]))
        matrix_error_list.append(np.zeros((self.num_nodes_per_layer[-1], 1)))
        
        
        self.activations = np.array(matrix_activation_list)
        self.weights = np.array(matrix_weight_list)
        self.errors = np.array(matrix_error_list)

    def sigmoid(self, x):
        return 1.0/(1+ np.exp(-x))

#    def sigmoid_derivative(self, x):
#        return x * (1.0 - x)
    
    def feedforward(self):
        # Input Layer - Coloca o vetor de entrada "i" como sendo a matriz de ativação da camada 0
        input_vector = np.array(pd.DataFrame(self.data.iloc[0, :]))
        self.activations[0][1:] = input_vector

        for layer_i in (range(self.num_layers))[0:-1]:
            self.activations[layer_i+1] = self.sigmoid(np.dot(self.weights[layer_i], self.activations[layer_i]))
    
    def backpropagation(self):
        # TO-DO: incompleto
        
        # Cálculo dos deltas
        # Camada de saída
        predict = self.activations[-1]
        output = np.array(pd.DataFrame(self.y.iloc[0, :]))
        self.errors[-1] = predict - output
        
        # Hidden Layers
        for layer_i in reversed((range(self.num_layers))[:-1]):
            weights_transposed = np.transpose(self.weights[layer_i])
            # Calcula em três partes o delta da camada
            part1 = np.dot(weights_transposed, self.errors[layer_i+1])
            part2 = np.multiply(self.activations[layer_i], (1-self.activations[layer_i]))
            self.errors[layer_i] = np.multiply(part1, part2)
        
        # Vetor dos gradientes
        for layer_i in reversed((range(self.num_layers))[:-1]):
            activations_transposed = np.transpose(self.activations[layer_i])
            # Calcula em três partes o delta da camada
            part1 = np.dot(weights_transposed, self.errors[layer_i+1])
            part2 = np.multiply(self.activations[layer_i], (1-self.activations[layer_i]))
            self.errors[layer_i] = np.multiply(part1, part2)
    
    # =============================================================================
    # Salva a estrutura e informações da rede em um arquivo txt    
    # =============================================================================
    def save_to_txt(self, filename):
        f = open(filename, "a")
        f.write("### Informações da Rede Neural ###\n\n")
        f.write('Quantidade de Inputs: ' + str(self.num_nodes_per_layer[0]) + "\n")
        f.write('Quantidade de Hidden Layers: ' + str(self.num_hidden_layers) + "\n")
        f.write('Quantidade Total de Layers: ' + str(self.num_layers) + "\n")
        f.write('Quantidade de Outputs: ' + str(self.num_nodes_per_layer[-1]) + "\n\n")
        
        f.write('Quantidade de Nodos por Layer: \n')
        
        for layer in range(self.num_layers):
            f.write('\t-> Layer ' + str(layer) + ': ' + str(self.num_nodes_per_layer[layer]) + "\n")
        
        f.write('\n\nMatrizes de cada Layer: \n\n')
        
        for layer in range(self.num_layers):
            if layer != 0:
                f.write('\n\n#################################\n')
            f.write('\n-> Layer ' + str(layer) + ' - Ativação: \n\n')
            np.savetxt(f, self.activations[layer], delimiter='    ', fmt='%1.4f')
            f.write('\n-> Layer ' + str(layer) + ' - Pesos: \n\n')
            np.savetxt(f, self.weights[layer], delimiter='    ', fmt='%1.4f')
            f.write('\n-> Layer ' + str(layer) + ' - Erros: \n\n')
            np.savetxt(f, self.errors[layer], delimiter='    ', fmt='%1.4f')
            
        f.close()
        
        
n = Neural() 
n.initiliaze_structure()
n.feedforward()
#n.save_to_txt("test2.txt")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
