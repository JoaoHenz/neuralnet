#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 14:18:34 2018

@author: jcdazeredo
"""

# =============================================================================
#
# TO-DO:
#    -> Backpropagation (Slide 130 - Aula 14)
#    -> Implementar Função de Custo (Slide 129 - Aula 14)   
#
# =============================================================================

import pandas as pd
import numpy as np


# =============================================================================
# Falta implementar algumas coisas nessa classe. Além disso, o data frame ta sendo
# colocado diretamente ali dentro, pois estou primeiro tentando construir o algoritmo
# de treino da rede.
# =============================================================================
class Neural(object):
    
    def __init__(self):
        y_column = -1
        data = pd.DataFrame([[5, 10, 1], [6, 12, 1], [7, 14, 0], [8, 16, 2]], columns = list('XZY'))
        self.y = pd.DataFrame(data.iloc[:, -1])
        self.data = data.drop(data.columns[y_column], axis=1)
        
        self.num_input_nodes = self.data.shape[1]
#        self.num_output_nodes = (np.unique(self.y)).shape[0]
        #TODO
        self.num_output_nodes = 2
        
        num_hidden_layers = 1
        num_nodes_per_hidden_layer = [2]
        self.num_layers = num_hidden_layers + 2
        self.num_nodes_per_layer = [2]*(self.num_layers)
        self.num_nodes_per_layer[1:-1] = num_nodes_per_hidden_layer
        self.num_nodes_per_layer[0] = self.num_input_nodes
        self.num_nodes_per_layer[-1] = self.num_output_nodes
        
        self.learning_rate = 0.1
        self.regularization = 1
        
        self.j = 0
        
    # =============================================================================
    # Inicializa a estrutura da rede neural. Criando quatro listas contendo matrizes,
    # uma para matrizes de ativação, outra para os pesos, outra para os gradientes
    # e outra para os erros. Cada elemento dessas quatro listas é uma matriz, e seus
    # índices são associados à cada camada da rede neural, sendo a camada input = 0 e
    # a camada output = -1 / último índice da lista.
    # =============================================================================
    def initiliaze_structure(self):
        matrix_activation_list = []
        matrix_weight_list = []
        matrix_gradient_list = []
        matrix_error_list = []

        # Input Layer
        new_matrix_activation = np.random.rand(self.num_nodes_per_layer[0] + 1, 1)
        new_matrix_activation[0, 0] = 1
        matrix_activation_list.append(new_matrix_activation)
        matrix_weight_list.append(np.random.rand(self.num_nodes_per_layer[1], self.num_nodes_per_layer[0] + 1))
        matrix_gradient_list.append(np.zeros((self.num_nodes_per_layer[1], self.num_nodes_per_layer[0] + 1)))
        matrix_error_list.append(np.array([[np.nan]]))
        
        # Hidden Layers
        for i in (range(self.num_layers))[1:-1]:
            new_matrix_activation = np.random.rand(self.num_nodes_per_layer[i] + 1, 1)
            new_matrix_activation[0, 0] = 1
            matrix_activation_list.append(new_matrix_activation)
            
            new_matrix_error = np.empty((self.num_nodes_per_layer[i]+1, 1))
            matrix_error_list.append(new_matrix_error)
            
            new_matrix_weight = np.random.rand(self.num_nodes_per_layer[i + 1], self.num_nodes_per_layer[i] + 1)
            matrix_weight_list.append(new_matrix_weight)
            
            new_matrix_gradient = np.zeros((self.num_nodes_per_layer[i + 1], self.num_nodes_per_layer[i] + 1))
            matrix_gradient_list.append(new_matrix_gradient)
        
        # Output Layer
        new_matrix_activation = np.random.rand(self.num_nodes_per_layer[-1] + 1, 1)
        new_matrix_activation[0, 0] = 1
        matrix_activation_list.append(new_matrix_activation)
        matrix_weight_list.append(np.array([[np.nan]]))
        matrix_gradient_list.append(np.array([[np.nan]]))
        matrix_error_list.append(np.random.rand(self.num_nodes_per_layer[-1] + 1, 1))
        
        self.activations = np.array(matrix_activation_list)
        self.weights = np.array(matrix_weight_list)
        self.errors = np.array(matrix_error_list)
        self.gradients = np.array(matrix_gradient_list)

    def sigmoid(self, x):
        return 1.0/(1+ np.exp(-x))
            
    # =============================================================================
    # Feedforward da rede para uma instância de exemplo    
    # =============================================================================
    def feedforward(self, row_number):
        # Input Layer - Coloca o vetor de entrada "i" como sendo a matriz de ativação da camada 0
        self.activations[0][1:] = np.transpose((np.array([self.data.iloc[row_number,:]])))
        
        for layer_i in (range(self.num_layers))[1:-1]:
            self.activations[layer_i][1:] = self.sigmoid(np.dot(self.weights[layer_i-1], self.activations[layer_i-1]))
            
        # Output - Ativa camada de saída
        self.activations[-1][1:] = self.sigmoid(np.dot(self.weights[-2], self.activations[layer_i-2]))
        
    def compute_errors(self, row_number):
        predict = self.activations[-1][1:]
        output = np.array(pd.DataFrame(self.y.iloc[row_number, :]))
        self.errors[-1][1:] = np.subtract(predict, output)
        
        # Cálculo dos deltas para hidden layers
        for layer_i in reversed((range(self.num_layers))[1:-1]):        
            weights_transposed = np.transpose(self.weights[layer_i])    
            # Calcula em três partes o delta da camada
            part1 = np.dot(weights_transposed, self.errors[layer_i+1][1:])
            part2 = np.multiply(self.activations[layer_i], (1-self.activations[layer_i]))
            self.errors[layer_i] = np.multiply(part1, part2)
    
    def accumulate_gradients(self):
        # Cálculo dos gradientes
        for layer_i in reversed((range(self.num_layers))[:-1]):
            activations_transposed = np.transpose(self.activations[layer_i])
            part1 = np.dot(self.errors[layer_i+1][1:], activations_transposed)
            self.gradients[layer_i] = np.add(part1, self.gradients[layer_i])
    
    def compute_final_gradients(self, num_examples):
        # Cálculo dos gradientes finais
        for layer_i in reversed((range(self.num_layers))[:-1]):
            matrix_p = np.multiply(self.regularization, self.weights[layer_i])
            
            # Zerar a primeira coluna pois bias não tem regularização
            matrix_p[:, 0] = 0
            
            d = np.add(self.gradients[layer_i], matrix_p)
            self.gradients[layer_i] = np.multiply((1/num_examples), d)
    
    def update_weights(self):
        for layer_i in reversed((range(self.num_layers))[:-1]):
            part1 = np.multiply(self.learning_rate, self.gradients[layer_i])  
            self.weights[layer_i] = np.subtract(self.weights[layer_i], part1)
    
    def fit(self):
        num_training_rows = self.data.shape[0]
        
        # Pra todos os exemplos
        for row_number in range(num_training_rows):
            print("Treinando exemplo " + str(row_number+1) + "/" + str(num_training_rows))
            self.feedforward(row_number)
            self.compute_errors(row_number)
            self.accumulate_gradients()
        
        # Regularização e Atualização de gradientes        
        self.compute_final_gradients(num_training_rows)
        self.update_weights()
    
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
n.fit()
#n.save_to_txt("test2.txt")
    



    
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
