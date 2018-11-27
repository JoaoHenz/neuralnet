#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import math

global j_list

class NeuralNet(object):
    # =============================================================================
    # Falta implementar algumas coisas nessa classe. Alcm disso, o dataframe ta sendo
    # colocado diretamente ali dentro, pois estou primeiro tentando construir o algoritmo
    # de treino da rede.
    # =============================================================================
    def __init__(self, dataset, y, hidden_lengths = [8,8],fator_reg=0.25,num_saida = 1,num_entrada=1,initial_weights = []):

        #TODO fazer a inicialização dos initial_weights se eles foram passados

        self.coluna_aserpredita = y
        self.data = dataset
        #self.num_input_nodes = self.data.shape[1]
        self.num_input_nodes = num_entrada
        #TODO arrumar questão do numero de inputs? e outputs?
        #        self.num_output_nodes = (np.unique(self.coluna_aserpredita)).shape[0]
        self.num_output_nodes = num_saida
        self.num_hidden_layers = len(hidden_lengths)
        self.num_nodes_per_hidden_layer = hidden_lengths
        self.num_layers = self.num_hidden_layers + 2
        self.num_nodes_per_layer = [2]*(self.num_layers)
        self.num_nodes_per_layer[1:-1] = self.num_nodes_per_hidden_layer
        self.num_nodes_per_layer[0] = self.num_input_nodes
        self.num_nodes_per_layer[-1] = self.num_output_nodes

        self.learning_rate = 0.1
        self.fator_reg = fator_reg
        self.j = 0
        self.j_regularized = 0
        self.initiliaze_structure()

    def initiliaze_structure(self):
        # =============================================================================
        # Inicializa a estrutura da rede neural. Criando quatro listas contendo matrizes,
        # uma para matrizes de ativaco, outra para os pesos, outra para os gradientes
        # e outra para os erros. Cada elemento dessas quatro listas c uma matriz, e seus
        # ­ndices so associados   cada camada da rede neural, sendo a camada input = 0 e
        # a camada output = -1 / ultimo ­ndice da lista.
        # =============================================================================
        matrix_activation_list = []
        matrix_weight_list = []
        matrix_gradient_list = []
        matrix_error_list = []

        # Input Layer
        new_matrix_activation = np.random.rand(self.num_nodes_per_layer[0] + 1, 1)
        new_matrix_activation[0, 0] = 1
      
        matrix_activation_list.append(new_matrix_activation)
        matrix_weight_list.append(np.random.uniform(-1,+1, size = (self.num_nodes_per_layer[1], self.num_nodes_per_layer[0] + 1)))
        matrix_gradient_list.append(np.zeros((self.num_nodes_per_layer[1], self.num_nodes_per_layer[0] + 1)))
        matrix_error_list.append(np.array([[np.nan]]))

        # Hidden Layers
        for i in (range(self.num_layers))[1:-1]:
            new_matrix_activation = np.random.rand(self.num_nodes_per_layer[i] + 1, 1)
            new_matrix_activation[0, 0] = 1
            matrix_activation_list.append(new_matrix_activation)
            
            new_matrix_error = np.empty((self.num_nodes_per_layer[i]+1, 1))
            matrix_error_list.append(new_matrix_error)
            
            new_matrix_weight = np.random.uniform(-1,+1, size = (self.num_nodes_per_layer[i + 1], self.num_nodes_per_layer[i] + 1))
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

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0) # only difference

    def feedforward(self, row_number):
        # =============================================================================
        # Feedforward da rede para uma instcncia de exemplo
        # =============================================================================
        # Input Layer - Coloca o vetor de entrada "i" como sendo a matriz de ativaco da camada 0
        self.activations[0][1:] = np.transpose(np.array([self.data[row_number,:]]))

        for layer_i in (range(self.num_layers))[1:-1]:
            self.activations[layer_i][1:] = self.sigmoid(np.dot(self.weights[layer_i-1], self.activations[layer_i-1]))

        # Output - Ativa camada de sa­da
        self.activations[-1][1:] = self.sigmoid(np.dot(self.weights[-2], self.activations[-2]))

    def compute_errors(self, row_number):
        predict = self.activations[-1][1:]
        predict = np.where(predict > 0.5, 1, 0)
        
        output = self.coluna_aserpredita[row_number, :][0]
        self.errors[-1][1:] = np.subtract(predict, output)

        # Calculo dos deltas para hidden layers
        for layer_i in reversed((range(self.num_layers))[1:-1]):
            weights_transposed = np.transpose(self.weights[layer_i])
            # Calcula em tres partes o delta da camada
            part1 = np.dot(weights_transposed, self.errors[layer_i+1][1:])
            part2 = np.multiply(part1, self.activations[layer_i])
            self.errors[layer_i] = np.multiply(part2, (1-self.activations[layer_i]))

    def accumulate_gradients(self):
        # Calculo dos gradientes
        for layer_i in reversed((range(self.num_layers))[:-1]):
            activations_transposed = np.transpose(self.activations[layer_i])
            part1 = np.dot(self.errors[layer_i+1][1:], activations_transposed)
            self.gradients[layer_i] = np.add(part1, self.gradients[layer_i])

    def compute_final_gradients(self, num_examples):
        # Calculo dos gradientes finais
        for layer_i in reversed((range(self.num_layers))[:-1]):
            matrix_p = np.multiply(self.fator_reg, self.weights[layer_i])

            # Zerar a primeira coluna pois bias no tem regularizaco
            matrix_p[:, 0] = 0

            d = np.add(self.gradients[layer_i], matrix_p)
            self.gradients[layer_i] = np.multiply((1/num_examples), d)

    def update_weights(self):
        for layer_i in reversed((range(self.num_layers))[:-1]):
            part1 = np.multiply(self.learning_rate, self.gradients[layer_i])
            self.weights[layer_i] = np.subtract(self.weights[layer_i], part1)

    def compute_j(self, row_number):
        output = np.array([self.coluna_aserpredita[row_number, :]])
        predict = self.activations[-1][1:]
#        predict = np.where(predict > 0.5, 1, 0)
        
        part1 = np.multiply((-output), np.log10(predict))
        part2 = np.multiply((1 - output), np.log10(1 - predict))
        j = np.subtract(part1, part2)
        self.j = self.j + np.sum(j)

    def sum_weights_squared(self):
        result = 0

        for layer_i in range(self.num_layers)[:-1]:
            part1 = np.multiply(self.weights[layer_i][:,1:], self.weights[layer_i][:,1:])
            result = result + np.sum(part1)

        return result

    def feedforward_classify(self, row_instance):
        # =============================================================================
        # Feedforward da rede para uma instcncia de exemplo
        # =============================================================================
        # Input Layer - Coloca o vetor de entrada "i" como sendo a matriz de ativaco da camada 0
        self.activations[0][1:] = np.transpose(np.array([row_instance]))

        for layer_i in (range(self.num_layers))[1:-1]:
            self.activations[layer_i][1:] = self.sigmoid(np.dot(self.weights[layer_i-1], self.activations[layer_i-1]))

        return self.sigmoid(np.dot(self.weights[-2], self.activations[-2]))[0][0]

    def classify(self, instances):
        num_rows = instances.shape[0]
        results = np.zeros((num_rows, 1))

        for row_i in range(num_rows):
            results[row_i,0] = self.feedforward_classify(instances[row_i,:])

        return results

    def compute_j_regularized(self, num_training_rows):
        self.j = self.j / num_training_rows
        s = self.sum_weights_squared()
        self.j_regularized = (self.fator_reg / (2 * num_training_rows)) * s

    # =============================================================================
    # teste
    # =============================================================================
    def zerar_matrix(self):
        for layer_i in (range(self.num_layers))[0:-1]:
            self.gradients[layer_i].fill(0)
        
    def fit(self,show=False,filenamefig='lastfitresuslt'):
        # funco de treinamento do modelo
        num_training_rows = self.data.shape[0]
        num_mini_batch = 50
        num_loops = math.floor(num_training_rows / num_mini_batch)
        num_rows_rest = num_training_rows - (num_loops*num_mini_batch)
        
        global j_list
        j_list = []

        repetitions = 50                         
        
        for i in range(repetitions):
            self.j = 0
            mini_i = 0
            print("Treinando Loop " + str(i+1) + "/" + str(repetitions))
            
            for row_number in range(num_training_rows):
                self.feedforward(row_number)
                self.compute_errors(row_number)
                self.accumulate_gradients()
                self.compute_j(row_number)
                
                if (row_number % num_mini_batch == 0) and row_number != 0:
                    mini_i += 1
#                    print("Mini-batch " + str(mini_i+1) + "/" + str(num_mini_batch+1))
                    # Regularizaco e Atualizacao de gradientes
                    self.compute_final_gradients(num_mini_batch)
                    self.update_weights()
                    self.zerar_matrix()
                
            if num_rows_rest > 0:                             
                # Regularizaco e Atualizacao de gradientes
                self.compute_final_gradients(num_rows_rest)
                self.update_weights()
                self.zerar_matrix()
                        
            # J Regularizado
            self.compute_j_regularized(num_mini_batch)
            j_list.append(self.j)
            
              

#        for i in range(loops):
#            self.j = 0
#            print("Treinando Loop " + str(i+1) + "/" + str(loops))
#            # Pra todos os exemplos
#            for row_number in range(num_training_rows):
#                #                print("Treinando exemplo " + str(row_number+1) + "/" + str(num_training_rows))
#                self.feedforward(row_number)
#
#                self.compute_j(row_number)
#
#                self.compute_errors(row_number)
#                self.accumulate_gradients()
#
#            # J Regularizado
#            self.compute_j_regularized(num_training_rows)
#
#            # Regularizaco e Atualizaco de gradientes
#            self.compute_final_gradients(num_training_rows)
#            self.update_weights()
#
#            j_list.append(self.j)
#            self.zerar_matrix()  

            







#        axis_x = range(num_loops)
#
#        fig, ax = plt.subplots()
#        ax.plot(axis_x, j_list)
#
#        ax.set(xlabel='Loop', ylabel='Error)', title='Error vs Loop')
#        ax.grid()
#
#        if show:
#            plt.savefig(filenamefig+'.png')
#            plt.show()
#        else:
#            plt.savefig(filenamefig+'.png')

    def savetofile(self, filename='lastneuralnet'):
        # =============================================================================
        # Salva a estrutura e informacoes da rede em um arquivo txt
        # =============================================================================
        f = open(filename, "a")

        f.write("### Informacoes da Rede Neural ###\n\n")
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
            f.write('\n-> Layer ' + str(layer) + ' - Ativaco: \n\n')
            np.savetxt(f, self.activations[layer], delimiter='    ', fmt='%1.4f')
            f.write('\n-> Layer ' + str(layer) + ' - Pesos: \n\n')
            np.savetxt(f, self.weights[layer], delimiter='    ', fmt='%1.4f')
            f.write('\n-> Layer ' + str(layer) + ' - Erros: \n\n')
            np.savetxt(f, self.errors[layer], delimiter='    ', fmt='%1.4f')
        f.close()

    def save_finalweights(self, testname = 'ultimoteste'):
        # pedido na definico do trab
        #salva os pesos finais da Rede
        self.weights = np.array(matrix_weight_list)
        f = open(testname+'_finalweights.txt', "a")
        for i in range(0, len(self.weights)):
            line = ''
            for j in range(0,len(self.weights[i])):
                for k in range(0,len(self.weights[i][j])):
                    line+= str(self.weights[i][j,k]) +','
                line+=';'
            f.write(line)
        f.close()



import pandas as pd
import generallib as gl

##le dataset
#y_column = -1
#data = pd.read_csv("/mnt/Data/neuralnet/data/Churn_Modelling_Edited.csv")
#
#y = np.array(pd.DataFrame(data.iloc[:, y_column]))
##
#dataset = np.array(data.drop(data.columns[y_column], axis=1))

#dataset = gl.normalization(dataset)
#


# Importing the dataset
dataset = pd.read_csv('data/Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

y_train = np.transpose(np.array([y_train]))
y_test = np.transpose(np.array([y_test]))

n = NeuralNet(X_train, y_train)
n.savetofile(filename='lastneuralnet')
n.fit() #
n.savetofile(filename='lastneuralnet2')

result = n.classify(X_test)

result = (result > 0.5)
expected = (y_test > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(expected, result)