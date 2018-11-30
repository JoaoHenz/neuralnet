#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import time
import math

global j_list, j_reg_list

class NeuralNet(object):
    # =============================================================================
    # Falta implementar algumas coisas nessa classe. Alcm disso, o dataframe ta sendo
    # colocado diretamente ali dentro, pois estou primeiro tentando construir o algoritmo
    # de treino da rede.
    # =============================================================================
    def __init__(self, dataset, y, hidden_lengths = [4], learning_rate = 0.1, fator_reg = 0, num_saida = 2, num_entrada = 1, initial_weights = [], numeric = False):

        self.output_column = y
        self.data = dataset
        self.num_input_nodes = num_entrada
        self.num_output_nodes = num_saida
        self.num_hidden_layers = len(hidden_lengths)
        self.num_nodes_per_hidden_layer = hidden_lengths
        self.num_layers = self.num_hidden_layers + 2
        self.num_nodes_per_layer = [2]*(self.num_layers)
        self.num_nodes_per_layer[1:-1] = self.num_nodes_per_hidden_layer
        self.num_nodes_per_layer[0] = self.num_input_nodes
        self.num_nodes_per_layer[-1] = self.num_output_nodes

        self.learning_rate = learning_rate
        self.fator_reg = fator_reg
        self.j = 0
        self.j_regularized = 0
        self.numeric = numeric
        
        if len(initial_weights) == 0:
          self.initialize_structure()
        else:
          self.initialize_with_weights(initial_weights)

        self.prepare_outputs_matrix()
        
    def prepare_outputs_matrix(self):
        num_rows = self.output_column.shape[0]
        
        if self.numeric:
            # Numeric Output
            self.output_matrix = np.zeros((num_rows, self.output_column.shape[1], 1))
            
            for row_i in range(num_rows):
                self.output_matrix[row_i] = np.transpose(np.array([self.output_column[row_i, :]]))
  
        else:
            # Classifier
            self.output_matrix = np.zeros((num_rows, self.num_output_nodes, 1))
            
            for row_i in range(num_rows):
                out = int(self.output_column[row_i, 0])
                self.output_matrix[row_i, out, 0] = 1

        
    def initialize_with_weights(self, initial_weights):
        matrix_activation_list = []
        matrix_gradient_list = []
        matrix_error_list = []

        # Input Layer
        new_matrix_activation = np.random.rand(self.num_nodes_per_layer[0] + 1, 1)
        new_matrix_activation[0, 0] = 1
      
        matrix_activation_list.append(new_matrix_activation)
        matrix_gradient_list.append(np.zeros((self.num_nodes_per_layer[1], self.num_nodes_per_layer[0] + 1)))
        matrix_error_list.append(np.array([[np.nan]]))

        # Hidden Layers
        for i in (range(self.num_layers))[1:-1]:
            new_matrix_activation = np.random.rand(self.num_nodes_per_layer[i] + 1, 1)
            new_matrix_activation[0, 0] = 1
            matrix_activation_list.append(new_matrix_activation)
            
            new_matrix_error = np.empty((self.num_nodes_per_layer[i]+1, 1))
            matrix_error_list.append(new_matrix_error)
            
            new_matrix_gradient = np.zeros((self.num_nodes_per_layer[i + 1], self.num_nodes_per_layer[i] + 1))
            matrix_gradient_list.append(new_matrix_gradient)

        # Output Layer
        new_matrix_activation = np.random.rand(self.num_nodes_per_layer[-1] + 1, 1)
        new_matrix_activation[0, 0] = 1
        matrix_activation_list.append(new_matrix_activation)
                
        matrix_gradient_list.append(np.array([[np.nan]]))
        matrix_error_list.append(np.random.rand(self.num_nodes_per_layer[-1] + 1, 1))
        
        self.activations = np.array(matrix_activation_list)        
        self.weights = np.append(initial_weights, [np.nan])        
        self.errors = np.array(matrix_error_list)
        self.gradients = np.array(matrix_gradient_list)
    
    def initialize_structure(self):
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
        
        output = self.output_matrix[row_number]
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
            gradient_layer_i = np.dot(self.errors[layer_i+1][1:], activations_transposed)
            self.gradients[layer_i] = np.add(gradient_layer_i, self.gradients[layer_i])

    def compute_final_gradients(self, num_examples):
        # Calculo dos gradientes finais
        for layer_i in reversed((range(self.num_layers))[:-1]):
            matrix_p = np.multiply(self.fator_reg, self.weights[layer_i])

            # Zerar a primeira coluna pois bias no tem regularizaco
            matrix_p[:, 0] = 0

            d = np.add(self.gradients[layer_i], matrix_p)
            self.gradients[layer_i] = np.divide(d, num_examples)
            
    def update_weights(self):
        for layer_i in reversed((range(self.num_layers))[:-1]):
            part1 = np.multiply(self.learning_rate, self.gradients[layer_i])
            self.weights[layer_i] = np.subtract(self.weights[layer_i], part1)

    def compute_j(self, row_number):
        output = self.output_matrix[row_number]
        predict = self.activations[-1][1:]
        
        part1 = np.multiply((-output), np.log(predict))
        part2 = np.multiply((1 - output), np.log(1 - predict))
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

        return self.sigmoid(np.dot(self.weights[-2], self.activations[-2]))

    def classify(self, instances):
        num_rows = instances.shape[0]
        results = np.zeros((num_rows, 1))

        for row_i in range(num_rows):
            results[row_i,0] = np.argmax(self.feedforward_classify(instances[row_i,:]))

        return results

    def compute_j_regularized(self, num_training_rows):
        j = self.j / num_training_rows
        s = self.sum_weights_squared()
        s = (self.fator_reg / (2 * num_training_rows)) * s
        self.j_regularized = j + s
  
    # =============================================================================
    # TODO
    # =============================================================================
    def zerar_matrix(self):
        for layer_i in (range(self.num_layers))[0:-1]:
            self.gradients[layer_i].fill(0)

    def fit(self, epochs = 1, batch_size = 0, show = False, verbose = True, filenamefig = 'lastfitresuslt', save_gradients = False):
        start_runtime_total = time.time()  
        num_training_rows = self.data.shape[0]
        if batch_size == 0:
            batch_size = num_training_rows
        num_loops = math.floor(num_training_rows / batch_size)
        num_rows_rest = num_training_rows - (num_loops*batch_size)
        
        global j_list, j_list2
        j_list = []
        j_reg_list = []                    
        
        for epoch_i in range(epochs):
            start_runtime_epoch = time.time()
            
            if verbose:
              print("Epoch " + str(epoch_i+1) + "/" + str(epochs))
            
            for row_number in range(num_training_rows):
                self.feedforward(row_number)
                self.compute_errors(row_number)
                self.accumulate_gradients()
                self.compute_j(row_number)

                if (row_number % batch_size-1 == 0) and row_number != 0:
                    # Regularizaco e Atualizacao de gradientes
                    self.compute_final_gradients(batch_size)
                    self.update_weights()
                    
                    if not save_gradients:
                        self.zerar_matrix()
                    
                    
            if num_rows_rest > 0:                             
                # Regularizaco e Atualizacao de gradientes
                self.compute_final_gradients(num_rows_rest)
                self.update_weights()
                
                if not save_gradients:
                    self.zerar_matrix()
            
            self.compute_j_regularized(num_training_rows)
            
            if save_gradients:
                self.save_final_report()
            
            total_runtime_epoch = time.time() - start_runtime_epoch
            if verbose:
                print(str(num_training_rows) + "/" + str(num_training_rows) + " – " + str("%.2f" % total_runtime_epoch) + "s – J: " + str(self.j) + " – J Reg: " + str(self.j_regularized))
            
            j_list.append(self.j)
            j_reg_list.append(self.j_regularized)
            
            self.j = 0
            self.j_regularized = 0
            final_runtime_total = time.time() - start_runtime_total
            if verbose:
                print(str("%.4f" % final_runtime_total))

#        axis_x = range(len(j_list))
#
#        fig, ax = plt.subplots()
#        ax.plot(axis_x, j_list)
#
#        ax.set(xlabel='Loop', ylabel='J)', title='J vs Loop')
#        ax.grid()
#
#        if show:
#            plt.savefig(filenamefig+'.png')
#            plt.show()
#        else:
#            plt.savefig(filenamefig+'.png')
#            
#        axis_x = range(len(j_list2))
#
#        fig, ax = plt.subplots()
#        ax.plot(axis_x, j_list2)
#
#        ax.set(xlabel='Loop', ylabel='J_Reg)', title='J Regularized vs Loop')
#        ax.grid()
#
#        if show:
#            plt.savefig(filenamefig+'2.png')
#            plt.show()
#        else:
#            plt.savefig(filenamefig+'2.png')

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
    
    def string_gradients(self):
        line = ''
        for i in range(len(self.gradients)-1):
            for j in range(len(self.gradients[i])):
                for k in range(len(self.gradients[i][j])):
                    line += "%.5f" % self.gradients[i][j,k]
                    
                    if k != len(self.gradients[i][j])-1:
                        line += ', '
                
                if j != len(self.gradients[i]) -1:
                    line += '; '
            
            if i != len(self.gradients)-2:
                line += "\n"
         
        return line
    
    def string_dataset(self):
        line = ''
        for line_i in range(self.data.shape[0]):
            for att in range(self.data.shape[1]):
                line += "%.5f" % self.data[line_i, att]
                if att != self.data.shape[1]-1:
                    line += ", "
                else:
                    line += "; "
            for out in range(self.output_column.shape[1]):
                line += "%.5f" % self.output_column[line_i, out]
                if out != self.output_column.shape[1]-1:
                    line += ", "
            
            if line_i != self.data.shape[0]-1:
                line += "\n"
        
        return line
    
    def save_final_report(self, filename = 'final'):
        f = open(filename + '_report.txt', "w+")
        
        # Escreve gradientes dos pesos
        f.write(self.string_gradients())
        
        f.write("\n\n")
        
        # Escreve fator de regularizacao
        f.write("Fator de Regularizacao: " + str(self.fator_reg))
        
        f.write("\n\n")
        
        f.write("J: %.5f \n" % self.j)
        f.write("J Regularizado: %.5f\n\n" % self.j_regularized)
        # Escreve dataset
        f.write(self.string_dataset())

        f.close()
        