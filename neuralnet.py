
# TODO -> Não sei como faz pra multiplas classes

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import generallib as gl
global j_list


class NeuralNet(object):
    # =============================================================================
    # Falta implementar algumas coisas nessa classe. Além disso, o data frame ta sendo
    # colocado diretamente ali dentro, pois estou primeiro tentando construir o algoritmo
    # de treino da rede.
    # =============================================================================
    def __init__(self, dataset, y):
        self.y = y
        self.data = dataset
        self.num_input_nodes = self.data.shape[1]
        #TODO
        #        self.num_output_nodes = (np.unique(self.y)).shape[0]
        self.num_output_nodes = 1
        self.num_hidden_layers = 2
        self.num_nodes_per_hidden_layer = [8, 8]
        self.num_layers = self.num_hidden_layers + 2
        self.num_nodes_per_layer = [2]*(self.num_layers)
        self.num_nodes_per_layer[1:-1] = self.num_nodes_per_hidden_layer
        self.num_nodes_per_layer[0] = self.num_input_nodes
        self.num_nodes_per_layer[-1] = self.num_output_nodes

        self.learning_rate = 0.1
        self.regularization = 1
        self.j = 0
        self.j_regularized = 0
        self.initiliaze_structure()

    def initiliaze_structure(self):
        # =============================================================================
        # Inicializa a estrutura da rede neural. Criando quatro listas contendo matrizes,
        # uma para matrizes de ativação, outra para os pesos, outra para os gradientes
        # e outra para os erros. Cada elemento dessas quatro listas é uma matriz, e seus
        # índices são associados à cada camada da rede neural, sendo a camada input = 0 e
        # a camada output = -1 / último índice da lista.
        # =============================================================================
        matrix_activation_list = []
        matrix_activation_c_list = []
        matrix_weight_list = []
        matrix_gradient_list = []
        matrix_error_list = []

        # Input Layer
        new_matrix_activation = np.random.rand(self.num_nodes_per_layer[0] + 1, 1)
        new_matrix_activation[0, 0] = 1
        matrix_activation_c_list.append(new_matrix_activation)
        matrix_activation_list.append(new_matrix_activation)
        matrix_weight_list.append(np.random.rand(self.num_nodes_per_layer[1], self.num_nodes_per_layer[0] + 1))
        matrix_gradient_list.append(np.zeros((self.num_nodes_per_layer[1], self.num_nodes_per_layer[0] + 1)))
        matrix_error_list.append(np.array([[np.nan]]))

        # Hidden Layers
        for i in (range(self.num_layers))[1:-1]:
            new_matrix_activation = np.random.rand(self.num_nodes_per_layer[i] + 1, 1)
            new_matrix_activation[0, 0] = 1
            matrix_activation_list.append(new_matrix_activation)
            matrix_activation_c_list.append(new_matrix_activation)

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
        matrix_activation_c_list.append(new_matrix_activation)
        matrix_weight_list.append(np.array([[np.nan]]))
        matrix_gradient_list.append(np.array([[np.nan]]))
        matrix_error_list.append(np.random.rand(self.num_nodes_per_layer[-1] + 1, 1))

        self.activations = np.array(matrix_activation_list)
        self.activations_c = np.array(matrix_activation_c_list)
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
        # Feedforward da rede para uma instância de exemplo
        # =============================================================================
        # Input Layer - Coloca o vetor de entrada "i" como sendo a matriz de ativação da camada 0
        self.activations[0][1:] = np.transpose(np.array([self.data[row_number,:]]))

        for layer_i in (range(self.num_layers))[1:-1]:
            self.activations[layer_i][1:] = self.sigmoid(np.dot(self.weights[layer_i-1], self.activations[layer_i-1]))

        # Output - Ativa camada de saída
        self.activations[-1][1:] = self.sigmoid(np.dot(self.weights[-2], self.activations[-2]))

    def compute_errors(self, row_number):
        predict = self.activations[-1][1:]
        output = self.y[row_number, :][0]
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

    def compute_j(self, row_number):
        output = self.y[row_number, :][0]

        part1 = np.multiply((-output), np.log10(self.activations[-1][1:]))
        part2 = np.multiply((-(1 - output)), np.log10(1 - self.activations[-1][1:]))
        j = np.add(part1, part2)
        self.j = self.j + np.sum(j)

    def sum_weights_squared(self):
        result = 0

        for layer_i in range(self.num_layers)[:-1]:
            part1 = np.multiply(self.weights[layer_i][:,1:], self.weights[layer_i][:,1:])
            result = result + np.sum(part1)

        return result

    def feedforward_classify(self, row_instance):
        # =============================================================================
        # Feedforward da rede para uma instância de exemplo
        # =============================================================================
        # Input Layer - Coloca o vetor de entrada "i" como sendo a matriz de ativação da camada 0
        self.activations_c[0][1:] = np.transpose(np.array([row_instance]))

        for layer_i in (range(self.num_layers))[1:-1]:
            self.activations_c[layer_i][1:] = self.sigmoid(np.dot(self.weights[layer_i-1], self.activations_c[layer_i-1]))

        return self.sigmoid(np.dot(self.weights[-2], self.activations_c[-2]))[0][0]

    def classify(self, instances):
        num_rows = instances.shape[0]
        results = np.zeros((num_rows, 1))

        for row_i in range(num_rows):
            results[row_i,0] = self.feedforward_classify(instances[row_i,:])

        return results

    def compute_j_regularized(self, num_training_rows):
        self.j = self.j / num_training_rows
        s = self.sum_weights_squared()
        self.j_regularized = (self.regularization / (2 * num_training_rows)) * s

    def fit(self,show=False,filenamefig='lastfitresult'):
        num_training_rows = self.data.shape[0]

        global j_list
        j_list = []

        loops = 100

        for i in range(loops):
            self.j = 0
            print("Treinando Loop " + str(i+1) + "/" + str(loops))
            # Pra todos os exemplos
            for row_number in range(num_training_rows):
                #                print("Treinando exemplo " + str(row_number+1) + "/" + str(num_training_rows))
                self.feedforward(row_number)

                self.compute_j(row_number)

                self.compute_errors(row_number)
                self.accumulate_gradients()

            # J Regularizado
            self.compute_j_regularized(num_training_rows)

            # Regularização e Atualização de gradientes
            self.compute_final_gradients(num_training_rows)
            self.update_weights()
            #            print(self.j_regularized)
            #            print(self.j)

            j_list.append(self.j)


        axis_x = range(loops)

        fig, ax = plt.subplots()
        ax.plot(axis_x, j_list)

        ax.set(xlabel='Loop', ylabel='Error)', title='Error vs Loop')
        ax.grid()

        if show:
            plt.show()
        plt.savefig(filenamefig+'.png')

    def savetofile(self, filename='lastneuralnet'):
        # =============================================================================
        # Salva a estrutura e informações da rede em um arquivo txt
        # =============================================================================
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
