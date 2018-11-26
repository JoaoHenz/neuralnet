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




    # =============================================================================
    # Feedforward da rede para uma instância de exemplo
    # =============================================================================
#    def feedforward(self):
#        # Input Layer - Coloca o vetor de entrada "i" como sendo a matriz de ativação da camada 0
#        self.activations[0][1:] = np.transpose((np.array([self.data.iloc[0,:]])))
##        self.activations[0] = np.append(self.activations[0], [1])
#
#        for layer_i in (range(self.num_layers))[1:-1]:
#            self.activations[layer_i] = self.sigmoid(np.dot(self.weights[layer_i-1], self.activations[layer_i-1]))
#            self.activations[layer_i] = np.append(self.activations[layer_i], [1])
#
#        # Output - Ativa camada de saída
#        self.activations[-1] = self.sigmoid(np.dot(self.weights[-2], self.activations[layer_i-2]))



#    # =============================================================================
#    # Baseado no slide 132 da aula 14
#    # =============================================================================
#    def backpropagation(self):
#        # TODO: incompleto
#
#        # Cálculo dos deltas
#        # Camada de saída
#        # TODO: ta certo?
#        predict = self.activations[-1]
#        output = np.array(pd.DataFrame(self.y.iloc[0, :]))
#        self.errors[-1] = predict - output
#
#        # Cálculo dos deltas para hidden layers
#        for layer_i in reversed((range(self.num_layers))[1:-1]):
#            weights_transposed = np.transpose(self.weights[layer_i])
#            # Calcula em três partes o delta da camada
#            part1 = np.dot(weights_transposed, self.errors[layer_i+1])
#            part2 = np.multiply(self.activations[layer_i], (1-self.activations[layer_i]))
#            self.errors[layer_i] = np.multiply(part1, part2)
#
#        # Cálculo dos gradientes
#        for layer_i in reversed((range(self.num_layers))[:-1]):
#            activations_transposed = np.transpose(self.activations[layer_i])
#
#            part1 = np.dot(activations_transposed, self.errors[layer_i+1][1:])
#            self.gradients[layer_i] = np.add(part1 + self.gradients[layer_i])
#
#        # Cálculo dos gradientes finais
#        for layer_i in reversed((range(self.num_layers))[:-1]):
#            matrix_p = np.multiply(self.regularization, self.weights[layer_i])
#            # zerar primeira coluna
#
#            self.gradients[layer_i] = np.add(part1 + self.gradients[layer_i])
