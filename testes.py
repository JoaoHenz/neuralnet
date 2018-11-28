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


#n = NeuralNet(X_train, y_train, num_entrada = X.shape[1], num_saida = 3, hidden_lengths = [8], fator_reg=0.25)
##n.savetofile(filename='lastneuralnet')
#n.fit() #
##n.savetofile(filename='lastneuralnet2')
##
#result = n.classify(X_test)
###
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, result)
#acerto = cm[0,0] + cm [1, 1]
#erro = cm[0,1] + cm [1, 0]
#
#acc = acerto/(acerto+erro)






#theta1 = np.array([[0.4, 0.1], [0.3, 0.2]])
#theta2 = np.array([[0.7, 0.5, 0.6]])
#
#dataset = np.array([[0.13, 0.9], [0.42, 0.23]])
#y = np.transpose(np.array([dataset[:,-1]]))
#dataset = np.transpose(np.array([dataset[:,0]]))
#
#pesos = np.array([theta1, theta2])
#
##n = NeuralNet(dataset, y, initial_weights = pesos)
#n = NeuralNet(dataset, y)
#
##n.savetofile(filename='lastneuralnet')
#n.fit() #
##n.savetofile(filename='lastneuralnet2')


##le dataset
#y_column = -1
#data = pd.read_csv("/mnt/Data/neuralnet/data/Churn_Modelling_Edited.csv")
#
#y = np.array(pd.DataFrame(data.iloc[:, y_column]))
##
#dataset = np.array(data.drop(data.columns[y_column], axis=1))

#dataset = gl.normalization(dataset)
#


## Importing the dataset
#dataset = pd.read_csv('data/wine.csv')
#X = dataset.iloc[:, 3:13].values
#y = dataset.iloc[:, 13].values
#
## Encoding categorical data
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#labelencoder_X_1 = LabelEncoder()
#X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
#labelencoder_X_2 = LabelEncoder()
#X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
#onehotencoder = OneHotEncoder(categorical_features = [1])
#X = onehotencoder.fit_transform(X).toarray()
#X = X[:, 1:]
#
## Splitting the dataset into the Training set and Test set
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
#
## Feature Scaling
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)
#
#y_train = np.transpose(np.array([y_train]))
#y_test = np.transpose(np.array([y_test]))
#
#n = NeuralNet(X_train, y_train, num_entrada = 11, num_saida = 2, hidden_lengths = [8,8])
##n.savetofile(filename='lastneuralnet')
#n.fit() #
##n.savetofile(filename='lastneuralnet2')
##
#result = n.classify(X_test)
##
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, result)
#acerto = cm[0,0] + cm [1, 1]
#erro = cm[0,1] + cm [1, 0]
#
#acc = acerto/(acerto+erro)

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
