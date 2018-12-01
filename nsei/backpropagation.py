import sys
import generallib as gl
import pandas as pd
import numpy as np
from neuralnet import *

network_struct = gl.read_networkstructfile(sys.argv[1])
initial_weights = gl.read_initialweightsfile(sys.argv[2])

#le dataset
y_column = -1
data = pd.read_csv(sys.argv[3]) #abre arquivo
y = np.array(pd.DataFrame(data.iloc[:, y_column]))
dataset = np.array(data.drop(data.columns[y_column], axis=1))
dataset = gl.normalization(dataset)

n = NeuralNet(dataset= dataset,initial_weights = initial_weights,y= y, num_entrada = network_struct['num_entrada'],num_saida = network_struct['num_saida'],fator_reg = network_struct['fator_reg'],hidden_lengths =network_struct['hidden_lengths'])
n.fit() #?
#n.savetofile()
result = n.classify(dataset) #?
result = (result > 0.5)
expected = (y > 0.5)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(expected, result) # ? resultado do teste com a rede neural?

n.save_finalweights()
