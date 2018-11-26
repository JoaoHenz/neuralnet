from neuralnet import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import generallib as gl



y_column = -1 #?

data = pd.read_csv("data/Wine.csv") #abre arquivo
coluna_aserpredita = np.array(pd.DataFrame(data.iloc[:, y_column])) #y é a coluna que eu quero prever
dataset = np.array(data.drop(data.columns[y_column], axis=1)) #cria dataset com os dados do arquivo
dataset = gl.normalization(dataset)

n = NeuralNet(dataset, coluna_aserpredita) #cria a rede neural.... já treina ela?
n.fit() #?
n.savetofile()

result = n.classify(dataset) #?
result = (result > 0.5)
expected = (y > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(expected, result) # ? resultado do teste com a rede neural?

#n.save_to_txt("test2.txt")
