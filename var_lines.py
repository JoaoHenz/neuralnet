import pandas as pd
from result import Result
import generallib as gl
import matplotlib.pyplot as plt

y_column = -1
filename = "data/pima.csv"
dataset = pd.read_csv(filename)
dataset, transformation = gl.transform_y(dataset, y_column)
dataset = dataset.sample(frac=1).reset_index(drop=True)

num_kfolds = 10
epochs = 30
batch_size = 10
network = [5]
fator_reg = 0.25
alpha = 0.3

percent_lines = 0.05
total = int(1/percent_lines)-4

num_rows_data = dataset.shape[0]

variacoes = [0]*total
j_mean_values = [0]*total
j_std_values = [0]*total

x = 0.2
for i in range(total):
    variacoes[i] = int(num_rows_data*x)
    x += percent_lines

for i in range(len(variacoes)):
    print("Calculando " + str(i+1) + "/" + str(total))
    num_lines = variacoes[i]
    result = gl.k_fold_training(num_kfolds, 
                                dataset.iloc[:num_lines,:], 
                                y_column, 
                                epochs, 
                                batch_size,
                                hidden_lengths = network,
                                fator_reg = fator_reg,
                                alpha = alpha,
                                verbose = False
                                )
    j_mean_values[i] = result.j_mean
    j_std_values[i] = result.j_std

nome_arquivo = filename.replace(".csv", "")
nome_arquivo = nome_arquivo.replace("data/", "")
plot_atual = plt.subplot()
plot_atual.plot(variacoes,j_mean_values)
plot_atual.errorbar(variacoes, j_mean_values, j_std_values, linestyle='None', marker='^')
plot_atual.set(xlabel = 'Qtd. Linhas', ylabel = 'J', title = filename)
plt.savefig('output/' + nome_arquivo + '.png')
plt.clf()