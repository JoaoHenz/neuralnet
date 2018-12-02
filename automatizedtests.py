import pickle
import result
import time
import pandas as pd
import generallib as gl

y_column = 0
filename = "data/wine.csv"
dataset = pd.read_csv(filename)
dataset, transformation = gl.transform_y(dataset, y_column)

num_kfolds = 10
epochs = 5
batch_size = 32

# 0.5 1.2 50 80 1000.0
#número de hidden layers,length das hidden layers, taxa de regularizacao, taxa de aprendizado
variacoes = [
    [1, 2, 3, 4, 5, 6],
    [5, 10, 15, 24],
    [0.1, 0.25, 0.5],
    [0.1, 0.3, 0.5]
]

file = open('output/results','wb')
file.truncate(0)

resultlist = []

print('\ncomeçando processamento de ',filename,'.......')

total_variations = len(variacoes[0])*len(variacoes[1])*len(variacoes[2])*len(variacoes[3])

start_runtime_total = time.time()
atual = 1
for i in range(len(variacoes[0])):
    for j in range(len(variacoes[1])):
        network = [variacoes[1][j]]*variacoes[0][i]
        for k in range(len(variacoes[2])):
            for l in range(len(variacoes[3])):
                print("Variacao " + str(atual) + "/" + str(total_variations))
                result = gl.k_fold_training(num_kfolds, 
                                            dataset, 
                                            y_column, 
                                            epochs, 
                                            batch_size,
                                            hidden_lengths = network,
                                            fator_reg = variacoes[2][k],
                                            alpha = variacoes[3][l],
                                            verbose = False
                                            )
                result.num_hidden = variacoes[0][i]
                result.nodes_per_hidden = variacoes[1][j]
                result.fator_reg = variacoes[2][k]
                result.learning_rate = variacoes[3][l]
                result.filename = filename
                resultlist.append(result)
                atual += 1

final_runtime_total = time.time() - start_runtime_total
print(str("Total time: %.4fs" % final_runtime_total))

pickle.dump(resultlist, file)
print(':::resultados foram salvos com sucesso!')