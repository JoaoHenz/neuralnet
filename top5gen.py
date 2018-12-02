import pickle
from result import Result
import numpy as np

file = open('output/results','rb')
result_list = pickle.load(file)

res = []
res2 = []
for result in result_list:
    res.append(result.acc_mean)
    res2.append(result)

res = np.array(res)
res2 = np.array(res2)

top = 5
melhores = []
for i in range(top):
    index = np.argmax(res)
    melhor = res2[index]
    melhores.append(melhor)
    res = np.delete(res, index)
    res2 = np.delete(res2, index)


f = open('output/top5.txt', "w+")
f.write("Top 5 parametros para o dataset " + melhores[0].filename + ":\n\n")
for resultado in melhores:
    f.write("F1: " + str(resultado.f1) + "\n")
    f.write("Acc Mean: " + str(resultado.acc_mean) + "\n")
    f.write("Acc Std: " + str(resultado.acc_std) + "\n")
    f.write("J Mean: " + str(resultado.j_mean) + "\n")
    f.write("J Std: " + str(resultado.j_std) + "\n")
    f.write("Num Hidden Layers: " + str(resultado.num_hidden) + "\n")
    f.write("Nodes/Hidden Layer: " + str(resultado.nodes_per_hidden) + "\n")
    f.write("Learning Rate: " + str(resultado.learning_rate) + "\n")
    f.write("Fator Reg: " + str(resultado.fator_reg) + "\n")
    f.write('\n\n\n')
f.close()
