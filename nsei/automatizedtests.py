from wtapassive import *
import os
import pickle
import statistics

from result import *



# 0.5 1.2 50 80 1000.0
#número de hidden layers,length das hidden layers, taxa de aprendizado, taxa de regularização
variacoes = [
    [1,5,10,15,20],
    [1,5,10,15,20],
    [0.1,0.5,1.0,1.5,2.0],
]
file = open('results','wb')
file.truncate(0)

med_len = 15
resultlist = []
for filename in os.listdir('./data'):
    print('\ncomeçando processamento de ',filename,'.......')

    for i in range(len(variacoes[0])):
        med_lista = []
        for j in range(med_len):
            result = onetest('./wta_instances/'+filename,alpha= variacoes[0][i])
            result.filename = filename
            result.testname = 'alpha'
            med_lista.append(result)
            print('alpha:',result.valordocaminho)
        print(':::fim de instância!')
        result = med_lista[0]
        result.valordocaminho = sum(x.valordocaminho for x in med_lista)/med_len
        result.standarddeviation = statistics.stdev(x.valordocaminho for x in med_lista)
        print('valor do alpha:',result.alpha)
        resultlist.append(result)
        print('média do alpha:',result.valordocaminho)

print('\ncomeçando a salvar os resultados.......')
pickle.dump(resultlist,file)
print(':::resultados foram salvos com sucesso!')
