import pickle
from result import Result
import matplotlib
import matplotlib.pyplot as plt


file = open('results2','rb')
result_list = pickle.load(file)
print('\nThis is the list of results:\nNumber of parameters',len(result_list)/18,'\nNumber of Instances:',len(result_list),'\n')

result_list.sort(key = lambda x: x.testname)
i=0

while i < len(result_list):
    testname_atual = result_list[i].testname
    plot_atual = plt.subplot()
    eixox = []
    eixoy = []
    errordata = []
    while i < len(result_list) and result_list[i].testname== testname_atual:
        eixox.append(getattr(result_list[i],testname_atual))
        eixoy.append(getattr(result_list[i],'valordocaminho'))
        errordata.append(getattr(result_list[i],'standarddeviation'))
        print(getattr(result_list[i],'standarddeviation'))
        i+=1
    plot_atual.plot(eixox,eixoy)
    plot_atual.errorbar(eixox,eixoy,errordata,linestyle='None', marker='^')
    plot_atual.set(xlabel= testname_atual,ylabel = 'valor do melhor caminho', title ='Simulação do parâmetro '+testname_atual)
    plt.savefig(testname_atual+'.png')
    plt.clf()
    i+=1

'''
Parametro = "alpha"
t = [0, 1, 2, 3]
y = [10, 20, 30, 40]
fig, ax = plt.subplots()
ax.plot(t, y)
ax.set(xlabel=Parametro, ylabel='Valor do melhor caminho',
       title='Simulação do parâmetro ' + Parametro)
ax.grid()
plt.savefig('fig'+'.png')
'''
