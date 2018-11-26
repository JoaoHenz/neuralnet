import sys
import generallib as gl


network_struct = read_networkstructfile(sys.argv[1])
initial_weights = read_initialweightsfile(sys.argv[2])

#le dataset
num_colunaaserpredita = -1
data = pd.read_csv(sys.argv[3]) #abre arquivo
coluna_aserpredita = np.array(pd.DataFrame(data.iloc[:, num_colunaaserpredita]))
dataset = np.array(data.drop(data.columns[num_colunaaserpredita], axis=1))
dataset = gl.normalization(dataset)


#TODO ler os arquivos acima
n = NeuralNet()
## TODO: realizar o aprendizado etc





n.save_finalweights()
