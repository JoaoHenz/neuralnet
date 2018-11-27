import sys
import generallib as gl


network_struct = read_networkstructfile(sys.argv[1])
initial_weights = read_initialweightsfile(sys.argv[2])

#le dataset
data = pd.read_csv(sys.argv[3]) #abre arquivo
y = np.array(pd.DataFrame(data.iloc[:, y_column]))
dataset = np.array(data.drop(data.columns[y_column], axis=1))
dataset = gl.normalization(dataset)

n = NeuralNet(initial_weights = initial_weights,y= y, num_entrada = network_struct['num_entrada'],num_saida = network_struct['num_saida'],fator_reg = network_struct['fator_reg'],hidden_lengths =network_struct['hidden_lengths'])

## TODO: realizar o aprendizado etc





n.save_finalweights()
