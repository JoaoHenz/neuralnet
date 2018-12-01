import pandas as pd
import numpy as np
import generallib as gl
import neuralnet as nn

### CLASSIFICAR DATASETS ###
correcao = True
#correcao = False

if correcao:
    network_file = "network.txt"
    weights_file = "initial_weights.txt"
    dataset_file = "dataset.txt"
    
    estrutura_rede = gl.read_networkstructfile(network_file)
    pesos_iniciais = gl.read_initialweightsfile(weights_file)
    X_train, y_train = gl.read_dataset(dataset_file)
    
    net = nn.NeuralNet(X_train, 
                       y_train, 
                       hidden_lengths = estrutura_rede["hidden_lengths"], 
                       fator_reg = estrutura_rede["fator_reg"],
                       num_entrada = estrutura_rede["num_saida"],
                       num_saida = estrutura_rede["num_entrada"],
                       initial_weights = pesos_iniciais,
                       numeric = True
                       )
        
    gradients_numeric = net.fit_numeric(e = 0.0000010000)
    string_numeric = net.string_gradients(gradients_numeric, 10)
    
    net = nn.NeuralNet(X_train, 
                       y_train, 
                       hidden_lengths = estrutura_rede["hidden_lengths"], 
                       fator_reg = estrutura_rede["fator_reg"],
                       num_entrada = estrutura_rede["num_saida"],
                       num_saida = estrutura_rede["num_entrada"],
                       initial_weights = pesos_iniciais,
                       numeric = True
                       )
    gradients_backpropagation = net.fit_back()
    string_back = net.string_gradients(gradients_backpropagation, 10)
    
    erro = gl.string_erro_gradients(gradients_numeric, gradients_backpropagation)
    gl.salvar_dados_corretude(string_numeric, string_back, erro)
    
    net = nn.NeuralNet(X_train, 
                       y_train, 
                       hidden_lengths = estrutura_rede["hidden_lengths"], 
                       fator_reg = estrutura_rede["fator_reg"],
                       num_entrada = estrutura_rede["num_saida"],
                       num_saida = estrutura_rede["num_entrada"],
                       initial_weights = pesos_iniciais,
                       numeric = True
                       )
    net.fit(save_gradients = True, verbose = False)
    
else:
    y_column = 0
    dataset = pd.read_csv('data/cancer.csv')
    dataset, transformation = gl.transform_y(dataset, y_column)
    beta = 1
    
    num_kfolds = 10
    epochs = 10
    batch_size = 10
    
    acc, cm = gl.k_fold_training(num_kfolds, dataset, y_column, epochs, batch_size)
    
    f1_measure = gl.f1measure_emlista(np.array(cm), beta)
    
    print("\n\n########### RESULTADOS FINAIS ###########\n")
    
    print("F1 Measure: %.3f" % f1_measure)
    mean_acc = gl.mean_acc(acc)
    print("Accuracy: " + str("%.3f" % mean_acc))