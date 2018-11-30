import pandas as pd
import numpy as np
import generallib as gl
import neuralnet as nn

### CLASSIFICAR DATASETS ###
#correcao = True
correcao = False

if correcao:
    network_file = "network.txt"
    weights_file = "initial_weights.txt"
    dataset_file = "dataset.txt"
    
    estrutura_rede = gl.read_networkstructfile(network_file)
    pesos_iniciais = gl.read_initialweightsfile(weights_file)
#    X_train = np.array([[0.32000, 0.68000], [0.83000, 0.02000]])
#    y_train = np.array([[0.75000, 0.98000], [0.75000, 0.28000]])
    X_train = np.array([[0.13000], [0.42000]])
    y_train = np.array([[0.90000], [0.23000]])
    
    net = nn.NeuralNet(X_train, 
                       y_train, 
                       hidden_lengths = estrutura_rede["hidden_lengths"], 
                       fator_reg = estrutura_rede["fator_reg"],
                       num_entrada = estrutura_rede["num_saida"],
                       num_saida = estrutura_rede["num_entrada"],
                       initial_weights = pesos_iniciais,
                       numeric = True
                       )
    
    net.fit(verbose = False,
           save_gradients = True
           )
        
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