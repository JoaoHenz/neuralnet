import pandas as pd
import numpy as np
import neuralnet as nn
import generallib as gl
from sklearn.metrics import confusion_matrix

y_column = 0
num_kfolds = 10
dataset = pd.read_csv('data/wine.csv')
k_fold_datasets = gl.stratified_k_fold(num_kfolds, y_column, dataset)

def k_fold_training():
    folds_original = gl.stratified_k_fold(num_kfolds, y_column, dataset)
    cm_list = []
    accuracy_list = []

    print("###### K-FOLD RUNNING ######")
    print()
    
    for i in range(num_kfolds):
        folds = folds_original.copy()
        teste = np.array(folds.pop(i))
        treino = folds        
        treino = np.array(pd.concat(folds))
            
        X_train = treino[:, 1:]
        X_train = gl.normalization(X_train)
        y_train = np.transpose(np.array([treino[:, y_column]])-1)
        
        X_test = teste[:, 1:]
        X_test = gl.normalization(X_test)
        y_test = np.transpose(np.array([teste[:, y_column]])-1)
        
        net = nn.NeuralNet(X_train, y_train, num_entrada = X_train.shape[1], num_saida = 3, hidden_lengths = [24], fator_reg = 0.25)
        print()
        print("###### TRAINING - " + str(i+1) + "/" + str(num_kfolds) + " folds ######")
        net.fit(epochs = 30, batch_size = 10)
        
        print()
        print("###### TESTING ######")
        print()
        y_pred = net.classify(X_test)
        
        # Resultado do classificador
        classifier_result = y_pred == y_test
        accuracy = np.sum(classifier_result) / y_test.shape[0]
        print("Parcial Accuracy: " + str("%.3f" % accuracy))
        accuracy_list.append(accuracy)
        cm = confusion_matrix(y_test, y_pred)
        cm_list.append(cm)
        
    return cm_list, accuracy_list

cms, accs = k_fold_training()
acc_mean = np.sum(accs)/len(accs)