from result import Result
import numpy as np
import pandas as pd
import time
import neuralnet as nn
import math
import statistics
from sklearn.metrics import confusion_matrix

def string_erro_gradients(list_numeric_gradients, list_back_gradients):
    line = ''

    for layer_i in range(list_numeric_gradients.shape[0]-1):
        erro = np.linalg.norm(np.subtract(list_numeric_gradients[layer_i], list_back_gradients[layer_i]))
        line += 'Theta ' + str(layer_i) + ': ' + str(erro)

        if layer_i != list_numeric_gradients.shape[0]-2:
            line += '\n'

    return line

def salvar_dados_corretude(string_numeric, string_back, string_error, filename = "resultado_corretude.txt"):
    f = open(filename, "w+")

    f.write("Gradientes calculados numericamente:\n\n")
    f.write(string_numeric)
    f.write("\n\n")

    f.write("Gradientes calculados por Backpropagation:\n\n")
    f.write(string_back)
    f.write("\n\n")

    f.write("Erro entre gradiente via backprop e gradiente numerico:\n\n")
    f.write(string_error)

    f.close()

def calcula_f1measure(matrix,beta):
    precision = []
    recall = []
    truepositive = []

    for i in range(0,len(matrix)):
        for j in range(0,len(matrix[0])):
            if i == j:
                truepositive.append(matrix[i][j])

    for i in range(0,len(truepositive)):
        precision.append(truepositive[i]/sum(matrix[i,:]))
        recall.append(truepositive[i]/sum(matrix[:,i]))

    for i in range(0,len(recall)):
        if math.isnan(recall[i]):
            recall[i]=1

    for i in range(0,len(precision)):
        if math.isnan(precision[i]):
            precision[i]=1

    recall = sum(recall)/len(recall)
    precision = sum(precision)/len(precision)

    f1_measure = 2 * (precision * recall)/(precision + recall)
    if math.isnan(f1_measure):
        f1_measure=1
    return f1_measure

def f1measure_emlista(matrix_list, beta):
    f1measure_mediatotal = []
    for i in range(0,len(matrix_list)):
        f1measure_mediatotal.append(calcula_f1measure(matrix_list[i],beta))
    return (sum(f1measure_mediatotal)/len(f1measure_mediatotal))

def mean_acc(acc_list):
    return (sum(acc_list)/len(acc_list))

def stratified_k_fold(k_folds, y_column, dataframe):
    # =============================================================================
    # K-fold Estratificado
    # Retorna uma lista, cada item sendo um dataframe (fold)
    # =============================================================================
    """
    1) Calcula quantas instâncias de cada classe devem ser inseridos
    em cada fold;

    2) Cria um fold por vez, inserindo N instâncias de cada classe no fold;

    3) No último fold, insere as instâncias que sobraram.

    *Qualquer instância do dataframe original deve estar somente em um único fold.
    """
    dataframe = dataframe.reset_index(drop = True)
    y = dataframe.iloc[:,y_column]
    classes = np.unique(y.iloc[:])
    num_per_fold = {}
    k_fold_dataframes = []
    classes_index = {}
    counter = {}

    for c in classes:
        total_rows_class = np.sum(y.iloc[:] == c)
        num = int(math.floor(total_rows_class/k_folds))
        num_per_fold[c] = num
        index = y[y.iloc[:] == c].index
        classes_index[c] = index
        counter[c] = 0

    for k in range(k_folds-1):
        index = np.array([], dtype = "int64")
        for c in classes:
            num = num_per_fold[c]
            limit = counter[c] + num
            index = np.concatenate((index, classes_index[c][counter[c]:limit].values))
            counter[c] += num
        k_fold_dataframes.append(dataframe.iloc[index])

    index = np.array([], dtype = "int64")
    for c in classes:
        limit = classes_index[c].shape[0]
        index = np.concatenate((index, classes_index[c][counter[c]:limit].values))
        counter[c] += num
    k_fold_dataframes.append(dataframe.iloc[index])

    return k_fold_dataframes

def normalization_x(x, v_max, v_min):
    # =============================================================================
    # Função que cálcula a normalização de um valor para o intervalo [-1, 1]
    # =============================================================================
    if v_max - v_min != 0:
        return (2*(((x-v_min)/(v_max - v_min)))) -1
    else:
        return x

def normalization(data):
    # =============================================================================
    # Normaliza um dataset para o intervalo [-1, 1]. Deve ser passado como parâmetro
    # um dataset em numpy array
    # =============================================================================
    num_columns = data.shape[1]
    new_data = np.empty_like(data)

    for column_i in range(num_columns):
        v_max = np.max(data[:, column_i])
        v_min = np.min(data[:, column_i])
        new_data[:, column_i] = normalization_x(data[:, column_i], v_max, v_min)

    return new_data

def read_networkstructfile(filename):
    network_struct = {}
    list_stuff = []

    f = open(filename,'r')

    for line in f:
        list_stuff.append(float(line))

    network_struct['fator_reg'] = list_stuff[0]
    network_struct['num_entrada'] = int(list_stuff[1])
    network_struct['num_saida'] = int(list_stuff[len(list_stuff)-1])

    hidden_lengths = []
    for i in range(2,len(list_stuff)-1):
        hidden_lengths.append(int(list_stuff[i]))
    network_struct['hidden_lengths'] = hidden_lengths
    f.close()

    return network_struct

def read_initialweightsfile(filename):
    initial_weights = []

    f = open(filename,'r')

    for line in f: #cada layer
        line.replace(' ','')
        layer_weights = []
        nodes = line.split(';')
        for i in range(0,len(nodes)): #para cada neuronio da layer
            weights = nodes[i].split(',')
            node_weights = []
            for i in range(0,len(weights)): #para cada peso da layer
                node_weights.append(float(weights[i]))
            layer_weights.append(node_weights)
        initial_weights.append(np.array(layer_weights))

    f.close()

    return initial_weights

def read_dataset(filename):
    # not sure if useful
    dataset = []
    f = open(filename,'r')

    for line in f:
        dic_instancia = {}
        line.replace(' ','')
        splits = line.split(';')
        for i in range(0,len(splits)):
            supersplits = splits[i].split(',')
            lista_valores = []
            for j in range(0,len(supersplits)):
                lista_valores.append(float(supersplits[j]))
            if i ==0:
                dic_instancia['atributos'] = lista_valores
            else:
                dic_instancia['saidas'] = lista_valores
        dataset.append(dic_instancia)

    f.close()

    X = []
    y = []

    for i in range(len(dataset)):
        d = dataset[i]
        att = d["atributos"]
        out = d["saidas"]

        X.append(att)
        y.append(out)

    X = np.array(X)
    y = np.array(y)

    return X, y

def transform_y(dataset, y_column):
    df = dataset.copy()
    classes_transform = {}
    classes = np.unique(df.iloc[:,y_column])

    for i in range(len(classes)):
        cl = classes[i]
        classes_transform[i] = cl
        df.iloc[:,y_column] = df.iloc[:,y_column].replace(cl, i)

    return df, classes_transform

def k_fold_training(k, dataset, y_column, epochs, batch_size, hidden_lengths = [24, 24], fator_reg = 0.25, alpha = 0.1, verbose = True):
    start_runtime_total = time.time()
    folds_original = stratified_k_fold(k, y_column, dataset)
    cm_list = []
    acc_list = []
    j_list = []
    
    if verbose:
        print("###### K-FOLD RUNNING ######")
        print()

    for i in range(k):
        folds = folds_original.copy()
        teste = np.array(folds.pop(i))
        treino = folds
        treino = np.array(pd.concat(folds))

        X_train = np.delete(treino, y_column, 1)
        X_train = normalization(X_train)
        y_train = np.transpose(np.array([treino[:, y_column]]))

        X_test = np.delete(teste, y_column, 1)
        X_test = normalization(X_test)
        y_test = np.transpose(np.array([teste[:, y_column]]))

        num_input = dataset.shape[1]-1
        num_output = len(np.unique(dataset.iloc[:,y_column]))

        net = nn.NeuralNet(X_train, 
                           y_train,
                           learning_rate = alpha,
                           num_entrada = num_input, 
                           num_saida = num_output, 
                           hidden_lengths = hidden_lengths, 
                           fator_reg = fator_reg)
        if verbose:
            print()
            print("###### TRAINING - " + str(i+1) + "/" + str(k) + " folds ######")
        j = net.fit(epochs = epochs, batch_size = batch_size, verbose = verbose)
        j_list.append(j)
        
        if verbose:
            print()
            print("###### TESTING ######")
            print()
        y_pred = net.classify(X_test)

        # Resultado do classificador
        classifier_result = y_pred == y_test
        accuracy = np.sum(classifier_result) / y_test.shape[0]
        if verbose:
            print("Parcial Accuracy: " + str("%.3f" % accuracy))
        acc_list.append(accuracy)
        cm = confusion_matrix(y_test, y_pred)
        cm_list.append(cm)
        
    f1 = f1measure_emlista(np.array(cm_list), 1)
    acc_std = statistics.stdev(acc_list)
    acc_mean = statistics.mean(acc_list)
    j_std = statistics.stdev(j_list)
    j_mean = statistics.mean(j_list)

    result = Result(f1, acc_mean, acc_std, j_mean, j_std)
    
    final_runtime_total = time.time() - start_runtime_total
    if verbose:
        print(str("Total time: %.4fs" % final_runtime_total))
    
    return result
