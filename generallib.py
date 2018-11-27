
import numpy as np



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

    y = dataframe.iloc[:,y_column]
    classes = np.unique(y.iloc[:])
    num_per_fold = {}
    k_fold_dataframes = []
    classes_index = {}
    counter = {}

    for c in classes:
        total_rows_class = np.sum(y.iloc[:] == c)
        num = int(round(total_rows_class/k_folds))
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

    #data = pd.read_csv("wine.csv", header = None)
    #
    #x = stratified_k_fold(10, 0, data)

def normalization_x(x, v_max, v_min):
    # =============================================================================
    # Função que cálcula a normalização de um valor para o intervalo [-1, 1]
    # =============================================================================
    return (2*(((x-v_min)/(v_max - v_min)))) -1

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

    #data = pd.read_csv("data/wine.csv")
    #data = np.array(data)

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
        initial_weights.append(layer_weights)

    f.close()

    return initial_weights

def read_dataset(filename):
    # not sure if useful
    dataset = []
    f = open(filename,'r')

    for line in f:
        dic_intancia = {}
        line.replace(' ','')
        splits = line.split(';')
        for i in range(0,len(splits)):
            supersplits = splits.split(',')
            lista_valores = []
            for j in range(0,len(supersplits)):
                lista_valores.append(supersplits[j])
            if i ==0:
                dic_intancia['atributos'] = lista_valores
            else:
                dic_intancia['saidas'] = lista_valores
        dataset.append(dic_intancia)

    f.close()
    return dataset

def grad_estacorreto(funcao,gradiente,x,epsilon=0.01, max_delta=0.1 ):

    aproximacao_numerica = (funcao(x+epsilon) - funcao(x-epsilon))/2*epsilon
    delta = gradiente - aproximacao_numerica

    if delta > max_delta:
        return False
    else:
        return True
