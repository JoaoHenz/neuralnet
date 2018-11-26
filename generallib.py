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
    pass
    ## TODO:

def read_initialweightsfile(filename):
    pass
    ## TODO:
