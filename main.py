import pandas as pd
import numpy as np
import generallib as gl

y_column = 0
dataset = pd.read_csv('data/cancer.csv')
dataset, transformation = gl.transform_y(dataset, y_column)

num_kfolds = 10
epochs = 10
batch_size = 10

acc, cm = gl.k_fold_training(num_kfolds, dataset, y_column, epochs, batch_size)

f1_measure = gl.f1measure_emlista(np.array(cm), 1)

print("\n\n########### RESULTADOS FINAIS ###########\n")

print("F1 Measure: %.3f" % f1_measure)
mean_acc = gl.mean_acc(acc)
print("Accuracy: " + str("%.3f" % mean_acc))