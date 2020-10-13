
import pandas as pd
import copy
from itertools import product
from classification_refined import classify
from processing import *
from readfiles import *
from decision import filter_summary
from subgraphclass import make_solver_summary
from metrics import *
''' Data computed for all 5 personality traits at once'''
# labels for the computed subjects, data.index is the subject id
num = 84  # number of nodes in the graph
tri = int(num * (num + 1) * 0.5)  # we want only the upper diagonal parts since everything below diagonal is 0
labels = ['NEOFAC_A', 'NEOFAC_O', 'NEOFAC_C', 'NEOFAC_N', 'NEOFAC_E']
edge_names = ['mean_FA']
big5 = ['Agreeableness', 'Openness', 'Conscientiousness', 'Neuroticism',
       'Extraversion']
mapping = {k: v for k, v in zip(big5, labels)}
mat = np.triu_indices(84)
mews = '/home/skapoor/Thesis/gmwcs-solver'
metrics = ['balanced_accuracy', 'accuracy', 'f1_weighted', 'roc_auc_ovr_weighted']
# note: right now the matrix whole is not scaled, for computing the fscores and correlation coeff it has to be so.
y_train = computed_subjects()
X_train = generate_combined_matrix(tri, list(
   y_train.index))  # need to check indices till here then convert to numpy array
num_strls = X_train.iloc[:, 2 * tri:]
# make_solver_summary(mapping, y_train, big5, mews, X_train, tri, num_strls, avg_thresh=False)
# filter_summary()
y_test = test_subjects()
X_test = generate_test_data(tri, y_test.index)
mean_FA = X_train.iloc[:, :tri]
mean_FA[:, diag_flattened_indices(84)]
mean_FA.iloc[:, diag_flattened_indices(84)].mean()
mean_FA.iloc[:, set(range(3570)).difference(set(diag_flattened_indices(84)))].mean().mean()
mean_FA.max().max()
mean_FA.mean().max()