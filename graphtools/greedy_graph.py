#!/usr/bin/env python
# coding: utf-8

# In[1]:


import copy
from itertools import product
from classification_refined import classify
from processing import *
from readfiles import *
from decision import filter_summary
from subgraphclass import make_solver_summary
from sklearn.model_selection import train_test_split
from classification_refined import *
import networkx as nx
from readfiles import *
from metrics import *
import numpy as np
import networkx as nx
import time 
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from readfiles import corresp_label_file
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
import concurrent.futures
import time
from functools import partial
import os


# In[2]:



def make_plot(node_names, target):
    cm = plt.cm.get_cmap('cool')
    fig, ax = plt.subplots(len(node_names.keys()), figsize=(60,60))
    for max_num_nodes, i in zip(node_names.keys(), range(len(node_names.keys()))):
        counts, bins, patches = ax[i].hist(node_names[max_num_nodes], 
                                           bins=len(pd.Series(node_names[max_num_nodes]).unique()))
        
        for c, p in zip(counts, patches):
            plt.setp(p, 'facecolor', cm(c/5))
    # Set the ticks to be at the edges of the bins.
        ax[i].set_title(f'{target} {max_num_nodes} nodes')
        ax[i].tick_params(labelrotation=55)
        ax[i].set_xticks(range(len(pd.Series(node_names[max_num_nodes]).unique())))
    plt.tight_layout()
    plt.savefig(f'outputs/{target}')
    #plt.show()


# In[3]:


num = 84  # number of nodes in the graph
tri = int(num * (num + 1) * 0.5)  # we want only the upper diagonal due to symettry of connections
mat = np.triu_indices(84)
mews = '/home/skapoor/Thesis/gmwcs-solver'


# In[4]:


num_nodes = [5,7,10,12,15,20]


# In[5]:


feature, edge, solver_node_wts = 'num_streamlines', 'pearson','const'
val, thresh = -0.01, 0
choice, classifier, refit_metric, feature_selection = 'random', 'ridge_reg', 'balanced_accuracy', 'baseline'
baseline_cases, self_loops = set(), False


# In[6]:


labels = ['ReadEng_Unadj', 'ReadEng_AgeAdj',  'PicVocab_Unadj', 
          'PicVocab_AgeAdj','ProcSpeed_Unadj', 'ProcSpeed_AgeAdj']


# In[7]:


# note: right now the matrix whole is not scaled, for computing the fscores and correlation coeff it has to be so.
y_tr, y_te = computed_subjects(), test_subjects()

labels = [label for label in labels if label in y_tr.columns ]

X_tr = generate_combined_matrix(tri, list(y_tr.index))  # need to check indices till here then convert to numpy array
X_te = generate_test_data(tri, y_te.index)
X_t, X_te = edge_filtering(feature, X_tr, X_te)
X = X_tr.append(X_te)

for target in labels:
    print(f'{target}')
    y_train_l, y_test_l = y_tr[target], y_te[target]
    y =  y_train_l.append(y_test_l)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state = 22)
    y_binned = pd.qcut(y, 5, labels=False, retbins=True)[0]

    skf.get_n_splits(X, y_binned)
    print(skf, '-'*100)
    results_solver = []
    avg_thresh, self_loops = False, False
    #nodes = []
    node_names = {k:[] for k in num_nodes}
    param_grid = [{'alpha': [10, 1e3]}]
    for max_num_nodes in num_nodes:
        print(f"Number of nodes {max_num_nodes}")
        i =0 
        for train_index, test_index in skf.split(X, y_binned):
            i+=1
            start = time.time()
            #print("TRAIN:",len(train_index), train_index, "TEST:", len(test_index),test_index)
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            med = y_train.median()
            y_train_l = pd.qcut(y_train, 5, labels=False, retbins=True)[0]
            # we need to pass the non-binned values for effective pearson correlation calc.
            # print('The number of training subjects which are to be removed:', sum(y_train_l == 2))
            y_train_l = y_train_l[y_train_l != 2]
            y_train_l = y_train_l // 3  # binarizing the values by removing the middle quartile
            y_test_l = y_test >= med 
            X_train_l = X_train.loc[y_train_l.index]
            assert list(X_train_l.index) == list(y_train_l.index)


            X_train_l, X_test_l, arr = process_raw(X_train_l, X_test, y_train_l, edge)
            graph = np.zeros((84,84))

            for k,j ,l in zip(np.triu_indices(84)[0], np.triu_indices(84)[1], range(tri)):
                graph[k,j] = abs(arr.iloc[l]) #  taking the absolute value of the pearson correlation 
            # graph is currently upper triangular and needs to be converted into full matrix
            graph = graph + graph.transpose() - 2 * np.diag(graph.diagonal()) # excluding the diagonal
            graph = pd.DataFrame(graph, index = corresp_label_file('fs_default.txt').values(), 
                         columns = corresp_label_file('fs_default.txt').values())

            while len(graph.iloc[0]) > max_num_nodes:
                node = graph.sum(axis=1).argmin()
                graph = graph.drop([graph.loc[node].name], axis = 0) # returns the row as series
                graph = graph.drop(graph.loc[:, node].name, axis = 1) # graph.loc[:, node].name
            node_names[max_num_nodes].extend(list(graph.columns))
            end = time.time()
            t = end - start
            print(f'Time for fold {i}:', round(t,3))
    make_plot(node_names, target )

