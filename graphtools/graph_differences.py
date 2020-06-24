"""
Experiments to do:
We make one networkx Graph for all sub
1. Node weights
    a. 1 or 0 for every node
    b. Maximum of the edge score
2. Edge weights
    a. Comprehensive score containing
        i. Pearson Correlation of the edge
        ii. Fscore of the edge
        iii. Random Forest score of the edge

Store the values in the a json dictionary
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from processing import generate_combined_matrix, hist_fscore, hist_correlation
from readfiles import computed_subjects
import matplotlib.pyplot as plt
import time
import datetime
import json
from paramopt import get_distributions
from graphweights import train_with_best_params
from classification import data_splitting
import networkx as nx
import random
#%%
data = computed_subjects()  # labels for the computed subjects
y = np.array(data['NEOFAC_A'] >= data['NEOFAC_A'].median()).astype(int)
num = 84  # number of nodes in the graph
tri = int(num * (num + 1) * 0.5)  # we want only the upper diagonal parts since everything below diagonal is 0
whole = generate_combined_matrix(tri, data.index)
# Values of C parameter of SVM

big5 = ['Agreeableness', 'Openness', 'Conscientiousness', 'Neuroticism',
        'Extraversion']
labels = ['NEOFAC_A', 'NEOFAC_O', 'NEOFAC_C', 'NEOFAC_N', 'NEOFAC_E']
edge_names = ['mean_FA', 'mean strl', 'num streamlines']
fscores = hist_fscore(data, whole, labels, big5, edge_names, tri)
#%%
corr = hist_correlation(data, whole, labels, edge_names,big5, tri)

#%%
mat = np.triu_indices(84)

with open('outputs/combined_params.json', 'r') as f:
    best_params = json.load(f)

    for i in range(5):  # different labels
        # print(labels[i], ':', big5[i])
        for per in [5, 10]: # we actually see that keeping 5-10% of the features is the best option
            val = np.percentile(fscores[i].flattten(), 100 - per)
            index = np.where(fscores[i].flatten() >= val)
            #for choice in ['qcut', 'median', 'throw median'
            # Let's say we only choose the throw median choice, because it is the one that makes more sense
            choice = 'throw median'
            X, y = data_splitting(choice, i, index, data, whole, labels) # this X is for random forests training
            params = best_params['RF'][big5[i]][per][choice]
            feature_imp = train_with_best_params('RF', params, X, y)
            feature_imp = np.reshape(feature_imp, (feature_imp.shape[0]//3, 3))
            g1 = nx.Graph()
            edges = {(0, 1)}
            # edges = {}
            edge_attributes = {}
            node_attributes = {x: random.randint(-10, 10) for x in range(84)}
            for j in range(len(mat[0])):
                edges.add((mat[0][j], mat[1][j]))
                edge_attributes[(mat[0][j], mat[1][j])] = [np.mean(fscores[i,:, j]),
                                                           np.mean(corr[i,:,j]),np.mean(feature_imp[j,:]) ]  # then we should have just one graph for all subjects
                # this graph is then needed to be put into the solver in order to get the maximum edge weighted subgraph
            g1.add_nodes_from(range(84))
            g1.add_edges_from(edges)
            # nx.set_node_attributes(g1, nodes, 'nodes')
            nx.set_edge_attributes(g1, edge_attributes, 'edges')
