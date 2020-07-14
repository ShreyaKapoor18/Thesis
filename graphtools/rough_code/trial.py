"""
===========================================================================
Graph classification on a randomly generated dataset of Erdos-Renyi graphs.
===========================================================================
Script makes use of :class:`grakel.Graph` and :class:`grakel.ShortestPath`
"""
from __future__ import print_function
print(__doc__)

import numpy as np

from random import random
from processing import generate_combined_matrix, hist_fscore
from readfiles import computed_subjects
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from grakel.kernels import PropagationAttr
from grakel import Graph
from grakel.kernels import ShortestPath, SvmTheta
import numpy as np
import networkx as nx
import random

from grakel.utils import graph_from_networkx

# Generates 3 sets of Erdos-Renyi graphs. Each edge is included in the graph with probability p
# independent from every other edge. The probability p is set equal to 0.25, 0.5 and 0.75 for
# the graphs of the 1st, 2nd and 3rd set, respectivery
#%%
data = computed_subjects()  # labels for the computed subjects
y = np.array(data['NEOFAC_A']>=data['NEOFAC_A'].median()).astype(int)

num = 84  # number of nodes in the graph
tri = int(num * (num + 1) * 0.5)  # we want only the upper diagonal parts since everything below diagonal is 0
whole = np.array(generate_combined_matrix(tri, data.index))
#%%
# Values of C parameter of SVM
wholex = np.reshape(whole,(whole.shape[0], whole.shape[1]//3, 3))
mat = np.triu_indices(84)
G = []
nodes = {x: 0 for x in range(84)}
for i in range(len(wholex)):
    edges = {(0,0)}
    #edges = {}
    edge_attributes = {}
    for j in range(len(mat[0])):
        edges.add((mat[0][j], mat[1][j])) # first feature is mean fa between nodes
        #edges[(mat[0][j], mat[1][j])] = 1
        edge_attributes[(mat[0][j], mat[1][j])] = wholex[i,j,0]* 100

    #print (edges, edge_attributes)
    G.append(Graph(edges, node_labels=nodes, edge_labels= edge_attributes))

# Splits the dataset into a training and a test set
G_train, G_test, y_train, y_test = train_test_split(G, y, test_size=0.1, random_state=42)

# Uses the shortest path kernel to generate the kernel matrices
gk = SvmTheta()
K_train = gk.fit_transform(G_train)
K_test = gk.transform(G_test)

# Uses the SVM classifier to perform classification
clf = SVC(kernel="precomputed")
clf.fit(K_train, y_train)
y_pred = clf.predict(K_test)

# Computes and prints the classification accuracy
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", str(round(acc*100, 2)) + "%")
#%%
from grakel.datasets import fetch_dataset
from grakel.kernels import WeisfeilerLehman, VertexHistogram, SvmTheta

# Splits the dataset into a training and a test set
G_train, G_test, y_train, y_test = train_test_split(G, y, test_size=0.3, random_state=22)

# Uses the Weisfeiler-Lehman subtree kernel to generate the kernel matrices
gk = WeisfeilerLehman(n_iter=100, base_graph_kernel=SvmTheta)
K_train = gk.fit_transform(G_train)
K_test = gk.transform(G_test)

# Uses the SVM classifier to perform classification
clf = SVC(kernel="precomputed")
clf.fit(K_train, y_train)
y_pred = clf.predict(K_test)

# Computes and prints the classification accuracy
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", str(round(acc*100, 2)) + "%")
#%%
G = []
nodes = {x: 0 for x in range(84)}
node = range(84)
for i in range(len(wholex)):
    g1 = nx.Graph()
    edges = {(0,0)}
    #edges = {}
    edge_attributes = {}
    for j in range(len(mat[0])):
        edges.add((mat[0][j], mat[1][j])) # first feature is mean fa between nodes
        #edges[(mat[0][j], mat[1][j])] = 1
        edge_attributes[(mat[0][j], mat[1][j])] = wholex[i,j,:]
    g1.add_nodes_from(node)
    g1.add_edges_from(edges)
    nx.set_node_attributes(g1, nodes, 'nodes')
    nx.set_edge_attributes(g1, edge_attributes, 'edges')
    G.append(g1)
    #print (edges, edge_attributes)

#%%
from grakel.kernels import SvmTheta
G_gr = graph_from_networkx(G, node_labels_tag='label', edge_labels_tag='labels2')
G_train, G_test, y_train, y_test = train_test_split(G_gr, y, test_size=0.1, random_state=24)

# Uses the graphhopper kernel to generate the kernel matrices
gk = SvmTheta()
K_train = gk.fit_transform(G_train)
K_test = gk.transform(G_test)

# Uses the SVM classifier to perform classification
clf = SVC(kernel="precomputed")
clf.fit(K_train, y_train)
y_pred = clf.predict(K_test)

# Computes and prints the classification accuracy
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", str(round(acc*100, 2)) + "%")