from processing import generate_combined_matrix, hist_fscore
from readfiles import computed_subjects
import numpy as np
import networkx as nx
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import json
from classification import data_splitting
import random
import os
# %%
def train_with_best_params(classifier, params, X, y):
    '''
    classifier: name of the classifier
    params: the best parameters obtained from the previous cross validation
    X: data
    y: labels
    '''
    if classifier == 'RF':
        clf = RandomForestClassifier(**params)  # try if this method works so that don't have to use explicit arguments
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        clf.fit(X_train, y_train)
        clf.predict(X_test)
        return clf.feature_importances_  # put this as the edge weight in the graph
    return None


# %%
data = computed_subjects()  # labels for the computed subjects
y = np.array(data['NEOFAC_A'] >= data['NEOFAC_A'].median()).astype(int)
num = 84  # number of nodes in the graph
tri = int(num * (num + 1) * 0.5)  # we want only the upper diagonal parts since everything below diagonal is 0
whole = generate_combined_matrix(tri, data.index)
big5 = ['Agreeableness', 'Openness', 'Conscientiousness', 'Neuroticism',
        'Extraversion']
labels = ['NEOFAC_A', 'NEOFAC_O', 'NEOFAC_C', 'NEOFAC_N', 'NEOFAC_E']
edge_names = ['mean_FA', 'mean strl', 'num streamlines']
fscores = hist_fscore(data, whole, labels, big5, edge_names, tri)

# without taking the edge type into consideration
new_fscores = np.reshape(fscores, (fscores.shape[0], fscores.shape[1] * fscores.shape[2]))

# Values of C parameter of SVM

mat = np.triu_indices(84)

with open('outputs/combined_params.json', 'r') as f:
    best_params = json.load(f)
    for i in range(5):  # different labels
        for per in [5, 10, 50, 100]: # we actually see that keeping 5-10% of the features is the best option
            val = np.percentile(new_fscores[i], 100 - per)
            index = np.where(new_fscores[i] >= val)
            for choice in ['qcut', 'median', 'throw median']:
                X, y = data_splitting(choice, i, index, data, whole, labels)
                params = best_params['RF'][big5[i]][per][choice]
                feature_imp = train_with_best_params('RF', params, X, y)
                #make a graph with these feature importsances
                 # can we give this weight to each of the columns in the SVM matrix
                 #these graph properties are then needed to be given to the solver?

                feature_imp2 = np.reshape(feature_imp, (feature_imp.shape[0] // 3, 3)) #first dimension shall be equal to the number of edges
                mat = np.triu_indices(84)
                # nodes = {x: 1 for x in range(84)}
                node = range(84)
                g1 = nx.Graph()
                edges = {(0, 1)}
                # edges = {}
                edge_attributes = {}
                node_attributes = {x: random.randint(-10,10) for x in range(84)}
                for j in range(len(mat[0])):
                    edges.add((mat[0][j], mat[1][j]))  # first feature is mean fa between nodes
                    # edges[(mat[0][j], mat[1][j])] = 1
                    edge_attributes[(mat[0][j], mat[1][j])] = feature_imp2[j, :] #then we should have just one graph for all subjects
                    # this graph is then needed to be put into the solver in order to get the maximum edge weighted subgraph
                g1.add_nodes_from(node)
                g1.add_edges_from(edges)
                # nx.set_node_attributes(g1, nodes, 'nodes')
                nx.set_edge_attributes(g1, edge_attributes, 'edges')
                # print (edges, edge_attributes)
                #%%
                #we can also make the graph from the numpy array, the numpy array will be 84x84 with three edge
                new = np.zeros((84,84,3))
                k = 0
                for j in range(len(mat[0])):
                       if k <= 3570:
                           new[mat[0][i], mat[1][i], :] = feature_imp2[k,:]
                           k+=1
                #construct the graph from the array
                G2 = nx.from_numpy_array(new)
                G = G2
                #can start comparing the arrays g1 and g2, then we can see how to input this into the solve
                filename = f'combined_graph{choice}{per}{big5[i]}'
                #%%
                nodes_file = open(f'{filename}_nodes', 'w+')
                edges_file = open(f'{filename}_edges', 'w+')
                nodes = []
                count = 0
                for node in nx.nodes(G):
                    #print(node)
                    if G.degree(node) >=2:
                        print(str(node) + ' ' + str(1), file=nodes_file)
                        #print(str(node) + ' ' + str(0), file=nodes_file)
                        nodes.append(node)
                        count +=1
                        #print(node, 'has degree >=2')
                print('Number of nodes having degree>=2', count)
                #print(len(nodes))

                for edge in nx.edges(G):
                    if edge[0] in nodes and edge[1] in nodes:
                         print(str(edge[0]) + ' ' + str(edge[1])+ ' '+ str(G.get_edge_data(edge[0], edge[1])['weight']),
                              file=edges_file)
                         #print(str(edge[0]) + ' ' + str(edge[1]) + ' ' + str(randint(-5,5)), file = edges_file)
                nodes_file.close()
                edges_file.close()

                print(filename, '*' * 100)
                mews = '/home/shreya/Desktop/Thesis/gmwcs-solver'
                cmd = (f' java -Xss4M -Djava.library.path=/opt/ibm/ILOG/CPLEX_Studio_Community129/cplex/bin/x86-64_linux/ '
                       f'-cp /opt/ibm/ILOG/CPLEX_Studio_Community129/cplex/lib/cplex.jar:{mews}/target/gmwcs-solver.jar '
                       f'ru.ifmo.ctddev.gmwcs.Main -e {filename}_edges '
                       f'-n {filename}_nodes ')
                os.system(cmd)
#%%
G = []
# nodes = {x: 1 for x in range(84)}
node = range(84)
for i in range(len(wholex)):
    g1 = nx.Graph()
    edges = {(0, 1)}
    # edges = {}
    edge_attributes = {}
    for j in range(len(mat[0])):
        edges.add((mat[0][j], mat[1][j]))  # first feature is mean fa between nodes
        # edges[(mat[0][j], mat[1][j])] = 1
        edge_attributes[(mat[0][j], mat[1][j])] = wholex[i,j,:] # this is what we need to put as random forest features
        edge_attributes[(mat[0][j], mat[1][j])] = feature_imp[j, :] #then we should have just one graph for all subjects
        # this graph is then needed to be put into the solver in order to get the maximum edge weighted subgraph
    g1.add_nodes_from(node)
    g1.add_edges_from(edges)
    # nx.set_node_attributes(g1, nodes, 'nodes')
    nx.set_edge_attributes(g1, edge_attributes, 'edges')
    G.append(g1)
    # print (edges, edge_attributes)
