from processing import generate_combined_matrix, hist_fscore
from readfiles import computed_subjects
import numpy as np
import networkx as nx
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
#%%
data = computed_subjects()  # labels for the computed subjects
y = np.array(data['NEOFAC_A']>=data['NEOFAC_A'].median()).astype(int)
num = 84  # number of nodes in the graph
tri = int(num * (num + 1) * 0.5)  # we want only the upper diagonal parts since everything below diagonal is 0
whole = generate_combined_matrix(tri, data.index)
whole = np.array(whole)
# Values of C parameter of SVM
wholex = np.reshape(whole,(whole.shape[0], whole.shape[1]//3, 3))
mat = np.triu_indices(84)
#%%
G = []
#nodes = {x: 1 for x in range(84)}
node = range(84)
for i in range(len(wholex)):
    g1 = nx.Graph()
    edges = {(0,1)}
    #edges = {}
    edge_attributes = {}
    for j in range(len(mat[0])):
        edges.add((mat[0][j], mat[1][j])) # first feature is mean fa between nodes
        #edges[(mat[0][j], mat[1][j])] = 1
        edge_attributes[(mat[0][j], mat[1][j])] = wholex[i,j,:] # this is what we need to put as random forest features
    g1.add_nodes_from(node)
    g1.add_edges_from(edges)
    #nx.set_node_attributes(g1, nodes, 'nodes')
    nx.set_edge_attributes(g1, edge_attributes, 'edges')
    G.append(g1)
    #print (edges, edge_attributes)
#%%
def train_with_best_params(classifier, params, X, y):
    if classifier == 'RF':
        clf = RandomForestClassifier(
            min_samples_leaf=params['min_samples_leaf'],
            min_samples_split=params['min_samples_split'],
            bootstrap =params['bootstrap'],
            max_features=params['max_features'],
            n_estimators=params['n_estimators'], max_depth=params['max_depth'])
        X_train, X_test, y_train, y_test = train_test_split(X,y)
        clf.fit(X_train, y_train)
        clf.predict(X_test)
        return clf.feature_importances_
