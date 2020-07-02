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
import os
import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from classification import data_splitting
import networkx as nx
from sklearn.preprocessing import scale


# %%
def train_with_best_params(classifier, params, X, y):
    """
    Train the specified classifier with the best parameters obtained from
    Cross Validation
    """
    if classifier == 'RF':
        clf = RandomForestClassifier(**params)  # try if this method works so that don't have to use explicit arguments
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        clf.fit(X_train, y_train)
        clf.predict(X_test)
        return clf.feature_importances_  # put this as the edge weight in the graph
    return None


def nested_outputdirs(mews):
    if not os.path.exists(f'{mews}/outputs'):
        os.mkdir(f'{mews}/outputs')
    if not os.path.exists(f'{mews}/outputs/nodes'):
        os.mkdir(f'{mews}/outputs/nodes')
    if not os.path.exists(f'{mews}/outputs/edges'):
        os.mkdir(f'{mews}/outputs/edges')
    if not os.path.exists(f'{mews}/outputs/solver'):
        os.mkdir(f'{mews}/outputs/solver')


def different_graphs(fscores, mat, big5, data, whole, labels, corr, mews):
    nested_outputdirs(mews='/home/skapoor/Thesis/gmwcs-solver')
    with open('outputs/combined_params.json', 'r') as f:
        best_params = json.load(f)

        for i in range(5):  # different labels
            # print(labels[i], ':', big5[i])
            # for per in [5, 10]: # we actually see that keeping 5-10% of the features is the best option
            val = np.percentile(fscores[i].flatten(), 0)
            index = np.where(fscores[i].flatten() >= val)
            # for choice in ['qcut', 'median', 'throw median'
            # Let's say we only choose the throw median choice, because it is the one that makes more sense
            choice = 'throw median'  # out of all these we will use these particular choices only!
            X, y = data_splitting(choice, i, index, data, whole, labels)  # this X is for random forests training
            params = best_params['RF'][big5[i]]["100"][choice]
            feature_imp = train_with_best_params('RF', params, X, y)
            feature_imp = np.reshape(feature_imp, (feature_imp.shape[0] // 3, 3))
            g1 = nx.Graph()
            edges = {(0, 1)}
            # edges = {}
            edge_attributes = []
            # node_attributes = {x: random.randint(-10, 10) for x in range(84)}
            for edge in ['fscores', 'pearson', 'feature_importance']:
                # for each edge type we have a different feature
                for j in range(len(mat[0])):
                    # edges.add((mat[0][j], mat[1][j]))
                    if edge == 'fscores':
                        # edge_attributes.append((mat[0][j], mat[1][j], np.mean(fscores[i, :, j])))
                        edge_attributes.append(
                            (mat[0][j], mat[1][j], fscores[i, 0, j]))  # only the first type of information, MEan FA
                    if edge == 'pearson':
                        # print(corr.shape, 'pearson')
                        # edge_attributes.append((mat[0][j], mat[1][j], np.mean(corr[i, :, j])))
                        edge_attributes.append((mat[0][j], mat[1][j], corr[i, 0, j]))
                    if edge == 'feature_importance':
                        # edge_attributes.append((mat[0][j], mat[1][j], np.mean(feature_imp[j, :])))
                        edge_attributes.append((mat[0][j], mat[1][j], feature_imp[j, 0]))
                        # then we should have just one graph for all subjects
                # this graph is then needed to be put into the solver in order to get the maximum edge weighted subgraph
                g1.add_nodes_from(range(84))
                # g1.add_edges_from(edges)
                g1.add_weighted_edges_from(edge_attributes)  # shall be a list of tuples
                node_labels = []
                for l in range(len(g1.nodes)):
                    g1.nodes[l]['label'] = max([g1[l][k]['weight'] for k in range(len(g1[l]))])
                    node_labels.append(g1.nodes[l]['label'])
                node_labels = scale(node_labels)  # standardizing the node labels
                for l in range(len(g1.nodes)):
                    g1.nodes[l]['label'] = node_labels[l]

                # putting this into the different text files

                filename = f'{big5[i]}_{edge}'  # make more nested directories
                nodes_file = open(f'{mews}/outputs/nodes/{filename}', 'w')
                edges_file = open(f'{mews}/outputs/edges/{filename}', 'w')
                print(filename)
                count = 0
                for x in g1.nodes:
                    # print(node)
                    if g1.degree(x) >= 2:
                        print(str(x) + ' ' * 3 + str(g1.nodes[x]['label']), file=nodes_file)
                        # print(str(node) + ' ' + str(0), file=nodes_file)
                        count += 1

                        # print(node, 'has degree >=2')
                print('Number of nodes having degree>=2', count)
                # print(len(nodes))

                for x in g1.nodes:
                    # if edge[0] in g1.nodes and edge[1] in g1.nodes:
                    for conn in g1[x]:
                        print(str(x) + ' ' * 3 + str(conn) + ' ' * 3 + str(g1[x][conn]['weight']),
                              file=edges_file)  # original file format was supposed to have 3 spaces
                        # print(str(edge[0]) + ' ' + str(edge[1]) + ' ' + str(randint(-5,5)), file = edges_file)
                nodes_file.close()
                edges_file.close()

                print('*' * 100, '\n', filename)

                os.chdir(mews)
                print('Current directory', os.getcwd())
                cmd = (f' java -Xss4M -Djava.library.path=/opt/ibm/ILOG/CPLEX_Studio1210/cplex/bin/x86-64_linux/ '
                       f'-cp /opt/ibm/ILOG/CPLEX_Studio1210/cplex/lib/cplex.jar:target/gmwcs-solver.jar '
                       f'ru.ifmo.ctddev.gmwcs.Main -e outputs/edges/{filename} '
                       f'-n outputs/nodes/{filename} > outputs/solver/{filename}')
                print(cmd)
                os.system(cmd)
                os.chdir("/home/skapoor/Thesis/graphtools")
                print('Current directory', os.getcwd())
