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
import json
import os
import networkx as nx
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
from classification import data_splitting
import matplotlib
import matplotlib.cm as cm
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


def nested_outputdirs(mews): # make a separate directory for each label, easier to do comparisons

        if not os.path.exists(f'{mews}/outputs'):
            os.mkdir(f'{mews}/outputs')
        if not os.path.exists(f'{mews}/outputs/nodes'):
            os.mkdir(f'{mews}/outputs/nodes')
        if not os.path.exists(f'{mews}/outputs/edges'):
            os.mkdir(f'{mews}/outputs/edges')
        if not os.path.exists(f'{mews}/outputs/solver'):
            os.mkdir(f'{mews}/outputs/solver')
        if not os.path.exists(f'{mews}/outputs/classification_results'):
            os.mkdir(f'{mews}/outputs/classification_results')


def different_graphs(fscores, mat, big5,personality_trait, data, edge,
                     whole, labels, corr, mews, threshold, node_wts, tri):
    nested_outputdirs(mews='/home/skapoor/Thesis/gmwcs-solver')
    with open('outputs/combined_params.json', 'r') as f:

        best_params = json.load(f)
        i = big5.index(personality_trait)
        index = list(range(3*tri))
        #for choice in ['qcut', 'median', 'throw median'
        # Let's say we only choose the throw median choice, because it is the one that makes more sense
        choice = 'throw median'  # out of all these we will use these particular choices only!
        X, y = data_splitting(choice, i, index, data, whole, labels)  # this X is for random forests training
        params = best_params['RF'][personality_trait]["100"][choice]  # maybe use the parameters that work the best for top 5%
        feature_imp = train_with_best_params('RF', params, X, y)

        assert len(feature_imp) == len(X[0])
        feature_imp = np.reshape(feature_imp, (3, feature_imp.shape[0] // 3)) #since we are training different features
        print('X:', X.shape)
        print('feature importance:', feature_imp.shape, 'len mat', len(mat[0]))

        for feature in ['mean_FA', 'mean_strl', 'num_str']:
            if edge == 'fscores':
                arr = fscores[i, :, :]
            if edge == 'pearson':
                arr = corr[i, :, :]
            if edge == 'feature_importance':
                arr = feature_imp
            #assert type(arr) == np.ndarray
            print('type of array', type(arr))
            #print('The array shape is:', arr.shape)
            #assert len(feature_imp) == len(mat[0])
            # for each edge type we have a different feature

            thresh = np.percentile(np.absolute(arr), threshold)  # remove bottom ex percent in absolute terms
            #assert np.percentile(np.absolute(arr), 50) == np.median(np.absolute(arr))
            #assert np.percentile(np.absolute(arr),10) < np.percentile(np.absolute(arr), 20)
            # in order to confirm that it actually makes the percentile distribution!
            print(f'Threshold value according to {threshold} percentile: {thresh}')
            index2 = np.where(np.absolute(arr) <= thresh)
            print('indexes', index2)
            xs = index2[0]
            ys = index2[1]
            #removed wrong indexing
            for p in range(len(index2[0])):
                arr[xs[p], ys[p]] = 0

            # try for for different types, one feature at a time maybe and then construct graph?
            nodes = set()
            edge_attributes = []
            for j in range(len(mat[0])):
                if feature == 'mean_FA':
                    value = arr[0, j]
                if feature == 'mean_strl':
                    value = arr[1,j]
                if feature == 'num_str':
                    value = arr[2,j]
                if abs(value) > thresh:
                    edge_attributes.append((mat[0][j], mat[1][j], value))
                    nodes.add(mat[0][j]) #add only the nodes which have corresponding edges
                    nodes.add(mat[1][j])
            # mean for the scores of three different labels
            assert nodes!= None
            g1 = nx.Graph()
            g1.add_nodes_from(nodes)
            g1.add_weighted_edges_from(edge_attributes)  # shall be a list of tuples

            node_labels = []
            for l in g1.nodes.keys():
                if node_wts == 'max':
                    g1.nodes[l]['label'] = max([g1[l][k]['weight'] for k in g1[l].keys()]) #max or max abs?
                elif node_wts == 'const':
                    g1.nodes[l]['label'] = 1
                node_labels.append(g1.nodes[l]['label'])

            node_labels = scale(node_labels)  # standardizing the node labels
            for n in g1.nodes.keys():
                g1.nodes[n]['label'] = node_labels[n]

            # now we want to have only the nodes which have degree >1

            # putting this into the different text files

            filename = f'{personality_trait}_{edge}_{node_wts}_{feature}'  # make more nested directories
            nodes_file = open(f'{mews}/outputs/nodes/{filename}', 'w')
            edges_file = open(f'{mews}/outputs/edges/{filename}', 'w')
            print(filename)
            count = 0
            connected_nodes = []
            for x in g1.nodes:
                # print(node)
                if g1.degree(x) >= 1:#solver documentation, 1 or 2
                    print(str(x) + ' ' * 3 + str(g1.nodes[x]['label']), file=nodes_file)
                    connected_nodes.append(x)
                    # print(str(node) + ' ' + str(0), file=nodes_file)
                    count += 1
            g2 = g1.subgraph(connected_nodes) #make a subgraph from the original one that only contains selected nodes
            options = {
                'node_color': 'black',
                'node_size': 1,
                'line_color': 'grey',
                'linewidths': 0,
                'width': 0.1,
            }
            edge_wts = []
            for m in g2.edges.data():
                #print(m)
                edge_wts.append(m[2]['weight'])
            minima = min(edge_wts)
            maxima = max(edge_wts)

            norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima)
            mapper = cm.ScalarMappable(norm=norm, cmap=plt.get_cmap('Spectral'))
            color = []
            for v in edge_wts:
                color.append(mapper.to_rgba(v))
            plt.figure()
            plt.title(f'Nodes with degree >2, input to the solver: {filename}\n Number of edges {len(g2.edges)}\n'
                      f'Percentage of features:{100-threshold}, Target: {personality_trait}, Feature:{feature}\n'
                      f'Edge type:{edge}, Node weighting:{node_wts}')
            nx.draw(g2, **options, edge_color=color)
            plt.show()
            # print(node, 'has degree >=2')
            print('Number of nodes having a degree>=2', count)
            # print(len(nodes))

            for x in g1.nodes:
                # if edge[0] in g1.nodes and edge[1] in g1.nodes:
                for conn in g1[x]:
                    print(str(x) + ' ' * 3 + str(conn) + ' ' * 3 + str(g1[x][conn]['weight']),
                          file=edges_file)  # original file format was supposed to have 3 spaces

            nodes_file.close()
            edges_file.close()

            print('*' * 100, '\n', filename)

            os.chdir(mews)
            print('Current directory', os.getcwd())
            cmd = (
                f' java -Xss4M -Djava.library.path=/opt/ibm/ILOG/CPLEX_Studio1210/cplex/bin/x86-64_linux/ '
                f'-cp /opt/ibm/ILOG/CPLEX_Studio1210/cplex/lib/cplex.jar:target/gmwcs-solver.jar '
                f'ru.ifmo.ctddev.gmwcs.Main -e outputs/edges/{filename} '
                f'-n outputs/nodes/{filename} > outputs/solver/{filename}')
            print(cmd)
            os.system(cmd)
            os.chdir("/home/skapoor/Thesis/graphtools")
            print('Current directory', os.getcwd())
