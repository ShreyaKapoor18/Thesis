import os
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from classification import data_splitting
from paramopt import get_distributions
import networkx as nx
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
# %%
# just for the time being
import warnings

warnings.filterwarnings("ignore")


# %%
# %%
def train_from_combined_graph(metrics, personality_trait, edge, node_wts, mat, mews,
                              big5, labels, data, whole, tri, **kwargs):
    """
    For all the features mean FA, mean streamline length and number of streamlines that have
    been reduced by the solver.  We combine them and then train a classifier on the combined graph.
    @param metrics: for evaluating the classifier
    @param personality_trait: the personality trait we want to visualize
    @param edge:the type of edge we have given input to the solver i.e. fscore, mean FA, mean strl
    @param node_wts: how we assign the weighting to the nodes
    @param mat: the upper triangular indices for a nxn matrix
    @param mews: the directory path
    @param big5: list personality traits
    @param labels:the labels in the original dataframe
    @param data: the original dataframe containing all the labels
    @param whole: the original matrix containing all feature data for all subjects
    @param tri: the number of features in the upper triangular form
    @param kwargs: if keyword arguments needed in future
    """
    i = big5.index(personality_trait)
    all_feature_indices = []
    for feature in ['mean_FA', 'mean_strl', 'num_strl']:
        filename = f'{personality_trait}_{edge}_{node_wts}_{feature}'  # make more nested directories
        print(filename + '.out')
        if os.path.exists(f'{mews}/outputs/nodes/{filename}.out') \
                and os.path.exists(f'{mews}/outputs/edges/{filename}.out'):
            with open(f'{mews}/outputs/nodes/{filename}.out', 'r') as nodes_file, \
                    open(f'{mews}/outputs/edges/{filename}.out', 'r') as edges_file:
                nodes = [x.split('\t') for x in nodes_file.read().split('\n')]
                edges = [x.split('\t') for x in edges_file.read().split('\n')]

                nodes_e = set()
                edges_e = set()
                for a in nodes[:-1]:
                    if a[1] != 'n/a':
                        nodes_e.add(int(a[0]))
                feature_indices = set()

                # 0 to range(len(mat)), everything in matrix whole corresponding to this edge is feature
                for existing_edge in edges[:-1]:
                    if existing_edge[-1] != 'n/a':
                        edges_e.add((int(existing_edge[0]), int(existing_edge[1]), float(existing_edge[2])))
                        for k in range(len(mat[0])):
                            if (int(existing_edge[0]), int(existing_edge[1])) == (mat[0][k], mat[1][k]):
                                feature_indices.add(k)
                                # all_feature_indices.extend([k, k+tri, k+2*tri]) # for the three types FA, n strl, strlen
                                # all_feature_indices.extend([k, k + tri, k + 2 * tri])
                                if feature == 'mean_FA':
                                    all_feature_indices.append(k)
                                    # feature_mat = whole.iloc[:, :tri]
                                if feature == 'mean_strl':
                                    all_feature_indices.append(k + tri)
                                    # feature_mat = whole.iloc[:, tri:2*tri]
                                if feature == 'num_str':
                                    all_feature_indices.append(k + 2 * tri)
                                    # feature_mat = whole.iloc[:, 2*tri:]

                g2 = nx.Graph()
                g2.add_nodes_from(nodes_e)
                g2.add_weighted_edges_from(edges_e)
                options = {
                    'node_color': 'black',
                    'node_size': 1,
                    'line_color': 'grey',
                    'linewidths': 0,
                    'width': 0.1,
                }
                edge_wts = []
                for m in g2.edges.data():
                    # print(m)
                    edge_wts.append(m[2]['weight'])
                if edge_wts != []:
                    minima = min(edge_wts)
                    maxima = max(edge_wts)

                    norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima)
                    mapper = cm.ScalarMappable(norm=norm, cmap=plt.get_cmap('Spectral'))
                    color = []
                    for v in edge_wts:
                        color.append(mapper.to_rgba(v))
                    plt.figure()
                    plt.title(
                        f'Nodes with degree >2, output from the solver: {filename}.out\n Number of edges {len(g2.edges)}\n'
                        f'Target: {personality_trait}, Feature:{feature}\n'
                        f'Edge type:{edge}, Node weighting:{node_wts}')
                    nx.draw(g2, **options, edge_color=color)

                    plt.show()

                    print('Number of features selected by the solver', len(all_feature_indices))

                else:
                    print("The file is empty, the solver didn\'t reduce anything in the network")

    feature_mat = whole.iloc[:, all_feature_indices]
    with open(f'{mews}/outputs/classification_results/{personality_trait}_{edge}_{node_wts}', 'w+') as results_file:
        for choice in ['throw median']:
            if all_feature_indices != []:
                X, y = data_splitting(choice, i, all_feature_indices, data, feature_mat, labels)
                for classifier in ['SVC', 'RF', 'MLP']:
                    print('Choice and classifier', choice, classifier)
                    clf, distributions = get_distributions(classifier)
                    rcv = RandomizedSearchCV(clf, distributions, random_state=55, scoring=metrics,
                                             refit='roc_auc_ovr_weighted',
                                             cv=5)  # maybe we can train with the best params here
                    # scores = cross_validate(clf, X, Y, cv=5, scoring=metrics)
                    search = rcv.fit(X, y)
                    scores = search.cv_results_
                    for metric in metrics:
                        print(f'mean_test_{metric}:', round(np.mean(scores[f'mean_test_{metric}']), 3),
                              file=results_file)
<<<<<<< HEAD
                        print(f'mean_test_{metric}:', round(np.mean(scores[f'mean_test_{metric}']), 3))
=======
                    print(f'mean_test_{metric}:', round(np.mean(scores[f'mean_test_{metric}']), 3))
>>>>>>> d318a20212b1db172478be9afc51f5ee6661fbe0
