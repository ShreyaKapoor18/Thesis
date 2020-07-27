import os
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from classification import data_splitting
from paramopt import get_distributions
from processing import generate_combined_matrix
from readfiles import computed_subjects
import networkx as nx
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from paramopt import graph_options
# %%
# just for the time being
import warnings
warnings.filterwarnings("ignore")


def make_and_visualize(nodes, edges, feature, target, edge, node_wts, mat, filename, degree,
                       plotting_options):
    nodes_e = set()
    edges_e = set()
    for a in nodes[:-1]:
        if a[1] != 'n/a':
            nodes_e.add(int(a[0]))
    feature_indices = []

    # 0 to range(len(mat)), everything in matrix whole corresponding to this edge is feature
    for existing_edge in edges[:-1]:
        if existing_edge[-1] != 'n/a':
            edges_e.add((int(existing_edge[0]), int(existing_edge[1]), float(existing_edge[2])))
            for k in range(len(mat[0])):
                if (int(existing_edge[0]), int(existing_edge[1])) == (mat[0][k], mat[1][k]):
                    # all_feature_indices.extend([k, k+tri, k+2*tri]) # for the three types FA, n strl, strlen
                    # all_feature_indices.extend([k, k + tri, k + 2 * tri])
                        feature_indices.append(k)
                        # feature_mat = whole.iloc[:, :tri]]

    g2 = nx.Graph()
    g2.add_nodes_from(nodes_e)
    g2.add_weighted_edges_from(edges_e)

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
            f'Nodes with degree >={degree}, output from the solver: {filename}.out\n Number of edges {len(g2.edges)}\n'
            f'Target: {target}, Feature:{feature}\n'
            f'Edge type:{edge}, Node weighting:{node_wts}')
        nx.draw(g2, **plotting_options, edge_color=color)
        print('reached output figure state')
        plt.show()

        print('Number of features selected by the solver', len(feature_indices))
        return feature_indices

    else:
        print("The file is empty, the solver didn\'t reduce anything in the network")
        return None


'''def summarize_ipop():
    Summarize the input and the output graphs for a particular choice that has been specified for the pipeline
    @return:
    
'''


# %%
def train_from_reduced_graph(metrics, target, edge, node_wts, mat, mews, feature_type,
                              big5, target_col, whole, degree, plotting_options):
    """

    @rtype: object
    """
    i = big5.index(target)
    all_feature_indices = []
    filename = f'{target}_{edge}_{node_wts}_{feature_type}'  # make more nested directories
    print(filename + '.out')
    if os.path.exists(f'{mews}/outputs/nodes/{filename}.out') \
            and os.path.exists(f'{mews}/outputs/edges/{filename}.out'):
        with open(f'{mews}/outputs/nodes/{filename}.out', 'r') as nodes_file, \
                open(f'{mews}/outputs/edges/{filename}.out', 'r') as edges_file:
            nodes = [x.split('\t') for x in nodes_file.read().split('\n')]
            edges = [x.split('\t') for x in edges_file.read().split('\n')]

            all_feature_indices.extend(make_and_visualize(nodes, edges, feature_type, target, edge,
                                                          node_wts, mat, filename, degree, plotting_options))
    #feature_mat = whole.iloc[:, all_feature_indices]
    with open(f'{mews}/outputs/classification_results/{target}_{edge}_{node_wts}', 'w+') as results_file:
        choice = 'throw median'
        if all_feature_indices:
            X, y = data_splitting(choice, all_feature_indices, whole, target_col)
            for classifier in ['SVC', 'RF', 'MLP']:
                print('Choice and classifier', choice, classifier)
                clf, distributions = get_distributions(classifier)
                rcv = RandomizedSearchCV(clf, distributions, random_state=55, scoring=metrics,
                                         refit='balanced_accuracy',
                                         cv=5)  # maybe we can train with the best params here
                # scores = cross_validate(clf, X, Y, cv=5, scoring=metrics)
                search = rcv.fit(X, y)
                scores = search.cv_results_
                for metric in metrics:
                    print(f'mean_test_{metric}:', round(np.mean(scores[f'mean_test_{metric}']), 3),
                          file=results_file)
                    print(f'mean_test_{metric}:', round(np.mean(scores[f'mean_test_{metric}']), 3))
