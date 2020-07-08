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
#%%
#just for the time being
import warnings
warnings.filterwarnings("ignore")
# %%
mat = np.triu_indices(84)
mews = '/home/skapoor/Thesis/gmwcs-solver'
big5 = ['Agreeableness', 'Openness', 'Conscientiousness', 'Neuroticism',
        'Extraversion']
labels = ['NEOFAC_A', 'NEOFAC_O', 'NEOFAC_C', 'NEOFAC_N', 'NEOFAC_E']
metrics = ['balanced_accuracy', 'accuracy', 'f1_weighted', 'roc_auc_ovr_weighted']
# %%
data = computed_subjects()  # labels for the computed subjects, data.index is the subject id
num = 84  # number of nodes in the graph
tri = int(num * (num + 1) * 0.5)  # we want only the upper diagonal parts since everything below diagonal is 0
whole = generate_combined_matrix(tri, list(data.index))  # need to check indices till here then convert to numpy array
assert list(whole.index) == list(data.index)
# %%
personality_trait = 'Openness'
edge = 'feature_importance'
node_wts = 'max'
feature = 'mean_FA'
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
                            #all_feature_indices.extend([k, k + tri, k + 2 * tri])
                            if feature == 'mean_FA':
                                all_feature_indices.append(k)
                                #feature_mat = whole.iloc[:, :tri]
                            if feature == 'mean_strl':
                                all_feature_indices.append(k+tri)
                                #feature_mat = whole.iloc[:, tri:2*tri]
                            if feature == 'num_str':
                                all_feature_indices.append(k+2*tri)
                                #feature_mat = whole.iloc[:, 2*tri:]
    
    
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
                #print(m)
                edge_wts.append(m[2]['weight'])
            if edge_wts!= []:
                minima = min(edge_wts)
                maxima = max(edge_wts)
    
                norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima)
                mapper = cm.ScalarMappable(norm=norm, cmap=plt.get_cmap('Spectral'))
                color = []
                for v in edge_wts:
                    color.append(mapper.to_rgba(v))
                plt.figure()
                plt.title(f'Nodes with degree >2, output from the solver: {filename}.out\n Number of edges {len(g2.edges)}\n'
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
                                         refit='roc_auc_ovr_weighted', cv=5) # maybe we can train with the best params here
                # scores = cross_validate(clf, X, Y, cv=5, scoring=metrics)
                search = rcv.fit(X, y)
                scores = search.cv_results_
                for metric in metrics:
                    print(f'mean_test_{metric}:', round(np.mean(scores[f'mean_test_{metric}']), 3),
                          file=results_file)
                    print(f'mean_test_{metric}:', round(np.mean(scores[f'mean_test_{metric}']), 3))