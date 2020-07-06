import os
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from classification import data_splitting
from paramopt import get_distributions
from processing import generate_combined_matrix
from readfiles import computed_subjects
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
for i in range(5):  # different labels
    for edge in ['fscores', 'pearson', 'feature_importance']:
        for node_wts in ['const', 'max']:
            for feature in ['mean_FA', 'mean_strl', 'num_str']:
                filename = f'{big5[i]}_{edge}_{node_wts}_{feature}'  # make more nested directories
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
                        all_feature_indices = []
                        # 0 to range(len(mat)), everything in matrix whole corresponding to this edge is feature
                        for existing_edge in edges[:-1]:
                            if existing_edge[-1] != 'n/a':
                                edges_e.add((int(existing_edge[0]), int(existing_edge[1])))
                                for k in range(len(mat[0])):
                                    if (int(existing_edge[0]), int(existing_edge[1])) == (mat[0][k], mat[1][k]):
                                        feature_indices.add(k)
                                        # all_feature_indices.extend([k, k+tri, k+2*tri]) # for the three types FA, n strl, strlen
                                        #all_feature_indices.extend([k, k + tri, k + 2 * tri])
                                        if feature == 'mean_FA':
                                            all_feature_indices.append(k)
                                            feature_mat = whole.iloc[:, :tri]
                                        if feature == 'mean_strl':
                                            all_feature_indices.append(k+tri)
                                            feature_mat = whole.iloc[:, tri:2*tri]
                                        if feature == 'num_str':
                                            all_feature_indices.append(k+2*tri)
                                            feature_mat = whole.iloc[:, 2*tri:]
                        # print('Existing nodes', nodes_e)
                        # print('Existing edges', edges_e)
                        # print('Feature indexes, edge indexes', feature_indices)
                        print('*' * 100)
                        # train classifier based on these edge indices
                        print('Number of features selected by the solver', len(all_feature_indices))
                        for choice in ['throw median']:
                            if all_feature_indices != []:
                                X, y = data_splitting(choice, i, all_feature_indices, data, feature_mat, labels)
                                for classifier in ['SVC', 'RF', 'MLP']:
                                    print(choice, classifier)
                                    clf, distributions = get_distributions(classifier)
                                    rcv = RandomizedSearchCV(clf, distributions, random_state=55, scoring=metrics,
                                                             refit='roc_auc_ovr_weighted', cv=5)
                                    # scores = cross_validate(clf, X, Y, cv=5, scoring=metrics)
                                    search = rcv.fit(X, y)
                                    scores = search.cv_results_
                                    for metric in metrics:
                                        print(f'mean_test_{metric}:', round(np.mean(scores[f'mean_test_{metric}']), 3))
