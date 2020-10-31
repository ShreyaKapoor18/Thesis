import copy
from itertools import product
from classification_refined import classify
from processing import *
from readfiles import *
from decision import filter_summary
from subgraphclass import make_solver_summary
from sklearn.model_selection import train_test_split
from classification_refined import *
import networkx as nx
#%%
def find_indices(lin):
    d1 = {}
    mat = np.triu_indices(84)
    for idx in lin:
        for i in range(len(mat[0])):
            if i == idx:
                d1[idx] = (mat[0][i], mat[1][i])
    return d1
#%%
num = 84  # number of nodes in the graph
tri = int(num * (num + 1) * 0.5)  # we want only the upper diagonal parts since everything below diagonal is 0
big5 = ['NEOFAC_A', 'NEOFAC_O', 'NEOFAC_C', 'NEOFAC_N', 'NEOFAC_E']
edge_names = ['mean_FA', 'mean_strl', 'num_streamlines']
labels = ['Agreeableness', 'Openness', 'Conscientiousness', 'Neuroticism',
         'Extraversion']
mapping = {k: v for k, v in zip(labels, big5)}
mat = np.triu_indices(84)
mews = '/home/skapoor/Thesis/gmwcs-solver'
metrics = ['balanced_accuracy', 'accuracy', 'f1_weighted', 'roc_auc_ovr_weighted']
edges = ['fscores', 't_test']
# note: right now the matrix whole is not scaled, for computing the fscores and correlation coeff it has to be so.
y_train = computed_subjects()
X_train = generate_combined_matrix(tri,
                                  list(y_train.index))  # need to check indices till here then convert to numpy array
num_strls = X_train.iloc[:, 2 * tri:]
labels = ['Gender']
mapping = {'Gender': 'Gender'}
y_test = test_subjects()
X_test = generate_test_data(tri, y_test.index)
#%%
target = 'Gender'
feature = 'num_streamlines'
edge = 'fscores'
val = -0.01
thresh = 0
solver_node_wts = 'const'
max_num_nodes = 10
per = 1.18
choice = 'random'
classifier = 'SVC'
baseline_cases = set()
refit_metric = 'balanced_accuracy'
self_loops = False
feature_selection = 'baseline'
y_train_l = y_train[target]
y_test_l = y_test[target]
y_train_l = y_train_l.map({'M': 0, 'F': 1})
y_test_l = y_test_l.map({'M': 0, 'F':1})
case = (classifier, target, choice, edge, feature_selection, feature, per, refit_metric, self_loops)
X_train_l, X_test_l = edge_filtering(feature, X_train, X_test)
if not self_loops:
        X_train_inl = X_train_l.drop(X_train_l.columns[diag_flattened_indices(84)], axis=1)
        X_test_inl = X_test_l.drop(X_test_l.columns[diag_flattened_indices(84)], axis=1)
        
X_train_inl, X_test_inl, arr, indices = transform_features(X_train_inl, X_test_inl, y_train_l, per,
                                                             edge)
#%%
d1= find_indices(indices)
#%%
#baseline with 42 features equivalent to 10 nodes on solver based method
nodes = set()
edges = list(d1.values())
links = np.zeros((len(edges),3))
#%%

links = np.zeros((len(edges), 3))

mapping = {}
g = nx.Graph()
g.add_edges_from(edges)
#%%
for i in range(len(g.nodes)):
    mapping[list(g.nodes)[i]] = i
#%%
for i, edge, index in zip(range(len(links)), g.edges, indices):
    links[i][0] = mapping[edge[0]]
    links[i][1] = mapping[edge[1]]
    vals = X_train_inl.loc[:, index]
    links[i][2] = vals.mean()
#