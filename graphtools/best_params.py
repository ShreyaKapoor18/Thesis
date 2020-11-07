#%%
import copy
from itertools import product
from classification_refined import classify
from processing import *
from readfiles import *
from decision import filter_summary
from subgraphclass import make_solver_summary
from sklearn.model_selection import train_test_split
from classification_refined import *
#%%
''' Data computed for all 5 personality traits at once'''
# labels for the computed subjects, data.index is the subject id
num = 84  # number of nodes in the graph
tri = int(num * (num + 1) * 0.5)  # we want only the upper diagonal parts since everything below diagonal is 0
big5 = ['NEOFAC_A', 'NEOFAC_O', 'NEOFAC_C', 'NEOFAC_N', 'NEOFAC_E']
edge_names = ['mean_FA', 'mean_strl', 'num_streamlines']
labels= ['Agreeableness', 'Openness', 'Conscientiousness', 'Neuroticism',
        'Extraversion']
mapping = {k: v for k, v in zip(labels, big5)}
mat = np.triu_indices(84)
mews = '/home/skapoor/Thesis/gmwcs-solver'
metrics = ['balanced_accuracy', 'accuracy', 'f1_weighted', 'roc_auc_ovr_weighted']
edges = [ 'fscores', 't_test']
# note: right now the matrix whole is not scaled, for computing the fscores and correlation coeff it has to be so.
y_train = computed_subjects()
X_train = generate_combined_matrix(tri, list(y_train.index))  # need to check indices till here then convert to numpy array
num_strls = X_train.iloc[:, 2 * tri:]
labels = ['Gender']
mapping = {'Gender': 'Gender'}
y_test = test_subjects()
X_test = generate_test_data(tri, y_test.index)
#X_train, X_test, y_train, y_test = train_test_split(
#    X_train, y_train, test_size=0.33, random_state=42)

target = 'Gender'
feature = 'num_streamlines'
edge = 'fscores'
val = -0.01
thresh = 0
solver_node_wts = 'const'
max_num_nodes = 20
choice = 'randome'
print('-' * 100)
feature_selection = 'baseline'
classifier = 'SVC'
refit_metric = 'roc_auc_ovr_weighted'
output_edges = 138
avg_thresh= False
strls_num = num_strls
# print('CSV parameter setting number')
# why is it always stopping at the first row
X_train_l, X_test_l = edge_filtering(feature, X_train, X_test)
assert len(X_train) == len(y_train)
if choice in ['test throw median', 'keep median']:
    y_train_l = y_train[mapping[target]]
    y_test_l = y_test[mapping[target]]
    med = int(y_train_l.median())  # the median is tried based on the training set
    y_train_l = pd.qcut(y_train_l, 5, labels=False, retbins=True)[0]
    # we need to pass the non-binned values for effective pearson correlation calc.
    # print('The number of training subjects which are to be removed:', sum(y_train_l == 2))
    y_train_l = y_train_l[y_train_l != 2]
    y_train_l = y_train_l // 3  # binarizing the values by removing the middle quartile
    X_train_l = X_train_l.loc[y_train_l.index]
    assert len(X_train_l) == len(y_train_l)
    # print('The choice that we are using', choice)
    if choice == 'test throw median':

        y_test_l = y_test_l[abs(y_test_l - med) > 1]  # maybe most of the values are close to the median
        y_test_l = y_test_l >= med  # binarizing the label check for duplicates
        X_test_l = X_test_l.loc[list(set(y_test_l.index))]
        # making sure that the training data is also for the same subjects
        assert len(X_test_l) == len(y_test_l)
    elif choice == 'keep median':
        y_test_l = y_test_l >= med  # we just binarize it and don't do anything else
    # now we do the cross validation search
else:
    y_train_l = y_train[target]
    y_test_l = y_test[target]
    y_train_l = y_train_l.map({'M': 0, 'F': 1})
    y_test_l = y_test_l.map({'M': 0, 'F': 1})
if feature_selection == 'solver':
    print(classifier, feature_selection, choice, refit_metric, target, feature, edge,
          solver_node_wts)
    strls_num_l = strls_num.loc[X_train_l.index, :]
    if avg_thresh == True:
        strls_num_l = strls_num.mean(axis=0, skipna=True)
    else:
        strls_num_l = strls_num_l.all()
    X_train_l, X_test_l, edge_wts, arr = solver(X_train_l, X_test_l, y_train_l, strls_num_l, feature, thresh,
                                                val,
                                                max_num_nodes, avg_thresh,
                                                node_wts=solver_node_wts, target=target, edge=edge)

    if len(edge_wts) != 0:
        train_res, test_res = cross_validation(classifier, X_train_l, y_train_l, X_test_l, y_test_l,
                                               metrics, refit_metric)
        # to make the program faster only do this when the solver is actually producing some results

elif feature_selection == 'baseline':
    per = (100 * output_edges) / 3486
    self_loops = False
    case = (classifier, target, choice, edge, feature_selection, feature, per, refit_metric, self_loops)

    if not self_loops:
        X_train_inl = X_train_l.drop(X_train_l.columns[diag_flattened_indices(84)], axis=1)
        X_test_inl = X_test_l.drop(X_test_l.columns[diag_flattened_indices(84)], axis=1)
    else:
        X_train_inl = X_train_l
        X_test_inl = X_test_l
    X_train_inl, X_test_inl, arr, index = transform_features(X_train_inl, X_test_inl, y_train_l, per,
                                                             edge)
    output_gr = BrainGraph(edge, f'{feature}_baseline', 'baseline', target, per, val, thresh)
    edges = []
    nodes = set()
    for ind in index:
        edges.append((np.triu_indices(84)[0][ind] + 1, np.triu_indices(84)[1][ind] + 1, 1))
        nodes.add(np.triu_indices(84)[0][ind] + 1)
        nodes.add(np.triu_indices(84)[1][ind] + 1)
    output_gr.add_nodes_from(nodes)
    for node in nodes:
        output_gr.nodes[node]['label'] = 1
    output_gr.add_weighted_edges_from(edges)
    output_gr.savefiles(mews)
    train_res, test_res = cross_validation(classifier, X_train_inl, y_train_l, X_test_inl,
                                           y_test_l, metrics, refit_metric)
