import numpy as np
import pandas as pd
from metrics import fscore
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from funcgraph import BrainGraph
from metrics import diag_flattened_indices, find_indices
from paramopt import graph_options
from sklearn.svm import SVC

# %%
def make_hist(X_train, y_train, ax):
    """
     Histogram to test the difference the scaling makes to the fscores.
    @param X_train:
    @param y_train:
    @param ax:
    @return:
    """

    scalar = StandardScaler()
    X_train_combined = pd.concat([X_train, y_train], axis=1)
    ax.hist(fscore(X_train_combined, class_col=len(X_train_combined.columns) - 1), label='before scaling',
            alpha=0.5)
    X_train_combined.iloc[:, :-1] = pd.DataFrame(scalar.fit_transform(X_train_combined.iloc[:, :-1]))
    ax.hist(fscore(X_train_combined, class_col=len(X_train_combined.columns) - 1), label='after scaling',
            alpha=0.5)
    ax.legend()
    return ax


# %%
def load_files(groups):
    combined_edges = []
    combined_nodes = []
    # first we will try classifying hc vs ad to check the appropriateness of the approach
    for group in groups:
        edges = np.load(f'{dir}/FC_{group}.npy', allow_pickle=True)
        print(f'Total number of NaN values found for group {group} {np.isnan(edges).sum().sum()}')
        edges = edges.reshape(edges.shape[0], edges.shape[1] * edges.shape[2])
        edges = pd.DataFrame(edges)
        edges['label'] = group
        edges_l = np.array(edges)
        combined_edges.extend(edges_l)
        nodes = np.load(f'{dir}/tau_{group}.npy', allow_pickle=True)
        # print(nodes.shape)
        nodes = pd.DataFrame(nodes)
        # print(nodes.head())
        nodes['label'] = group
        nodes_l = np.array(nodes)
        # print(nodes_l.shape)
        combined_nodes.extend(nodes_l)
        # print(len(combined_nodes), len(combined_nodes[0]))
    mapping = {'ad': 2, 'hc': 0, 'mci': 1}
    # print(len(combined_nodes), len(combined_nodes[0]))

    combined_edges, combined_nodes = pd.DataFrame(combined_edges), pd.DataFrame(combined_nodes,
                                                                                columns=range(len(combined_nodes[0])))
    combined_edges.iloc[:, -1] = combined_edges.iloc[:, -1].map(mapping)
    combined_nodes = combined_nodes.replace('m00', np.nan)
    #print(combined_nodes.isna().any(axis=1))
    combined_nodes.dropna(axis=0, inplace=True)
    #combined_nodes.dropna(inplace=True)  # drop the subject that contains this error
    combined_edges = combined_edges.loc[combined_nodes.index, :]
    #print(combined_edges.isna().any(axis=1))
    combined_edges.dropna(axis=0, inplace=True)
    combined_nodes = combined_nodes.loc[combined_edges.index]
    print('Nodes:', combined_nodes.shape)
    print('Edges:', combined_edges.shape)
    return combined_edges, combined_nodes


def graph_processing(fscores, num_nodes, node_weights, preserved_nodes):

    input_graph = BrainGraph(edge='fscores', feature_type='FC', node_wts='given_value',
                             target='AD', max_num_nodes=preserved_nodes)

    input_graph.make_graph(fscores, num_nodes=num_nodes)
    input_graph.delete_per_edges(20) # delete bottom 10% of the edges
    input_graph.set_node_labels(node_wts='given_value',
                                wts=node_weights)  # set the node weights as the mean for the training subjects

    input_graph.savefiles('/home/shreya/git/Thesis/gmwcs-solver')
    input_graph.visualize_graph('/home/shreya/git/Thesis/gmwcs-solver', True,
                                graph_options(color='red', node_size=5, linewidhts=0.1, width=1),
                                figs=(20,20))
    input_graph.run_solver('/home/shreya/git/Thesis/gmwcs-solver', max_num_nodes=preserved_nodes)
    print('Nodes in input graph', input_graph.nodes)



def fscore_hist(combined_edges):

    scalar = StandardScaler()
    fscores = fscore(combined_edges, class_col=len(combined_edges.columns) - 1)
    trans_edges = pd.DataFrame(scalar.fit_transform(combined_edges.iloc[:, :-1]))
    trans_fscores = fscore(pd.concat([trans_edges, combined_edges.iloc[:, -1]], axis=1),
                           class_col=len(combined_edges.columns) - 1)
    # has the data already been standardized, if so in which way?
    plt.hist(fscores, label='before scaling', alpha=0.5)
    plt.hist(trans_fscores, label='after scaling to unit variance', alpha=0.5)
    plt.ylabel('Number of features')
    plt.xlabel('Fscores')
    plt.legend()
    plt.show()


# %%
def run_baseline(X_train, X_test, y_train, y_test, fscores, per):
    # filter according to the fscores
    val = np.percentile(fscores, per)
    index = []
    #print('Indices for the fscores in baseline')
    #assert fscores.index == X_train.index
    for i in fscores.index:
        if fscores[i] >= val:
            #include all such values
            #print(i, ':',fscores[i])
            index.append(i)
    clf = SVC()
    #print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    clf.fit(X_train.loc[:, index], y_train)
    test_score = clf.score(X_test.loc[:, index], y_test)
    params = clf.get_params()
    print('The score on the fold', round(test_score, 3))
    return test_score, params


# %%
dir = '/home/shreya/git/Thesis/data'
for groups in [['hc', 'ad'], ['hc', 'mci']]:
    print('Performance for groups', groups)
    combined_edges, combined_nodes = load_files(groups=groups)
    #fscore_hist(combined_edges)  # fscores of the edges
    #fscore_hist(combined_nodes)  # fscores of the node features
    num_nodes = len(combined_nodes.columns) - 1  # because there is one column for the label here
    assert not sum(combined_edges.isna().any())
    fig, ax = plt.subplots(5, 1, figsize=(15, 10))
    skf = StratifiedKFold(n_splits=5)
    i = 0
    ax = ax.ravel()
    base_scores = []
    base_params = []
    indices = list(find_indices(shape=num_nodes).keys())
    X = combined_edges.iloc[:, indices]
    y = combined_edges.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    for train_idx, cv_idx in skf.split(X_train,y_train):
        if i < 5:
            scalar = StandardScaler()
            X_train_cv, y_train_cv = X_train.iloc[train_idx, :],  y_train.iloc[train_idx]
            X_train_cv = pd.DataFrame(scalar.fit_transform(X_train_cv), columns=X_train_cv.columns)
            X_test_cv, y_test_cv = X_train.iloc[cv_idx, :], y_train.iloc[cv_idx]
            X_test_cv = pd.DataFrame(scalar.transform(X_test_cv), columns = X_test_cv.columns)
            node_weights = combined_nodes.iloc[train_idx, :-1].mean(axis=0)
            node_weights = round(node_weights, 3)

            #print('Shapes:', X_train.shape, y_train.shape, X_test.shape, y_test.shape)
             # select only the uppertriangular features due to symmetric matrix
            X_y_uppert = pd.concat([X_train_cv, y_train_cv], axis=1)
            fscores = fscore(X_y_uppert, class_col=y_train_cv.name)
            #print('Fscores shape', fscores.shape)
            score, params = run_baseline(X_train_cv, X_test_cv, y_train_cv, y_test_cv, fscores[:-1], 20)
            base_scores.append(score)
            base_params.append(params)

            # only the diagonal indices since we are considering the matrix to be symmetric
            #graph_processing(fscores[:-1], num_nodes, node_weights, preserved_nodes=20)
            i += 1
    avg_score = np.mean(np.array(base_scores))
    idx = np.argmax(np.array(base_scores))
    params = base_params[idx]
    print('The average score on k-fold', round(avg_score, 3))
    # now fit the best estimator from the above
    clf = SVC(**params)
    scalar = StandardScaler()
    X_train = scalar.fit_transform(X_train)
    X_test = scalar.transform(X_test)
    clf.fit(X_train, y_train)
    print('Final scorea on independent test set' , round(clf.score(X_test, y_test), 3))