import sys
sys.path.append('..')
import numpy as np
import pandas as pd
from metrics import fscore
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, GridSearchCV
from funcgraph import BrainGraph
from paramopt import graph_options
from sklearn.metrics import balanced_accuracy_score
import logging
import os
import itertools as it
from metrics import diag_flattened_indices, find_indices
global dir1
dir1 = '/home/shreya/git/Thesis/graphtools/fc_data/'
handler = logging.handlers.WatchedFileHandler(
os.environ.get("LOGFILE", f"{dir1}/outputs/sample.log"))
formatter = logging.Formatter(logging.BASIC_FORMAT)
handler.setFormatter(formatter)
logger = logging.getLogger('info')
logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))
logger.addHandler(handler)
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
def load_files(dir, groups):
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
    combined_edges = combined_edges.loc[combined_nodes.index, :]
    # Number of subjects in each group
    #print(combined_edges.isna().any(axis=1))
    combined_edges.dropna(axis=0, inplace=True)
    combined_nodes = combined_nodes.loc[combined_edges.index]
    print('Nodes:', combined_nodes.shape)
    logger.info(f'Nodes: {combined_nodes.shape}')
    print('Edges:', combined_edges.shape)
    logger.info(f'Edges: {combined_edges.shape}')
    print(mapping)
    logger.info(f'{mapping}')
    print('Number of subjects in each group:\n', combined_edges.iloc[:, -1].value_counts())
    logger.info(f'Number of subjects in each group:\n {combined_edges.iloc[:, -1].value_counts()}s')
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

def fit_classifier(X_train, X_test, y_train, y_test, n_fold, index):
    cv_inner = KFold(n_splits=3, shuffle=True, random_state=1)
    # define the model
    clf = RandomForestClassifier(random_state=10, n_jobs=-1)
    # define search space

    space = {"max_features": [0.9, 1],
                "min_samples_split": [30, 40],
                "max_depth": [4, 5],
                "random_state": [5, 6],
                "criterion": ['gini', 'entropy']}
    # define search
    search = GridSearchCV(clf, space, scoring='balanced_accuracy', cv=cv_inner, refit=True)
    # execute search
    search.fit(X_train.loc[:, index], y_train)
    #print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    best_model = search.best_estimator_

    #test_score = best_model.score(X_test.loc[:, index], y_test)
    y_pred = best_model.predict(X_test.loc[:, index])
    test_score = balanced_accuracy_score(y_test, y_pred)

    params = search.best_params_
    logger.info(f'The score on the fold {n_fold} {round(test_score, 3)}')
    #print(f'The score on the fold {n_fold}:', round(test_score, 3))
    return test_score, params

# %%
def run_baseline(X_train, X_test, y_train, y_test, fscores, per, n_fold):
    # filter according to the fscores
    val = np.percentile(fscores, per)
    index = []

    for i in fscores.index:
        if fscores[i] >= val:

            index.append(i)
    test_score, params = fit_classifier(X_train, X_test, y_train, y_test, n_fold, index)
    return test_score, params, index

def cv_split(X_train, y_train, train_idx, test_cv_idx):
    scalar = StandardScaler()
    x_train_cv, y_train_cv = X_train.iloc[train_idx, :], y_train.iloc[train_idx]
    x_train_cv = pd.DataFrame(scalar.fit_transform(x_train_cv), columns=x_train_cv.columns)
    x_test_cv, y_test_cv = X_train.iloc[test_cv_idx, :], y_train.iloc[test_cv_idx]
    x_test_cv = pd.DataFrame(scalar.transform(x_test_cv), columns=x_test_cv.columns)

    # print('Shapes:', X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    # select only the upper triangular features due to symmetric matrix
    X_y_uppert = pd.concat([x_train_cv, y_train_cv], axis=1)
    fscores = fscore(X_y_uppert, class_col=y_train_cv.name)
    # print('Fscores shape', fscores.shape)
    return x_train_cv, y_train_cv, x_test_cv, y_test_cv, fscores

def find_greedy_features(X_train, X_test, y_train, y_test, n_fold, fscores, max_num_nodes):
    """

    @param fscores: the flattened fscores of the upper triangular matrix
    @param max_num_nodes: maximum number of nodes we want to preserve in the output graph
    @return:
    """
    graph = np.zeros((num_nodes, num_nodes))
    # write all the upper triangular parts into the graph!

    for key in dict_idx.keys():
        graph[dict_idx[key]] = fscores[key]
    # graph is currently upper triangular and needs to be converted into full matrix
    graph = graph + graph.transpose() - 2 * np.diag(graph.diagonal())  # excluding the diagonal
    graph = pd.DataFrame(graph)

    while len(graph.iloc[0]) > max_num_nodes:
        node = graph.sum(axis=1).argmin()
        graph = graph.drop([graph.iloc[node].name], axis=0)  # returns the row as series
        graph = graph.drop(graph.iloc[:, node].name, axis=1)  # graph.loc[:, node].name

    graph_idxs = []
    for u, v in it.product(graph.columns, repeat=2):
        for key in dict_idx.keys():
            if dict_idx[key] == (u, v):
                graph_idxs.append(key)
    test_score, params = fit_classifier(X_train, X_test, y_train, y_test, n_fold, graph_idxs)

    return test_score, params, graph_idxs




def plot_comparison():
    plt.plot(percentages, list(score_by_percentage.values()))
    plt.scatter(percentages, list(score_by_percentage.values()), label='baseline', alpha=0.5)
    #num_edges = np.array([0.5*(n*(n+1)) for n in num_nodes_preserved])
    #total_edges = (num_edges**2 + num_edges) * 0.5 * 0.01
    plt.plot(percentages, list(score_by_nodes.values()))
    plt.scatter(percentages, list(score_by_nodes.values()), label='greedy approach', alpha=0.5)
    plt.xlabel('Percentage of fscores selected')
    plt.ylabel('Performance of classifier')
    plt.legend()
    plt.savefig(os.path.join(dir1, 'outputs', 'comparison_performance_'+groups[0]+'_'+ groups[1]))
    plt.clf()

def evaluate(index, X_train, X_test, y_train, y_test, clf):
    """

    @param index:
    @param X_train:
    @param X_test:
    @param y_train:
    @param y_test:
    @param clf:
    @return:
    """
    scalar = StandardScaler()
    X_train = X_train.loc[:, index]
    X_test = X_test.loc[:, index]
    X_train = scalar.fit_transform(X_train)
    X_test = scalar.transform(X_test)
    clf.fit(X_train, y_train)
    logger.info(f'Final score on independent test set  {round(clf.score(X_test, y_test), 3)}')
    y_pred = clf.predict(X_test)
    return round(balanced_accuracy_score(y_test, y_pred), 3)


def main():
    global i, score_by_percentage, data_dir, num_nodes, dict_idx
    global groups, num_nodes_preserved, score_by_nodes
    data_dir = '/home/shreya/git/Thesis/graphtools/fc_data/data'
    for groups in [['hc', 'ad'], ['hc', 'mci']]:

        print('Performance for groups', groups)
        combined_edges, combined_nodes = load_files(data_dir, groups=groups)
        #fscore_hist(combined_edges)  # fscores of the edges
        #fscore_hist(combined_nodes)  # fscores of the node features
        num_nodes = len(combined_nodes.columns) - 1  # because there is one column for the label here
        assert not sum(combined_edges.isna().any())
        skf = StratifiedKFold(n_splits=5)

        dict_idx = find_indices(shape=num_nodes)
        indices = list(dict_idx.keys())
        X = combined_edges.iloc[:, indices] # removing the redunance of the features
        y = combined_edges.iloc[:, -1]
        fscores_all = fscore(pd.concat([X, y], axis=1), class_col=y.name)[:-1]
        # we do not need the fscores of label with itself

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
        X_test = pd.DataFrame(X_test, columns=X.columns)
        X_train = pd.DataFrame(X_train, columns=X.columns)
        y_train = pd.Series(y_train, name=y.name)
        y_test = pd.Series(y_test, name=y.name)

        global percentages
        percentages = [2, 5, 10, 50, 100]
        num_nodes_preserved = []
        for per in percentages:
            coeff = [1, 1, (-0.01)*(num_nodes**2 + num_nodes)*per]
            r = abs(round(np.roots(coeff)[1]))//1  #get the positive root
            num_nodes_preserved.append(r)
        score_by_percentage = {key: None for key in percentages}
        score_by_nodes = {node: None for node in num_nodes_preserved}

        for per, max_num_nodes in zip(percentages, num_nodes_preserved):
            base_scores = []
            base_params = []
            greedy_scores = []
            greedy_params = []
            i = 0
            for train_idx, test_cv_idx in skf.split(X_train, y_train):
                if i < 5:
                    x_train_cv, y_train_cv, x_test_cv, y_test_cv, fscores = cv_split(X_train, y_train,
                                                                                     train_idx, test_cv_idx)
                    # To do: ensure the train and test are split in a stratified way
                    score, params, base_idxs = run_baseline(x_train_cv, x_test_cv, y_train_cv,
                                                        y_test_cv, fscores[:-1], per, i)

                    base_scores.append(score)
                    base_params.append(params)
                    greedy_score, greedy_param, greedy_idxs = find_greedy_features(x_train_cv,
                                                                                   x_test_cv, y_train_cv, y_test_cv, i,
                                                                                    fscores_all, max_num_nodes)
                    # Number of features in max number of nodes
                    logger.info(f'The number of nodes {max_num_nodes} \n The number of edges preserved: '
                                f'{len(np.triu_indices(max_num_nodes)[0])}')
                    greedy_scores.append(greedy_score)
                    greedy_params.append(greedy_param)
                    # only the diagonal indices since we are considering the matrix to be symmetric
                    #graph_processing(fscores[:-1], num_nodes, node_weights, preserved_nodes=20)
                    i += 1
            avg_score_base, avg_score_greedy = np.mean(np.array(base_scores)), np.mean(np.array(greedy_scores))
            idx_base, idx_greedy = np.argmax(np.array(base_scores)), np.argmax(np.array(greedy_scores))
            base_params, greedy_params = base_params[idx_base], greedy_params[idx_greedy]
            logger.info(f'The average score on k-fold baseline {round(avg_score_base, 3)}'
                        f' and greedy graph {round(avg_score_greedy, 3)}')
            # now fit the best estimator from the above
            clf_base = RandomForestClassifier(**base_params, n_jobs=-1)
            score_by_percentage[per] = evaluate(base_idxs, X_train, X_test, y_train, y_test, clf_base)
            clf_greedy = RandomForestClassifier(**greedy_params, n_jobs=-1)
            score_by_nodes[max_num_nodes] = evaluate(greedy_idxs, X_train, X_test, y_train, y_test, clf_greedy)
        plot_comparison()



# %%
if __name__ == '__main__':
    main()

