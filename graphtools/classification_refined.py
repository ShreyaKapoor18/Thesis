from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from metrics import fscore, compute_scores
from metrics import stratify_sampling, plot_grid_search
from paramopt import get_distributions
from scipy.stats import ttest_ind
import itertools
from graphclass import *
import copy

# %%
mews = "/home/skapoor/Thesis/gmwcs-solver"
tri = len(np.triu_indices(84)[0])


def process_raw(X_train, X_test, y_train, feature):
    scalar2 = StandardScaler()
    print('feature selection')
    # assert len(np.unique(y_train)) > 2 #to make sure that we are getting the unbinned personality traits
    # print('Initial X_train, y_train, X_val, y_val',  X_train.shape, y_train.shape, X_val.shape, y_val.shape)
    X_train = pd.DataFrame(scalar2.fit_transform(X_train), index=X_train.index)
    X_test = pd.DataFrame(scalar2.transform(X_test), index=X_test.index)

    stacked = pd.concat([X_train, y_train], axis=1)
    cols = []
    cols.extend(range(X_train.shape[1]))  # the values zero to the number of columns
    cols.append(y_train.name)
    stacked.columns = cols
    if feature == 'fscore':
        name = y_train.name
        stacked[name] = stacked[name] >= stacked[name].median()
        arr = fscore(stacked, class_col=name)[:-1]
        # fscore is different for the multiclass and binary case; has been incorporated above
    if feature == 'pearson':
        arr = stacked.corr().iloc[:-1, -1]
    if feature == 't_test':
        name = y_train.name
        stacked[name] = stacked[name] >= stacked[name].median()
        group0 = stacked[stacked[name] == 0]
        group1 = stacked[stacked[name] == 1]
        arr = []
        for i in range(X_train.shape[1]):
            arr.append(ttest_ind(group0.iloc[:, i], group1.iloc[:, i]).pvalue)
    arr = pd.DataFrame(arr)
    arr.fillna(0, inplace=True)
    return X_train, X_test, arr


def transform_features(X_train, X_test, y_train, per, feature):
    X_train, X_test, arr = process_raw(X_train, X_test, y_train, feature)
    arr = np.array(arr)
    val = np.nanpercentile(arr, 100 - per)
    index = np.where(arr >= val)
    X_train = X_train.iloc[:, index[0]]
    X_test = X_test.iloc[:, index[0]]

    assert list(X_train.index) == list(y_train.index)
    return X_train, X_test, arr


def solver(X_train, X_test, y_train, feature, node_wts=None, target=None, edge=None,factor=None, sub_val=None):
    X_train, X_test, arr = process_raw(X_train, X_test, y_train, feature)
    arr = arr.abs()
    arr = pd.DataFrame(arr, index=arr.index)  # scale the array according to the index
    arr = arr * factor
    arr = pd.DataFrame(arr, index=arr.index)
    input_graph = BrainGraph(edge, feature, node_wts, target)
    input_graph.make_graph(arr, sub_val)
    if node_wts == 'const':
        input_graph.set_node_labels(node_wts, const_val=0)
    else:
        input_graph.set_node_labels(node_wts)
    input_graph.savefiles(mews)
    input_graph.run_solver(mews)
    output_graph = BrainGraph(edge, feature, node_wts, target)
    reduced_feature_indices = output_graph.read_from_file(mews)

    if output_graph.node_labels != [] and output_graph.edge_weights != []:
        X_train = X_train.iloc[:, reduced_feature_indices]
        X_test = X_test.iloc[:, reduced_feature_indices]
    os.remove(f'{mews}/outputs/edges/{input_graph.filename}')
    os.remove(f'{mews}/outputs/solver/{input_graph.filename}')
    if os.path.exists(f'{mews}/outputs/edges/{input_graph.filename}.out'):
        os.remove(f'{mews}/outputs/edges/{input_graph.filename}.out')
    os.remove(f'{mews}/outputs/nodes/{input_graph.filename}')
    if os.path.exists(f'{mews}/outputs/nodes/{input_graph.filename}.out'):
        os.remove(f'{mews}/outputs/nodes/{input_graph.filename}.out')
    return X_train, X_test, output_graph.edge_weights, arr


def cross_validation(classifier, X_train, y_train, X_test, y_test, metrics, refit_metric):
    clf, distributions = get_distributions(classifier, True, None)
    rcv = RandomizedSearchCV(clf, distributions, random_state=55, scoring=metrics,
                             refit=refit_metric, cv=5, n_iter=200,
                             n_jobs=-1)
    # this is already producing 5 folds so we need to do something different?
    print('starting to train')
    search = rcv.fit(X_train, y_train)
    clf_out = search.best_estimator_
    # plot_grid_search(search.cv_results_, refit_metric)  # need to see what we will plot or not

    clf_out.fit(X_train, y_train)
    y_pred = clf_out.predict(X_test)
    assert len(y_pred) == len(X_test)
    outer_test = compute_scores(y_test, y_pred, clf_out.predict_proba(X_test)[:, 1], metrics=metrics)
    ytrain_pred = clf_out.predict(X_train)
    outer_train = compute_scores(y_train, ytrain_pred, clf_out.predict_proba(X_train)[:, 1], metrics=metrics)
    print('Test results', outer_test)
    return outer_train, outer_test


# %%
def classify(classifiers, X_train, X_test, y_train, y_test, metrics, mapping):
    # try making a csv file for the same

    cols_base = ['Classifier', 'Target', 'Choice', 'Edge', 'Feature Selection', 'Type of feature', 'Percentage',
                 'Refit Metric']
    cols_solver = copy.deepcopy(cols_base)
    cols_solver.extend(['Node_weights', 'Factor', 'Subtracted_value', 'Num edges', '% Positive edges'])
    percentages = [2, 5, 10, 50, 100]
    cols_base.extend(metrics)
    cols_solver.extend(metrics)
    results_base = []
    results_solver = []
    s_params = pd.read_csv('/home/skapoor/Thesis/gmwcs-solver/outputs/solver/filtered.csv')
    for i in range(len(s_params)):
        for classifier, params, feature_selection, choice, refit_metric in itertools.product(classifiers,  [s_params.iloc[i,:6]],
                                                                                             ['baseline', 'solver'],['test throw median', 'keep median'],
                                                                                             metrics):
            target = params['Target']
            factor = params['Factor']
            solver_edge = params['Edge']
            edge = params['Feature_type']
            solver_node_wts = params['Node_weights']
            sub_val = params['Subtracted_value']
            y_train_l = y_train[mapping[target]]
            y_test_l = y_test[mapping[target]]
            print('-' * 100)
            print(classifier, target, edge, feature_selection,choice, refit_metric, feature_selection)
            if edge == 'mean_FA':
                X_train_l = X_train.iloc[:, :tri]
                X_test_l = X_test.iloc[:, :tri]
            elif edge == 'mean_strl':
                X_train_l = X_train.iloc[:, :tri]
                X_test_l = X_test.iloc[:, :tri]
            elif edge == 'num_streamlines':
                X_train_l = X_train.iloc[:, 2 * tri:]  # input one feature at a time
                X_test_l = X_test.iloc[:, 2 * tri:]
            assert len(X_train) == len(y_train)
            med = int(y_train_l.median())  # the median is tried based on the training set
            print('median of the training data', med)
            y_train_l = pd.qcut(y_train_l, 5, labels=False, retbins=True)[0]
            # we need to pass the non-binned values for effective pearson correlation calc.
            print('The number of training subjects which are to be removed:', sum(y_train_l == 2))
            y_train_l = y_train_l[y_train_l != 2]
            y_train_l = y_train_l // 3  # binarizing the values by removing the middle quartile
            X_train_l = X_train_l.loc[y_train_l.index]
            assert len(X_train_l) == len(y_train_l)
            print('The choice that we are using', choice)
            if choice == 'test throw median':
                # removing subjects that are close to the median of the training data
                print(sum(abs(y_test_l - med) <= 1), 'is the number of test subjects with labels '
                                                     'within difference of 1.0 from the median value')
                # length_sub = sum(abs(y_test - med) <= 1)
                y_test_l = y_test_l[abs(y_test_l - med) > 1]  # maybe most of the values are close to the median
                y_test_l = y_test_l >= med  # binarizing the label check for duplicates
                X_test_l = X_test_l.loc[list(set(y_test_l.index))]
                # making sure that the training data is also for the same subjects
                assert len(X_test_l) == len(y_test_l)
            elif choice == 'keep median':
                y_test_l = y_test_l >= med  # we just binarize it and don't do anything else
            # now we do the cross validation search
            if feature_selection == 'solver':

                X_train_l, X_test_l, edge_wts, arr = solver(X_train_l, X_test_l, y_train_l, solver_edge,
                                                            solver_node_wts, target, edge, factor, sub_val)
                train_res, test_res = cross_validation(classifier, X_train_l, y_train_l, X_test_l, y_test_l,
                                                       metrics, refit_metric)
                results_solver.append([classifier, target, choice, edge, feature_selection, solver_edge, len(edge_wts)*100/tri,
                                      refit_metric, solver_node_wts, factor, sub_val,
                                       len(edge_wts), sum([edge>0 for edge in edge_wts])*100/len(edge_wts)])

                for metric in metrics:
                    results_solver[-1].extend([test_res[metric]])

            elif feature_selection == 'baseline':
                for per in percentages:
                    print('the percentages being used are', per)
                    X_train_l, X_test_l, arr = transform_features(X_train_l, X_test_l, y_train_l, per, solver_edge)
                    train_res, test_res = cross_validation(classifier, X_train_l, y_train_l, X_test_l, y_test_l, metrics,
                                                           refit_metric)
                    results_base.append([classifier, target, choice, edge, feature_selection, solver_edge, per, refit_metric])
                    for metric in metrics:
                        results_base[-1].extend([test_res[metric]])

    file1 = pd.DataFrame(results_base, index=cols_base)
    file1.to_csv(f'{mews}/outputs/csvs/results_baseline.csv')
    file2 = pd.DataFrame(results_solver, index=cols_solver)
    file2.to_csv(f'{mews}/outputs/csvs/results_solver.csv')
