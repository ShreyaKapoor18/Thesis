import itertools
from scipy.stats import describe
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_ind  # independent sample t-test
from graphclass import *
from metrics import fscore
from readfiles import corresp_label_file

#%%
tri = (84*85)/2
# %%
def nested_outputdirs(mews):  # make a separate directory for each label, easier to do comparisons

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
    if not os.path.exists(f'{mews}/outputs/figs'):
        os.mkdir(f'{mews}/outputs/figs')


def filter_indices(whole, feature_type, tri):
    if feature_type == 'mean_FA':
        X = whole.iloc[:, :tri]
    elif feature_type == 'mean_strl':
        X = whole.iloc[:, tri:2 * tri]
    elif feature_type == 'num_streamlines':
        X = whole.iloc[:, 2 * tri:]  # input one feature at a time
    print(f'the {feature_type} feature is being used; the shape of the matrix is:', whole.shape)
    return X


def solver_edge_filtering(edge, X_cut, y):
    stacked = pd.concat([X_cut, y], axis=1)
    cols = []
    cols.extend(range(X_cut.shape[1]))  # the values zero to the number of columns
    cols.append(y.name)
    stacked.columns = cols
    if edge == 'fscores' or edge == 'fscore':
        name = y.name
        stacked[name] = stacked[name] >= stacked[name].median()
        arr = fscore(stacked, class_col=y.name)[:-1]
        #arr = arr * 100  # take this only from the training data
    if edge == 'pearson':
        arr = stacked.corr().iloc[:-1, -1]
    if edge == 'train_dat':
        arr = X_cut.mean()
    if edge == 't_test':
        name = y.name
        stacked[name] = stacked[name] >= stacked[name].median()
        group0 = stacked[stacked[name] == 0]
        group1 = stacked[stacked[name] == 1]
        arr = []
        for i in range(X_cut.shape[1]):
            arr.append((-1) * np.log10(ttest_ind(group0.iloc[:, i], group1.iloc[:, i]).pvalue))
        arr = pd.DataFrame(arr, index=range(X_cut.shape[1]))
    return arr


def input_graph_processing(arr, edge, feature_type, node_wts, val, target,
                           output_file, mews, strls_num, max_num_nodes, thresh, avg_thresh):
    input_graph = BrainGraph(edge, feature_type, node_wts, target, max_num_nodes, val, thresh)
    if not os.path.exists(f'{mews}/outputs/edges/{input_graph.filename}.out')\
            and not os.path.exists(f'{mews}/outputs/nodes/{input_graph.filename}.out')\
            and not os.path.exists(f'{mews}/outputs/nodes/{input_graph.filename}')\
            and not os.path.exists(f'{mews}/outputs/edges/{input_graph.filename}'):
        input_graph.make_graph(arr, strls_num, thresh, avg_thresh)
        if node_wts == 'const':
            input_graph.set_node_labels(node_wts, const_val=val)
        else:
            input_graph.set_node_labels(node_wts)
        input_graph.savefiles(mews)
        input_graph.run_solver(mews, max_num_nodes=max_num_nodes)
    else:
        input_graph.read_from_file(mews, input_graph=True)
        if node_wts == 'const':
            input_graph.set_node_labels(node_wts, const_val=val)
        else:
            input_graph.set_node_labels(node_wts)

    print('Input graph\n', f'Nodes: {len(input_graph.nodes)}\n', f'Edges:{len(input_graph.edges)}\n', file=output_file)
    summary = [len(input_graph.nodes), len(input_graph.edges),
               100 * sum([True for wt in input_graph.edge_weights if wt > 0]) / len(input_graph.edges)]
    # we want to calculate the strictly positive edges

    return input_graph, summary


def output_graph_processing(input_graph, edge, feature_type, node_wts, target, mews,
                            output_file, max_num_nodes, val, thresh):
    output_graph = BrainGraph(edge, feature_type, node_wts, target, max_num_nodes, val, thresh)
    reduced_feature_indices = output_graph.read_from_file(mews, False)
    print('Output graph')
    print(f'Number of edges:{len(output_graph.edge_weights)} and Number of nodes:{len(output_graph.nodes)}')
    if output_graph.node_labels != [] and output_graph.edge_weights != []:
        print('Output graph\n', f'Nodes: {len(output_graph.nodes)}\n', f'Edges:{len(output_graph.edges)}\n',
              f'Edges description:{describe(output_graph.edge_weights)}\n', file=output_file)
        print(f'Percentage of features preserved {round(len(output_graph.edges) * 100 / len(input_graph.edges), 3)}',
              file=output_file)
        summary = [len(output_graph.nodes), len(output_graph.edges),
                   100 * sum([True for wt in output_graph.edge_weights if wt > 0]) / len(
                       output_graph.edges),
                   round(len(output_graph.edges) * 100 /tri, 3)]

        #dict_lut = corresp_label_file('fs_default.txt')
        #for node in output_graph.nodes:
        #    print(node, dict_lut[node + 1])  # since our node numbering starts from 0 and LUT starts with 1
    else:
        f = open(f'{mews}/outputs/solver/{input_graph.filename}')
        solveroutput = f.read()

        f.close()
        summary = ['', '', '', '']
    return output_graph, summary


def delete_files(mews, input_graph):
    os.remove(f'{mews}/outputs/edges/{input_graph.filename}')
    os.remove(f'{mews}/outputs/solver/{input_graph.filename}')
    if os.path.exists(f'{mews}/outputs/edges/{input_graph.filename}.out'):
        os.remove(f'{mews}/outputs/edges/{input_graph.filename}.out')
    os.remove(f'{mews}/outputs/nodes/{input_graph.filename}')
    if os.path.exists(f'{mews}/outputs/nodes/{input_graph.filename}.out'):
        os.remove(f'{mews}/outputs/nodes/{input_graph.filename}.out')


def make_solver_summary(edges, mapping, data, targets, mews, whole, tri, num_strls, avg_thresh, cat):

    node_wtsl = ['const']
    metrics = ['balanced_accuracy', 'accuracy', 'f1_weighted', 'roc_auc_ovr_weighted']
    # note: right now the matrix whole is not scaled, for computing the fscores and correlation coeff it has to be so.
    feature_types = ['mean_FA', 'mean_strl', 'num_streamlines']
    # see how the factors make a difference
    output_file = open('/home/skapoor/Thesis/gmwcs-solver/solver_summary.txt', 'w')

    columns = ['Feature_type', 'Target', 'Edge', 'Node_weights',
               'Input_Graph_nodes', 'Input_Graph_edges', 'Input_graph_posedge_per',
               'Output_Graph_nodes', 'Output_Graph_edges', 'Output_graph_posedge_per',
               'Features_preserved_per', 'ROI_strl_thresh']
    summary_data = []
    if avg_thresh == True:
        threshs = [0.01, 0.001, 0.005, 0.0055]
    else:
        threshs = [0]
    for j, (feature_type, target, edge, node_wts, thresh) in \
            enumerate(itertools.product(feature_types, targets, edges, node_wtsl, threshs)):
        val = -0.01
        if cat == True:
            y = data[mapping[target]]
            X = filter_indices(whole, feature_type, tri)
            nested_outputdirs(mews=mews)
            scalar = StandardScaler()
            X = pd.DataFrame(scalar.fit_transform(X), index=X.index)
            med = int(y.median())  # the median is tried based on the training set
            # print('median of the training data', med)
            y_cut = pd.qcut(y, 5, labels=False, retbins=True)[0]
            # we need to pass the non-binned values for effective pearson correlation calc.
            # print('The number of training subjects which are to be removed:', sum(y_train_l == 2))
            y_cut = y_cut[y_cut != 2]
            y_cut = y_cut // 3  # binarizing the values by removing the middle quartile
            X_cut = X.loc[y_cut.index]
            assert len(X_cut) == len(y_cut)
            strls_num_train = num_strls.loc[X_cut.index, :]
            arr = solver_edge_filtering(edge, X_cut, y)
        else:
            y = data[target]
            y = y.map({'M':0, 'F':1})
            X = filter_indices(whole, feature_type, tri)
            strls_num_train = num_strls.loc[X.index, :]
            arr = solver_edge_filtering(edge, X, y)
        if avg_thresh:
            strls_num_train = strls_num_train.mean(axis=0, skipna=True)
        else:
            strls_num_train = strls_num_train.all()

        arr.fillna(0, inplace=True)
        arr = arr.abs()
        arr = arr.round(3)
        #for num_nodes in [5, 10, 15, 20, 25, 30]:
        for num_nodes in range(5,11):
            print('*' * 100)
            print('*' * 100, file=output_file)
            print(f'Case:feature_type, target,edge, Node weights, Num_nodes, Thresh', file=output_file)
            print(f'Case:{feature_type},{target},{edge},{node_wts}, {num_nodes},{thresh}', file=output_file)
            input_graph, summary = input_graph_processing(arr, edge, feature_type, node_wts, val, target,
                                                          output_file, mews, strls_num_train, num_nodes, thresh, avg_thresh)
            summary_data.append([feature_type, target, edge, val])
            summary_data[-1].extend(summary)

            # print('the degree of all the nodes', input_graph.degree(input_graph.nodes))
            output_graph, summary_out = output_graph_processing(input_graph, edge, feature_type, node_wts, target, mews,
                                                                output_file, num_nodes, val, thresh)
            summary_data[-1].extend(summary_out)
            summary_data[-1].append(thresh)

        # output_graph.visualize_graph(mews, False, sub_val, plotting_options)
        #delete_files(mews, input_graph)  # so that the files from the previous run can be deleted


    output_file.close()
    df = pd.DataFrame(summary_data, columns=columns)
    df.to_csv('/home/skapoor/Thesis/graphtools/outputs/csvs/summary.csv')
