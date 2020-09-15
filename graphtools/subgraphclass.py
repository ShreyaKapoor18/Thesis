import itertools
from scipy.stats import describe
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_ind #independent sample t-test
from graphclass import *
from metrics import fscore
from paramopt import graph_options
from processing import generate_combined_matrix
from readfiles import computed_subjects


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


# %%

 #
#maybe we can make this mean, median, mode or something on basis of features

def make_solver_summary(mapping, data, targets,mews, whole, tri):
    edges = ['train_dat', 't_test', 'fscores', 'pearson']
    node_wtsl = ['max', 'const', 'avg']
    metrics = ['balanced_accuracy', 'accuracy', 'f1_weighted', 'roc_auc_ovr_weighted']
    # note: right now the matrix whole is not scaled, for computing the fscores and correlation coeff it has to be so.
    feature_types = ['mean_FA', 'mean_strl', 'num_streamlines']
    factors = [1, 10, 100, 1000, 5, 3, 4]  # ee how the factors make a difference
    output_file = open('/home/skapoor/Thesis/gmwcs-solver/solver_summary.txt', 'w')
    plotting_options = graph_options(color='red', node_size=5, line_color='white', linewidhts=0.1, width=1)

    columns = ['Feature_type', 'Factor', 'Target','Edge', 'Node_weights', 'Subtracted_value',
               'Input_Graph_nodes', 'Input_Graph_edges', 'Input_graph_posedge_per',
                'Output_Graph_nodes', 'Output_Graph_edges', 'Output_graph_posedge_per',
               'Features_preserved_per']
    summary_data = []
    #%%
    for j, (feature_type, target, edge, node_wts,factor) in \
            enumerate(itertools.product(feature_types,targets,edges, node_wtsl, factors)):
        y = data[mapping[target]]
        if factor in [5,3,4]:
            sub_vals = [factor/(x) for x in [2, 3,5,7]]
        else:
            sub_vals = [factor/(x) for x in [2, 5, 10, 20, 25, 3]]
        for sub_val in sub_vals:
            print('*'*100)
            print('*' * 100, file=output_file)
            print(f'Case {j}:{feature_type},{target},{edge},{node_wts},{sub_val}')
            print(f'Case {j}:feature_type, target,edge, Node weights, Subtracted value', file=output_file)
            print(f'Case {j}:{feature_type},{target},{edge},{node_wts},{sub_val}', file=output_file)
            target_col = data[mapping[target]]

            if feature_type == 'mean_FA':
                X = whole.iloc[:, :tri]
            elif feature_type == 'mean_strl':
                X = whole.iloc[:, tri:2 * tri]
            elif feature_type == 'num_streamlines':
                X = whole.iloc[:, 2 * tri:]  # input one feature at a time
            print(f'the {feature_type} feature is being used; the shape of the matrix is:', whole.shape)

            nested_outputdirs(mews=mews)
            scalar = StandardScaler()
            X = pd.DataFrame(scalar.fit_transform(X), index=X.index)
            med = int(y.median())  # the median is tried based on the training set
            #print('median of the training data', med)
            y_cut = pd.qcut(y, 5, labels=False, retbins=True)[0]
            # we need to pass the non-binned values for effective pearson correlation calc.
            #print('The number of training subjects which are to be removed:', sum(y_train_l == 2))
            y_cut = y_cut[y_cut != 2]
            y_cut = y_cut // 3  # binarizing the values by removing the middle quartile
            X_cut = X.loc[y_cut.index]
            assert len(X_cut) == len(y_cut)

            summary_data.append([feature_type, factor, target, edge, node_wts, sub_val])

            stacked = pd.concat([X_cut, target_col], axis=1)
            cols = []
            cols.extend(range(X_cut.shape[1]))  # the values zero to the number of columns
            cols.append(target_col.name)
            stacked.columns = cols
            if edge == 'fscores':
                name = target_col.name
                stacked[name] = stacked[name] >= stacked[name].median()
                arr = fscore(stacked, class_col=target_col.name)[:-1]  # take this only from the training data
            if edge == 'pearson':
                arr = stacked.corr().iloc[:-1, -1]
            if edge == 'train_dat':
                arr = X.mean()
            if edge == 't_test':
                name = target_col.name
                stacked[name] = stacked[name] >= stacked[name].median()
                group0 = stacked[stacked[name] == 0]
                group1 = stacked[stacked[name] ==1]
                arr = []
                for i in range(X.shape[1]):
                    arr.append((-1)*np.log10(ttest_ind(group0.iloc[:, i], group1.iloc[:,i]).pvalue))
                arr = pd.DataFrame(arr, index=range(X.shape[1]))
            arr.fillna(0, inplace=True)
            arr = arr.abs()
            arr = pd.DataFrame(arr, index=arr.index)# scale the array according to the index
            arr = arr * factor
            arr = pd.DataFrame(arr, index=arr.index)
            input_graph = BrainGraph(edge, feature_type, node_wts, target)
            input_graph.make_graph(arr, sub_val)

            if node_wts == 'const':
                input_graph.set_node_labels(node_wts, const_val=0)
            else:
                input_graph.set_node_labels(node_wts)
            # input_graph.normalize_node_attr()

            input_graph.savefiles(mews)
            #input_graph.visualize_graph(mews, True, sub_val, plotting_options)

            print('Input graph\n', f'Nodes: {len(input_graph.nodes)}\n', f'Edges:{len(input_graph.edges)}\n',
                  f'Edges description:{describe(input_graph.edge_weights)}\n',
                  f'Nodes description:{describe(input_graph.node_labels)}\n', file=output_file)
            summary_data[-1].extend([len(input_graph.nodes), len(input_graph.edges),
                                     100*sum([True for wt in input_graph.edge_weights if wt>0])/len(input_graph.edges)])
            # we want to calculate the strictly positive edges
            input_graph.run_solver('/home/skapoor/Thesis/gmwcs-solver')

            output_graph = BrainGraph(edge, feature_type, node_wts, target)
            reduced_feature_indices = output_graph.read_from_file(mews)

            #output_graph.visualize_graph(mews, False, sub_val, plotting_options)
            if output_graph.node_labels != [] and output_graph.edge_weights != []:
                print('Output graph\n', f'Nodes: {len(output_graph.nodes)}\n', f'Edges:{len(output_graph.edges)}\n',
                      f'Edges description:{describe(output_graph.edge_weights)}\n',
                      f'Nodes description:{describe(output_graph.node_labels)}\n', file=output_file)
                print(f'Percentage of features preserved {round(len(output_graph.edges) * 100 / len(input_graph.edges),3)}',
                      file=output_file)
                summary_data[-1].extend([len(output_graph.nodes), len(output_graph.edges),
                                     100* sum([True for wt in output_graph.edge_weights if wt>0])/len(output_graph.edges),
                                         round(len(output_graph.edges) * 100 / len(input_graph.edges),3)])
                f = open(f'{mews}/outputs/solver/{input_graph.filename}')
                solveroutput = f.read()
                print('-'*100)
                print(f'Solver output:\n {solveroutput}\n', file=output_file)
                print('-'*100)
                f.close()
                print('Describing the node labels of the output graph', describe(output_graph.node_labels))
                print('Describing the edge weights of the output graph', describe(output_graph.edge_weights))
            else:
                summary_data[-1].extend(['', '', '', ''])
        os.remove(f'{mews}/outputs/edges/{input_graph.filename}')
        os.remove(f'{mews}/outputs/solver/{input_graph.filename}')
        if os.path.exists(f'{mews}/outputs/edges/{input_graph.filename}.out'):
            os.remove(f'{mews}/outputs/edges/{input_graph.filename}.out')
        os.remove(f'{mews}/outputs/nodes/{input_graph.filename}')
        if os.path.exists(f'{mews}/outputs/nodes/{input_graph.filename}.out'):
            os.remove(f'{mews}/outputs/nodes/{input_graph.filename}.out')

    output_file.close()
    df = pd.DataFrame(summary_data, columns=columns)
    df.to_csv('/home/skapoor/Thesis/graphtools/outputs/csvs/summary.csv')