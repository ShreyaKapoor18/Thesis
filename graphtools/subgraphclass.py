import itertools
from scipy.stats import describe
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_ind #independent sample t-test
from graphclass import *
from metrics import fscore
from readfiles import corresp_label_file

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

def filter_indices(whole, feature_type,tri):

def delete_files(mews, input_graph):
    os.remove(f'{mews}/outputs/edges/{input_graph.filename}')
    os.remove(f'{mews}/outputs/solver/{input_graph.filename}')
    if os.path.exists(f'{mews}/outputs/edges/{input_graph.filename}.out'):
        os.remove(f'{mews}/outputs/edges/{input_graph.filename}.out')
    os.remove(f'{mews}/outputs/nodes/{input_graph.filename}')
    if os.path.exists(f'{mews}/outputs/nodes/{input_graph.filename}.out'):
        os.remove(f'{mews}/outputs/nodes/{input_graph.filename}.out')



def make_solver_summary(mapping, data, targets,mews, whole, tri, num_strls):
    edges = ['pearson']
    node_wtsl = ['const']

    metrics = ['balanced_accuracy', 'accuracy', 'f1_weighted', 'roc_auc_ovr_weighted']
    # note: right now the matrix whole is not scaled, for computing the fscores and correlation coeff it has to be so.
    feature_types = ['mean_FA', 'mean_strl', 'num_streamlines']
    # see how the factors make a difference
    output_file = open('/home/skapoor/Thesis/gmwcs-solver/solver_summary.txt', 'w')

    columns = ['Feature_type','Target','Edge', 'Node_weights',
               'Input_Graph_nodes', 'Input_Graph_edges', 'Input_graph_posedge_per',
                'Output_Graph_nodes', 'Output_Graph_edges', 'Output_graph_posedge_per',
               'Features_preserved_per']
    summary_data = []
    for j, (feature_type, target, edge, node_wts) in \
            enumerate(itertools.product(feature_types,targets,edges, node_wtsl)):
        val = -0.1

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
        strls_num_train = strls_num_train.mean(axis=0, skipna=True)
        arr = solver_edge_filtering(edge, X_cut, y)
        arr.fillna(0, inplace=True)
        arr = arr.abs()
        arr = arr.round(3)
        for num_nodes in [5,10,15,20,25,30]:
            print('*' * 100)
            print('*' * 100, file=output_file)
            print(f'Case:{feature_type},{target},{edge},{node_wts},{num_nodes}')
            print(f'Case:feature_type, target,edge, Node weights, Num_nodes', file=output_file)
            print(f'Case:{feature_type},{target},{edge},{node_wts}, {num_nodes}', file=output_file)
            input_graph, summary= input_graph_processing(arr, edge, feature_type, node_wts, target,
                                                         output_file,mews,val, strls_num_train, num_nodes)
            summary_data.append([feature_type, target, edge, val])
            summary_data[-1].extend(summary)
            #print('the degree of all the nodes', input_graph.degree(input_graph.nodes))
            output_graph, summary_out = output_graph_processing(input_graph, edge, feature_type, node_wts, val,
                                                           target, mews, output_file, num_nodes)
            summary_data[-1].extend(summary_out)


    output_file.close()
    df = pd.DataFrame(summary_data, columns=columns)
    df.to_csv('/home/skapoor/Thesis/graphtools/outputs/csvs/summary.csv')
    os.remove(f'{mews}/outputs/edges/{input_graph.filename}')
    os.remove(f'{mews}/outputs/solver/{input_graph.filename}')
    if os.path.exists(f'{mews}/outputs/edges/{input_graph.filename}.out'):
        os.remove(f'{mews}/outputs/edges/{input_graph.filename}.out')
    os.remove(f'{mews}/outputs/nodes/{input_graph.filename}')
    if os.path.exists(f'{mews}/outputs/nodes/{input_graph.filename}.out'):
        os.remove(f'{mews}/outputs/nodes/{input_graph.filename}.out')



def make_solver_summary(mapping, data, targets,mews, whole, tri, num_strls):
    edges = ['pearson']
    node_wtsl = ['const']

    metrics = ['balanced_accuracy', 'accuracy', 'f1_weighted', 'roc_auc_ovr_weighted']
    # note: right now the matrix whole is not scaled, for computing the fscores and correlation coeff it has to be so.
    feature_types = ['mean_FA', 'mean_strl', 'num_streamlines']
    # see how the factors make a difference
    output_file = open('/home/skapoor/Thesis/gmwcs-solver/solver_summary.txt', 'w')

    columns = ['Feature_type','Target','Edge', 'Node_weights',
               'Input_Graph_nodes', 'Input_Graph_edges', 'Input_graph_posedge_per',
                'Output_Graph_nodes', 'Output_Graph_edges', 'Output_graph_posedge_per',
               'Features_preserved_per']
    summary_data = []
    for j, (feature_type, target, edge, node_wts) in \
            enumerate(itertools.product(feature_types,targets,edges, node_wtsl)):
        val = -0.1

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
        strls_num_train = strls_num_train.mean(axis=0, skipna=True)
        arr = solver_edge_filtering(edge, X_cut, y)
        arr.fillna(0, inplace=True)
        arr = arr.abs()
        arr = arr.round(3)
        for num_nodes in [5,10,15,20,25,30]:
            print('*' * 100)
            print('*' * 100, file=output_file)
            print(f'Case:{feature_type},{target},{edge},{node_wts},{num_nodes}')
            print(f'Case:feature_type, target,edge, Node weights, Num_nodes', file=output_file)
            print(f'Case:{feature_type},{target},{edge},{node_wts}, {num_nodes}', file=output_file)
            input_graph, summary= input_graph_processing(arr, edge, feature_type, node_wts, target,
                                                         output_file,mews,val, strls_num_train, num_nodes)
            summary_data.append([feature_type, target, edge, val])
            summary_data[-1].extend(summary)
            #print('the degree of all the nodes', input_graph.degree(input_graph.nodes))
            output_graph, summary_out = output_graph_processing(input_graph, edge, feature_type, node_wts, val,
                                                           target, mews, output_file, num_nodes)
            summary_data[-1].extend(summary_out)


    output_file.close()
    df = pd.DataFrame(summary_data, columns=columns)
    df.to_csv('/home/skapoor/Thesis/graphtools/outputs/csvs/summary.csv')
