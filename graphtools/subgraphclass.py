from inputgraphs import *  # for importing the function names exactly
from processing import *
from readfiles import computed_subjects
from read_graphs import train_from_reduced_graph
from classification import run_classification
from paramopt import graph_options
from graphclass import *
from metrics import *
# %%
''' Data computed for all 5 personality traits at once'''
data = computed_subjects()  # labels for the computed subjects, data.index is the subject id
num = 84  # number of nodes in the graph
tri = int(num * (num + 1) * 0.5)  # we want only the upper diagonal parts since everything below diagonal is 0
whole = generate_combined_matrix(tri, list(data.index))  # need to check indices till here then convert to numpy array
assert list(whole.index) == list(data.index)
labels = ['NEOFAC_A', 'NEOFAC_O', 'NEOFAC_C', 'NEOFAC_N', 'NEOFAC_E']
edge_names = ['mean_FA', 'mean_strl', 'num_streamlines']
big5 = ['Agreeableness', 'Openness', 'Conscientiousness', 'Neuroticism',
        'Extraversion']
mapping = {k: v for k, v in zip(big5, labels)}
mat = np.triu_indices(84)
mews = '/home/skapoor/Thesis/gmwcs-solver'
# before
# this is what is supposed to be done
metrics = ['balanced_accuracy', 'accuracy', 'f1_weighted', 'roc_auc_ovr_weighted']
#note: right now the matrix whole is not scaled, for computing the fscores and correlation coeff it has to be so.
feature_type = 'mean_FA'
target = 'Agreeableness'
target_col = data[mapping[target]]
edge = 'fscores'
node_wts = 'max'
threshold = 85
plotting_options = graph_options(color='red', node_size=5, line_color='white', linewidhts=0.1, width=1)
#%%
if feature_type == 'mean_FA':
    whole = whole.iloc[:, :tri]
elif feature_type == 'mean_strl':
    whole = whole.iloc[:, tri:2*tri]
elif feature_type == 'num_streamlines':
    whole = whole.iloc[:, 2*tri:] # input one feature at a time
print (f'the {feature_type} feature is being used; the shape of the matrix is:', whole.shape)
#%%

nested_outputdirs(mews='/home/skapoor/Thesis/gmwcs-solver')
with open(f'outputs/dicts/{target}_combined_params.json', 'r') as f:
    best_params = json.load(f)
    i = big5.index(target)
    index = list(range(tri))  # if we are using only one type of feature at a time, lets say mean FA
    # since we are using one feature at a time
    # for choice in ['qcut', 'median', 'throw median']
    # Let's say we only choose the throw median choice, because it is the one that makes more sense
    choice = 'throw median'  # out of all these we will use these particular choices only!
    X, y = data_splitting(choice, index, whole, target_col)  # this X is for random forests training
    params = best_params['RF'][choice]["100"]  # maybe use the parameters that work the best for top 5%
    feature_imp = train_with_best_params('RF', params, X, y)

    X_train, X_test, y_train, y_test = train_test_split(X,y)
    #train a graph based only on the training set
    #get fscores for all the features, only based on the training set.
    # maybe I can make a graph class based on this and initialize on the basis of the properties
    stacked = pd.concat([X_train, y_train], axis=1)
    cols = []
    cols.extend(range(X_train.shape[1]))  # the values zero to the number of columns
    cols.append(target_col.name)
    stacked.columns = cols
    if edge == 'fscores':
        arr = fscore(stacked, class_col=target_col.name)[:-1] #take this only from the training data
    if edge == 'pearson':
        arr = stacked.corr().iloc[:, -1]
    if edge == 'feature_importance':
        arr = feature_imp
        # assert type(arr) == np.ndarray
    print('type of array', type(arr))
    arr.fillna(0, inplace=True)
    arr = np.absolute(arr)  # need to standardize after taking the absolute value
    thresh = np.percentile(arr, threshold)
    print(f'Threshold value according to {threshold} percentile: {thresh}')
    index2 = np.where(arr <= thresh)
    print('indexes', index2)
    xs = index2[0]

    # removed wrong indexing
    for p in range(len(index2[0])):
        arr[xs[p]] = 0
    # we want to standardize the whole array together
    # the values are x.y and the values in itself
    # standardization of the array itself, need to preserve the non zero parts only
    arr = pd.DataFrame(arr)
    stdvals = arr[arr != 0]
    stdvals = (stdvals - stdvals.mean()) / stdvals.std()
    arr[arr != 0] = stdvals # before giving these values as edge values
    # try for for different types, one feature at a time maybe and then construct graph?
    nodes = set()
    edge_attributes = []
    for j in range(len(mat[0])):
        value = float(arr.iloc[j])
        if abs(value) > thresh:
            edge_attributes.append((mat[0][j], mat[1][j], value))
            nodes.add(mat[0][j])  # add only the nodes which have corresponding edges
            nodes.add(mat[1][j])
    # mean for the scores of three different labels
    assert nodes != None
    input_graph = BrainGraph(edge, feature_type, node_wts, target)
    input_graph.add_nodes_from(nodes)
    input_graph.add_weighted_edges_from(edge_attributes)
    input_graph.set_node_labels('max')
    input_graph.normalize_node_attr()
    input_graph.savefiles(mews, degree=2)
    input_graph.visualize_graph(mews, True, threshold, plotting_options)

    output_graph = BrainGraph(edge, feature_type, node_wts, target)
    output_graph.read_from_file(mews)
    output_graph.visualize_graph(mews, False, threshold, plotting_options)
    #get nodes and edges of this graph
    #train algorithm accordingly
    output_graph.edges

