from graphclass import *
from paramopt import graph_options
from processing import generate_combined_matrix
from readfiles import computed_subjects
from metrics import fscore
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import json

from scipy.stats import describe
from sklearn.preprocessing import scale


# %%
def train_with_best_params(classifier, params, X, y):
    """
    Train the specified classifier with the best parameters obtained from
    Cross Validation
    """
    if classifier == 'RF':
        clf = RandomForestClassifier(**params)  # try if this method works so that don't have to use explicit arguments
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        clf.fit(X_train, y_train)
        clf.predict(X_test)
        return clf.feature_importances_  # put this as the edge weight in the graph
    return None


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
solver_summary = {}
metrics = ['balanced_accuracy', 'accuracy', 'f1_weighted', 'roc_auc_ovr_weighted']
#%%
# note: right now the matrix whole is not scaled, for computing the fscores and correlation coeff it has to be so.
feature_type = 'mean_FA'
target = 'Agreeableness'
target_col = data[mapping[target]]
edge = 'pearson'
node_wts = 'max'
threshold = 85
sub_val = 2
use_case = dict(feature_type=feature_type, edge=edge, node_wts=node_wts, subtracted_val=sub_val)

plotting_options = graph_options(color='red', node_size=5, line_color='white', linewidhts=0.1, width=1)
# %%
if feature_type == 'mean_FA':
    whole = whole.iloc[:, :tri]
elif feature_type == 'mean_strl':
    whole = whole.iloc[:, tri:2 * tri]
elif feature_type == 'num_streamlines':
    whole = whole.iloc[:, 2*tri:] # input one feature at a time
print (f'the {feature_type} feature is being used; the shape of the matrix is:', whole.shape)
#%%

nested_outputdirs(mews='/home/skapoor/Thesis/gmwcs-solver')
with open(f'/home/skapoor/Thesis/graphtools/outputs/dicts/{target}_combined_params.json', 'r') as f:
    best_params = json.load(f)
    i = big5.index(target)
    index = list(range(tri))  # if we are using only one type of feature at a time, lets say mean FA
    # since we are using one feature at a time
    # for choice in ['qcut', 'median', 'throw median']
    # Let's say we only choose the throw median choice, because it is the one that makes more sense
    choice = 'throw median'  # out of all these we will use these particular choices only!
   # this X is for random forests training
    X,y = whole, target_col

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    scalar = StandardScaler()
    X_train = pd.DataFrame(scalar.fit_transform(X_train), index=X_train.index)
    X_test = pd.DataFrame(scalar.transform(X_test), index=X_test.index)
    # train a graph based only on the training set
    # get fscores for all the features, only based on the training set.
    # maybe I can make a graph class based on this and initialize on the basis of the properties
    stacked = pd.concat([X_train, y_train], axis=1)
    cols = []
    cols.extend(range(X_train.shape[1]))  # the values zero to the number of columns
    cols.append(target_col.name)
    stacked.columns = cols
    if edge == 'fscores':
        arr = fscore(stacked, class_col=target_col.name)[:-1]  # take this only from the training data
    if edge == 'pearson':
        arr = stacked.corr().iloc[:-1, -1]

        # assert type(arr) == np.ndarray
    arr = pd.DataFrame(scale(arr), index=arr.index)
    input_graph = BrainGraph(edge, feature_type, node_wts, target)
    input_graph.make_graph(arr, sub_val)
    input_graph.set_node_labels(node_wts)
    # input_graph.normalize_node_attr()
    use_case['input_graph'] =dict(nodes=len(input_graph.subgraph.nodes), edges= input_graph.subgraph.edges)
    input_graph.savefiles(mews)
    input_graph.visualize_graph(mews, True, sub_val, plotting_options)
    print('Describing the node labels of the input graph', describe(input_graph.node_labels))
    print('Describing the edge weights of the input graph', describe(input_graph.edge_weights))
    input_graph.run_solver(mews)

    output_graph = BrainGraph(edge, feature_type, node_wts, target)
    reduced_feature_indices = output_graph.read_from_file(mews)
    output_graph.visualize_graph(mews, False, sub_val, plotting_options)
    if output_graph.node_labels != [] and output_graph.edge_weights != []:
        print('Describing the node labels of the output graph', describe(output_graph.node_labels))
        print('Describing the edge weights of the output graph', describe(output_graph.edge_weights))
    os.remove(f'{mews}/outputs/edges/{input_graph.filename}')
    os.remove(f'{mews}/outputs/edges/{input_graph.filename}.out')
    os.remove(f'{mews}/outputs/nodes/{input_graph.filename}')
    os.remove(f'{mews}/outputs/nodes/{input_graph.filename}.out')
    #get nodes and edges of this graph
    #train algorithm accordingly
    #X_train = X_train.iloc[:, reduced_feature_indices]
    #X_test = X_test.iloc[:, reduced_feature_indices]

solver_summary[target] = use_case

