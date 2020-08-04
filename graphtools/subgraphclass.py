from inputgraphs import *  # for importing the function names exactly
from processing import *
from readfiles import computed_subjects
from read_graphs import train_from_reduced_graph
from classification import run_classification
from paramopt import graph_options
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
fscores = hist_fscore(data, whole, labels, big5, edge_names, tri)
corr = hist_correlation(data, whole, labels, edge_names, big5, tri)
feature_type = 'mean_FA'
target = 'Agreeableness'
target_col = data[mapping[target]]
#%%

nested_outputdirs(mews='/home/skapoor/Thesis/gmwcs-solver')
with open('outputs/combined_params.json', 'r') as f:
    best_params = json.load(f)
    i = big5.index(target)
    index = list(range(tri))  # since we are using one feature at a time
    # for choice in ['qcut', 'median', 'throw median']
    # Let's say we only choose the throw median choice, because it is the one that makes more sense
    choice = 'throw median'  # out of all these we will use these particular choices only!
    X, y = data_splitting(choice, index, whole, target_col)  # this X is for random forests training
    params = best_params['RF'][target]["100"][choice]  # maybe use the parameters that work the best for top 5%
    feature_imp = train_with_best_params('RF', params, X, y)

    X_train, X_test, y_train, y_test = train_test_split(X,y)
    #train a graph based only on the training set
    #get fscores for all the features, only based on the training set.
    # maybe I can make a graph class based on this and initialize on the basis of the properties