from inputgraphs import *  # for importing the function names exactly
from processing import *
from readfiles import computed_subjects
from read_graphs import train_from_reduced_graph
from classification import run_classification
from paramopt import graph_options
from sklearn.model_selection import train_test_split
# %%
'''Data computed for all 5 personality traits at once'''
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
metrics = ['balanced_accuracy', 'accuracy', 'f1_weighted', 'roc_auc_ovr_weighted']

#for the classification step
target = input('Enter the label you want to be classified')
target_col = data[mapping[target]]
feature_type = input('Enter the feature type you want for the graph')
feature_type = 'mean_FA'
if feature_type == 'mean_FA':
    whole = whole.iloc[:, :tri]
elif feature_type == 'mean_strl':
    whole = whole.iloc[:, tri:2*tri]
elif feature_type == 'num_streamlines':
    whole = whole.iloc[:, 2*tri:]  #input one feature at a time
# The labels i.e. the ones from unrestricted_files # the order in which the subjects

