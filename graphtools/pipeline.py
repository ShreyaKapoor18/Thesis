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
#%%
'''

if feature_type == 'mean_FA':
    whole = whole.iloc[:, :tri]
elif feature_type == 'mean_strl':
    whole = whole.iloc[:, tri:2*tri]
elif feature_type == 'num_streamlines':
    whole = whole.iloc[:, 2*tri:] # input one feature at a time
print (f'the {feature_type} feature is being used; the shape of the matrix is:', whole.shape)
'''
# The labels i.e. the ones from unrestricted_files # the order in which the subjects
#%%
run_classification(whole, metrics, 'Agreeableness', data[mapping['Agreeableness']], 'fscore')
#the split must be the same when we are comparing all the functions

#%%
#add choice for the features we want to use and slice the whole matrix accordingly, no need to change the processing.py

plotting_options = graph_options(color='red', node_size=5, line_color='white', linewidhts=0.1, width=1)
hyperparams = {'target': 'Agreeableness',
               'edge': 'pearson', 'threshold': 65, 'node_wts': 'max', 'tri': tri, 'degree': 1,
               'plotting_options': plotting_options}
# based on these hyperparameters, search the input files, the output files and the reduced number of edges and nodes
'''
Here we will need to take the threshold and degree as hyperparameters, change them and compute the result accordingly
maybe make the dictionary like dict['degree'] = val when the val is in a loop. We shall put all the options differently
'''
# %%
'''This one is generating graphs for all three feature types at once maybe we can reduce this'''
different_graphs(fscores=fscores, mat=mat, big5=big5, whole=whole, corr=corr, mews=mews, target_col=data[mapping['Agreeableness']],
                 feature_type=feature_type, **hyperparams)
#also compute the fscores for the training data only here, make sure here also the splitting and the training data, random state is the same
# %%
# now we have to read these graphs, in order to read these graphs we will have to see which all edges are present
del hyperparams['threshold']
del hyperparams['tri']
#%%
hyperparams['feature_type'] = feature_type
train_from_reduced_graph(metrics=metrics, target_col=data[mapping['Agreeableness']],
                          mat=mat, big5=big5, whole=whole, mews=mews, **hyperparams)
'''
To do:
def pipeline_summary():
    # fscores based on the label
    # pearson correlation based on the target variable
    # classification for the target variable
    # input to the solver
    # output from the solver
    # classification on the basis of the solver output
'''
