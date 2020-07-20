from typing import List
from inputgraphs import *  # for importing the function names exactly
from processing import *
from readfiles import computed_subjects
from read_graphs import train_from_combined_graph
from classification import run_classification

# %%
data = computed_subjects()  # labels for the computed subjects, data.index is the subject id
num = 84  # number of nodes in the graph
tri = int(num * (num + 1) * 0.5)  # we want only the upper diagonal parts since everything below diagonal is 0
whole = generate_combined_matrix(tri, list(data.index))  # need to check indices till here then convert to numpy array
assert list(whole.index) == list(data.index)
# The labels i.e. the ones from unrestricted_files # the order in which the subjects
# were traversed is according to that of the index
# %%
labels: List[str] = ['NEOFAC_A', 'NEOFAC_O', 'NEOFAC_C', 'NEOFAC_N', 'NEOFAC_E']
edge_names = ['mean_FA', 'mean strl', 'num streamlines']
big5 = ['Agreeableness', 'Openness', 'Conscientiousness', 'Neuroticism',
        'Extraversion']  # maybe instead of putting these labels together we can put one a time as we are doing in the case

mapping = {k: v for k, v in zip(big5, labels)}
# before
metrics = ['balanced_accuracy', 'accuracy', 'f1_weighted', 'roc_auc_ovr_weighted']
options = dict(data=data, whole=whole, label=mapping['Agreeableness'], big5=['Agreeableness'], edge_names=edge_names,
               tri=tri)
fscores = hist_fscore(**options)
corr = hist_correlation(**options)
# without taking the edge type into consideration
new_fscores = np.reshape(fscores, (fscores.shape[0], fscores.shape[1] * fscores.shape[2]))
# %%
mat = np.triu_indices(84)
mews = '/home/skapoor/Thesis/gmwcs-solver'
for k in ['edge_names', 'tri']:
    del options[k]
options['new_fscores'], options['metrics'] = new_fscores, metrics
# %%
run_classification(**options)
dict3 = {'fscores': fscores, 'mat': mat, 'big5': big5,
         'data': data, 'whole': whole,
         'labels': labels, 'corr': corr, 'mews': mews}

plotting_options = graph_options(color='red', node_size=3, line_color='white', linewidhts=0.1, width=1)
hyperparams = {'target': 'Agreeableness',
               'edge': 'pearson', 'threshold': 85, 'node_wts': 'max', 'tri': tri, 'degree': 1,
               'plotting_options': plotting_options}
# based on these hyperparameters, search the input files, the output files and the reduced number of edges and nodes
'''
Here we will need to take the threshold and degree as hyperparameters, change them and compute the result accordingly
maybe make the dictionary like dict['degree'] = val when the val is in a loop. We shall put all the options differently
'''

# %%
different_graphs(**dict3, **hyperparams)
# %%
# now we have to read these graphs, in order to read these graphs we will have to see which all edges are present
'''
fscores, mat, big5,personality_trait, data, edge,
                     whole, labels, corr, mews, threshold, feature, node_wts'''
dict3['metrics'] = metrics  # the metrics we want to use
train_from_combined_graph(**dict3, **hyperparams)
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
