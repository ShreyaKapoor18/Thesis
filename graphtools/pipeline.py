from classification import *
from inputgraphs import *
from processing import *
from paramopt import *
from readfiles import computed_subjects
from read_graphs import train_from_combined_graph
# %%
data = computed_subjects()  # labels for the computed subjects, data.index is the subject id
num = 84  # number of nodes in the graph
tri = int(num * (num + 1) * 0.5)  # we want only the upper diagonal parts since everything below diagonal is 0
whole = generate_combined_matrix(tri, list(data.index))  # need to check indices till here then convert to numpy array
assert list(whole.index) == list(data.index)
# The labels i.e. the ones from unrestricted_files # the order in which the subjects
# were traversed is according to that of the index
# %%
labels = ['NEOFAC_A', 'NEOFAC_O', 'NEOFAC_C', 'NEOFAC_N', 'NEOFAC_E']
edge_names = ['mean_FA', 'mean strl', 'num streamlines']
big5 = ['Agreeableness', 'Openness', 'Conscientiousness', 'Neuroticism',
        'Extraversion']
metrics = ['balanced_accuracy', 'accuracy', 'f1_weighted', 'roc_auc_ovr_weighted']

dict1 = {'data': data, 'whole': whole,'labels': labels,
        'big5': big5, 'edge_names': edge_names, 'tri': tri}
fscores = hist_fscore(**dict1)
corr = hist_correlation(**dict1)

# without taking the edge type into consideration
new_fscores = np.reshape(fscores, (fscores.shape[0], fscores.shape[1] * fscores.shape[2]))

# %%
mat = np.triu_indices(84)
mews = '/home/skapoor/Thesis/gmwcs-solver'
dict2 = {'whole': whole, 'metrics': metrics, 'big5': big5,
         'data':data, 'new_fscores':new_fscores, 'labels' : labels}
dict3 = {'fscores': fscores, 'mat': mat, 'big5': big5,
         'data': data, 'whole': whole,
         'labels': labels, 'corr': corr, 'mews': mews, 'personality_trait': 'Openness',
         'edge': 'feature_importance', 'threshold': 85, 'node_wts': 'max', 'tri':  tri}
# %%
#run_classification(**dict2)
#%%
different_graphs(**dict3)
#%%
# now we have to read these graphs, in order to read these graphs we will have to see which all edges are present
'''
fscores, mat, big5,personality_trait, data, edge,
                     whole, labels, corr, mews, threshold, feature, node_wts'''
dict3['metrics'] = metrics
train_from_combined_graph(**dict3)