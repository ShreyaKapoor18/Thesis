
from processing import *
from readfiles import *
from classification_refined import classify

# %%
''' Data computed for all 5 personality traits at once'''
  # labels for the computed subjects, data.index is the subject id
num = 84  # number of nodes in the graph
tri = int(num * (num + 1) * 0.5)  # we want only the upper diagonal parts since everything below diagonal is 0
labels = ['NEOFAC_A', 'NEOFAC_O', 'NEOFAC_C', 'NEOFAC_N', 'NEOFAC_E']
edge_names = ['mean_FA', 'mean_strl', 'num_streamlines']
big5 = ['Agreeableness', 'Openness', 'Conscientiousness', 'Neuroticism',
        'Extraversion']
mapping = {k: v for k, v in zip(big5, labels)}
mat = np.triu_indices(84)
mews = '/home/skapoor/Thesis/gmwcs-solver'
metrics = ['balanced_accuracy', 'accuracy', 'f1_weighted', 'roc_auc_ovr_weighted']
# note: right now the matrix whole is not scaled, for computing the fscores and correlation coeff it has to be so.
y_train = computed_subjects()
X_train = generate_combined_matrix(tri, list(y_train.index))  # need to check indices till here then convert to numpy array
y_test = test_subjects()
X_test = generate_test_data(tri, y_test.index)

# The labels i.e. the ones from unrestricted_files # the order in which the subjects
# %%but now we have to give it both in a split
# the split must be the same when we are comparing all the functions
# %%
# add choice for the features we want to use and slice the whole matrix accordingly, no need to change the processing.py
classify(['SVC', 'RF', 'MLP'], X_train, X_test, y_train, y_test, metrics, mapping)
# based on these hyperparameters, search the input files, the output files and the reduced number of edges and nodes
'''
Here we will need to take the threshold and degree as hyperparameters, change them and compute the result accordingly
maybe make the dictionary like dict['degree'] = val when the val is in a loop. We shall put all the options differently
'''
# %%
#run the classification on bases of graph
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
