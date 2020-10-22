from processing import *
from readfiles import *
from decision import filter_summary
from subgraphclass import make_solver_summary
#%%
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
edges = ['t_test', 'fscores', 'pearson']
# note: right now the matrix whole is not scaled, for computing the fscores and correlation coeff it has to be so.
y_train = computed_subjects()
X_train = generate_combined_matrix(tri, list(y_train.index))  # need to check indices till here then convert to numpy array
num_strls = X_train.iloc[:, 2 * tri:]
y_test = test_subjects()
X_test = generate_test_data(tri, y_test.index)
#%%
total = pd.concat([X_train, X_test], axis=0)
# in order to visualize the group averaged mean FA feature

total = total.iloc[:,2*tri:]
#%%
total = total.mean(axis=0)
a = np.zeros((84,84))
mat = np.triu_indices(84)
for i in range(tri):
    a[mat[0][i], mat[1][i]]= total[i]
#%%
def upper_triangular_to_symmetric(ut):
    ut += np.triu(ut, k=1).T
    return ut

a = upper_triangular_to_symmetric(a)
f = open('visualization/group_averaged_numstrls.txt', 'w')
for i in range(84):
    for j in range(84):
        f.write(str(a[i,j]) + ' ')
    f.write('\n')
f.close()