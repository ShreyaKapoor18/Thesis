import numpy as np
import pandas as pd
from metrics import fscore
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from graphclass import BrainGraph
from metrics import diag_flattened_indices, find_indices
#%%
def make_hist(X_train, y_train, ax):
    '''
    Histogram to test the difference the scaling makes to the fscores.
    '''
    X_train_combined = pd.concat([X_train, y_train], axis=1)
    ax.hist(fscore(X_train_combined, class_col=len(X_train_combined.columns) - 1), label='before scaling',
               alpha=0.5)
    X_train_combined.iloc[:, :-1] = pd.DataFrame(scalar.fit_transform(X_train_combined.iloc[:, :-1]))
    ax.hist(fscore(X_train_combined, class_col=len(X_train_combined.columns) - 1), label='after scaling',
               alpha=0.5)
    ax.legend()
    return ax
#%%
names = ['FC_ad.npy', 'FC_hc.npy', 'FC_mci.npy', 'tau_ad.npy', 'tau_hc.npy', 'tau_mci.npy']
dir = '/home/shreya/git/Thesis/data'

for name in names:
    var = np.load(f'{dir}/{name}', allow_pickle=True)
    print(name)
    print (var.shape)

    # a few questions about the data
    # How many subjects?
    # Why are the shapes of the arrays different
    # What does each array represent
    # How many nodes were talked about? Maybe 52 nodes
    # Some have the node weights and the others have the edge weights
    # Maybe the node information is in the tau variables. 


    # the shapes are corresponding for fc, ad, hc and mci types maybe a grouping needs to be established accordingly

    # ad is for the alzheimer's patients
    # hc is for the healthy controls
    # mci is for? mild cognitive impairment (sort of the middle group for the onset of Alzheimer's disease)
    # I have to prepare a label for these. Each of these three has a different label associated with the
#%%
def load_files():
    combined_edges = []
    combined_nodes = []
    for group in ['ad', 'hc', 'mci']:

        edges = np.load(f'{dir}/FC_{group}.npy', allow_pickle=True)
        edges = edges.reshape(edges.shape[0], edges.shape[1]*edges.shape[2])
        edges = pd.DataFrame(edges)
        edges['label'] = group
        edges_l = np.array(edges)

        combined_edges.extend(edges_l)

        nodes = np.load(f'{dir}/tau_{group}.npy', allow_pickle=True)
        print(nodes.shape)
        nodes = pd.DataFrame(nodes)
        print(nodes.head())
        nodes['label'] = group
        nodes_l = np.array(nodes)
        print(nodes_l.shape)
        combined_nodes.extend(nodes_l)
        print(len(combined_nodes), len(combined_nodes[0]))
    mapping = {'ad': 2, 'hc': 0, 'mci': 1}
    print(len(combined_nodes), len(combined_nodes[0]))

    combined_edges, combined_nodes = pd.DataFrame(combined_edges), pd.DataFrame(combined_nodes, columns=range(len(combined_nodes[0])))
    combined_edges.iloc[:, -1] = combined_edges.iloc[:, -1].map(mapping)
    combined_nodes = combined_nodes.replace('m00', np.nan)
    combined_nodes.fillna(combined_nodes.mean(), inplace=True)

    return combined_edges, combined_nodes
#%%
combined_edges, combined_nodes = load_files()
#%%
fscores = fscore(combined_edges, class_col=len(combined_edges.columns)-1)
trans_edges = pd.DataFrame(scalar.fit_transform(combined_edges.iloc[:, :-1]))
trans_fscores = fscore(pd.concat([trans_edges, combined_edges.iloc[:,-1]]),
                class_col=len(combined_edges)-1)
scalar = StandardScaler()
# has the data already been standardized, if so in which way?
for log in [True, False]:
    plt.hist(sorted(fscores), label='before scaling', log=log, alpha=0.5)
    plt.hist(trans_fscores, label='after scaling to unit variance', log=log, alpha=0.5)
    plt.ylabel('Number of features')
    plt.xlabel('Fscores')
    plt.legend()
    if log == True:
        plt.savefig('outputs/fscores_log.png')
    else:
        plt.savefig('outputs/fscores.png')
    plt.show()
#%%
fscores = fscore(combined_edges, class_col=len(combined_edges.columns)-1)
trans_edges = pd.DataFrame(scalar.fit_transform(combined_edges.iloc[:, :-1]))
trans_fscores = fscore(pd.concat([trans_edges, combined_edges.iloc[:,-1]]),
                class_col=len(combined_edges)-1)
plt.hist(fscores[fscores!=0], label='before scaling')
plt.hist(trans_fscores[trans_fscores!=0], label='after scaling to unit variance')
plt.ylabel('Number of features')
plt.title('Without 0 fscores')
plt.xlabel('Fscores')
plt.legend()
plt.savefig('outputs/fscores_new_nonzero.png')
plt.show()
#%%
fig, ax = plt.subplots(5,1, figsize = (15,10))
skf = StratifiedKFold(n_splits=5)
i = 0
ax = ax.ravel()
for train_idx, test_idx in skf.split(combined_edges.iloc[:,:-1], combined_edges.iloc[:, -1]):
    if i < 5:
        X_train, y_train = combined_edges.iloc[train_idx, :-1], combined_edges.iloc[train_idx, -1]
        X_test, y_test = combined_edges.iloc[test_idx, :-1], combined_edges.iloc[test_idx, -1]
        ax[i] = make_hist(X_train, y_train, ax[i])
        input_graph = BrainGraph(edge='fscores', feature_type='FC', node_wts='given_value',
                           target='AD', max_num_nodes=10, val=0, thresh=0)
        indices = list(find_indices(range(int(52*53/2)), shape=52).keys()) # only the diagonal indices since we are considering the matrix to be symmetric
        X_y_uppert = pd.concat([X_train.iloc[:, indices], y_train], axis=1)
        input_graph.make_graph(fscore(X_y_uppert,class_col=X_y_uppert.columns[-1]),
                               strls_num=X_train.sum(axis=0).iloc[indices] > 0, thresh=0, avg_thresh=False, num_nodes=52)
        input_graph.set_node_labels(node_wts='given_value', wts=combined_nodes.iloc[train_idx, :-1].mean(axis=0)) # set the node weights as the mean for the training subjects
        input_graph.savefiles('/home/shreya/git/Thesis/gmwcs-solver')
        input_graph.run_solver('/home/shreya/git/Thesis/gmwcs-solver', max_num_nodes=10)
        i += 1
