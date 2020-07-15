import numpy as np
import matplotlib.pyplot as plt
from metrics import fscore
from readfiles import get_subj_ids
import pandas as pd
from sklearn.preprocessing import StandardScaler
# %%
def generate_combined_matrix(tri, present_subjects):
    """
    There are three features that we want to add to the matrix for all subjects
    1. Mean FA between the two nodes
    2. The mean length of the streamlines between the two nodes
    3. The number of streamlines between the two nodes
    """
    #norm = Normalizer()
    scale = StandardScaler()
    whole = np.zeros((len(get_subj_ids()), tri * 3))
    # but the matrix is upper triangular so we should only take that into account, then number of features will
    # get reduced
    # assert get_subj_ids() == present_subjects
    # make sure labels are ordered the same way in which we read data
    j = 0
    for subject in present_subjects:

        out_diff = f'/data/skapoor/HCP/results/{subject}/T1w/Diffusion'
        files = [f'{out_diff}/mean_FA_connectome_1M_SIFT.csv', f'{out_diff}/distances_mean_1M_SIFT.csv',
                 f'{out_diff}/connectome_1M.csv']
        i = 0
        for file in files:
            # print(file) # we need to make all edges as a feature for each subject!
            # so we will have 84x84 features
            edge_feature = np.array(pd.read_csv(file, sep=' ', header=None))
            # the file shall be number of subjects x 7056
            edge_feature = edge_feature[np.triu_indices(84)]  # get only the upper triangular indices
            whole[j, i * tri:(i + 1) * tri] = edge_feature
            # print(i,j)
            i += 1
        j += 1
    """
    We need to normalise the data since the scales are different 
    and we still want to retain the variance
    """
    #whole = norm.fit_transform(whole) # seems wrong since one subject gets scaled to a unit norm
    whole = scale.fit_transform(whole)
    whole = pd.DataFrame(whole)
    whole.index = present_subjects
    return whole


def hist_correlation(data, whole, labels, edge_names, big5, tri):
    corr = np.zeros((5, 3, tri))
    fig, ax = plt.subplots(5, 3, figsize=(10, 10))
    for j in range(len(labels)):
        label = data[labels[j]]
        # check if the variance of each feature is not zero if it is then remove it
        # correlation of mean FA edges, mean str length, number of strl with Openness
        for i in range(3):
            map_o = pd.concat((pd.DataFrame(whole.iloc[:, i * tri:(i + 1) * tri]), label), axis=1)
            #corr = np.corrcoef(map_o, rowvar=False)
            corr[j, i, :] = np.array(map_o.drop(labels[j], axis=1).apply(lambda x: x.corr(map_o[labels[j]])))
            if np.isnan(corr[j][i]).any():
                # print(j,i)
                corr[j][i][np.isnan(corr[j][i])] = 0
            ax[j][i].hist(corr[j, i, :], log=True, bins=100)
            ax[j][i].set_title(big5[j] + ' ' + edge_names[i])
            ax[j][i].set_ylabel('Num edges')
            ax[j][i].set_xlabel('Correlation coeff')
    plt.tight_layout()
    plt.savefig('reports/correlation_distribution.png')
    #plt.show()
    return corr


def hist_fscore(data, whole, labels, big5, edge_names, tri):
    # to return the fscore in order to get the best performing features according to fscore
    fscores = np.zeros((5, 3, tri))
    fig, ax = plt.subplots(5, 3, figsize=(15, 15))
    for j in range(len(labels)):
        # threshold for converting the data to binary format for classification
        bin_label = data[labels[j]] >= data[labels[j]].median()  # this ensures that the labels are balanced?
        bin_label = bin_label.astype(int)
        bin_label.reset_index(drop=True, inplace=True)
        for i in range(3):
            data_edges = pd.DataFrame(whole.iloc[:, i * tri:(i + 1) * tri])
            data_edges.reset_index(inplace=True, drop=True)
            # print(bin_label.head())
            # print('x1', data[label])
            assert len(data_edges) == len(data[labels[j]])

            df = pd.concat([data_edges, bin_label], axis=1)
            l1 = list(range(len(data_edges.columns)))
            l1.append(labels[j])
            df.columns = l1


            fscores[j][i] = fscore(df, labels[j])[:-1]
            if np.isnan(fscores[j][i]).any():
                # print(j,i)
                fscores[j][i][np.isnan(fscores[j][i])] = 0
            # to do resolve nan values!
            assert np.isnan(fscores[j][i]).any() == False
            ax[j][i].hist(np.log1p(fscores[j][i]), log=True, bins=100)
            ax[j][i].set_title(big5[j] + ' ' + edge_names[i])
            ax[j][i].set_xlabel('F-Score')
            ax[j][i].set_ylabel('Number of edges')
    plt.tight_layout()
    plt.savefig('reports/fscore_distribution.png')
    #plt.show()
    return fscores
