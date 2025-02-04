import numpy as np
import matplotlib.pyplot as plt
from metrics import fscore
from readfiles import get_subj_ids
import pandas as pd
import glob
import os


# %%
def generate_combined_matrix(tri, present_subjects):
    """
    There are three features that we want to add to the matrix for all subjects
    1. Mean FA between the two nodes
    2. The mean length of the streamlines between the two nodes
    3. The number of streamlines between the two nodes
    """
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
    # scaling will be done according to the training and test data
    whole = pd.DataFrame(whole)
    whole.index = present_subjects
    return whole

def generate_training_data(tri, present_subjects):
    """
    There are three features that we want to add to the matrix for all subjects
    1. Mean FA between the two nodes
    2. The mean length of the streamlines between the two nodes
    3. The number of streamlines between the two nodes
    """
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
    # scaling will be done according to the training and test data
    whole = pd.DataFrame(whole)
    whole.index = present_subjects
    return whole

def generate_test_data(tri, index):
    location = '/data/skapoor/test_data/*/T1w/Diffusion/mean_FA_connectome_1M_SIFT.csv'
    test_data = np.zeros((len(index), tri * 3))
    subjects = []
    j = 0
    for subject in sorted(glob.glob(location)):
        s = subject.split('/')[-4]
        if s[-1] != 'h':
            if int(s) in index:
                files = [f'/data/skapoor/test_data/{s}/T1w/Diffusion/mean_FA_connectome_1M_SIFT.csv',
                         f'/data/skapoor/test_data/{s}/T1w/Diffusion/distances_mean_1M_SIFT.csv',
                         f'/data/skapoor/test_data/{s}/T1w/Diffusion/connectome_1M.csv']
                subjects.append(int(s))
                i = 0
                for file in files:
                    if os.path.exists(file):
                        # print(file) # we need to make all edges as a feature for each subject!
                        # so we will have 84x84 features
                        edge_feature = np.array(pd.read_csv(file, sep=' ', header=None))
                        # the file shall be number of subjects x 7056
                        # print(file)
                        edge_feature = edge_feature[np.triu_indices(84)]  # get only the upper triangular indices
                        test_data[j, i * tri:(i + 1) * tri] = edge_feature
                        # print(i,j)
                        i += 1
                j += 1
    # scaling will be done according to the training and test data
    test_data = pd.DataFrame(test_data, index=subjects)
    assert list(test_data.index) == list(index)
    return test_data


def hist_correlation(data, whole, labels, edge_names, big5, tri):
    corr = np.zeros((5, 3, tri))
    fig, ax = plt.subplots(5, 3, figsize=(10, 10))
    for j in range(len(labels)):
        label = data[labels[j]]
        # check if the variance of each feature is not zero if it is then remove it
        for i in range(3):
            map_o = pd.concat((pd.DataFrame(whole.iloc[:, i * tri:(i + 1) * tri]), label), axis=1)
            # corr = np.corrcoef(map_o, rowvar=False)
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
    # plt.show()
    return corr


def hist_fscore(data, whole, labels, big5, edge_names, tri):
    # to return the fscore in order to get the best performing features according to fscore
    fscores = np.zeros((5, 3, tri))  # 5 is for the big5 personality traits, 3 is for the three different edge names
    fig, ax = plt.subplots(5, 3, figsize=(15, 15))
    for j in range(len(labels)):
        # threshold for converting the data to binary format for classification
        bin_label = data[labels[j]] >= data[labels[j]].median()  # this ensures that the labels are balanced?
        bin_label = bin_label.astype(int)  # fscore here is based on the binary labelling using median value!
        bin_label.reset_index(drop=True, inplace=True)
        for i in range(len(edge_names)):
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
    return fscores


def feature_matrix(subject_id, feature):
    diffusion_path = f'/data/skapoor/HCP/results/{subject_id}/T1w/Diffusion'
    if feature == 'mean_FA':
        csv = f'{diffusion_path}/mean_FA_connectome_1M_SIFT.csv'
    elif feature == 'mean_strl_len':
        csv = f'{diffusion_path}/distances_mean_1M_SIFT.csv'
    elif feature == 'num_strls':
        csv = f'{diffusion_path}/connectome_1M.csv'
    df = pd.read_csv(csv, sep=' ', header=None)
    return df

def print_graphs_info(input_graph, output_graph):
    print('The number of nodes in the Input graph', len(input_graph.nodes))
    print('The number of edges in the Input graph', len(input_graph.edges))
    print('The number of nodes in the output graph', len(output_graph.nodes))
    print('The number of edges in the output graph', len(output_graph.edges))