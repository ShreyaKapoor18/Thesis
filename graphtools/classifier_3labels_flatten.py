import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from readfiles import get_subj_ids
from readfiles import computed_subjects
import matplotlib.pyplot as plt
from metrics import fscore
#%%
def generate_combined_matrix():
    '''
    There are three features that we want to add to the matrix for all subjects
    1. Mean FA between the two nodes
    2. The mean length of the streamlines between the two nodes
    3. The number of streamlines between the two nodes
    '''
    norm = Normalizer()
    whole = np.zeros((len(get_subj_ids()), 7056 * 3))
    j = 0
    for subject in get_subj_ids():
        out_diff = f'/data/skapoor/HCP/results/{subject}/T1w/Diffusion'
        files = [f'{out_diff}/mean_FA_connectome_1M_SIFT.csv', f'{out_diff}/distances_mean_1M_SIFT.csv',
                 f'{out_diff}/connectome_1M.csv']
        i = 0
        for file in files:
            #print(file) # we need to make all edges as a feature for each subject!
            # so we will have 84x84 features
            edge_feature = np.array(pd.read_csv(file, sep =' ', header= None))
            # the file shall be number of subjects x 7056
            edge_feature = np.reshape(edge_feature, (7056,))
            whole[j,i*7056:(i+1)*7056] = edge_feature
            #print(i,j)
            i+=1
        j+=1
    '''we need to normalise the data since the scales are different 
    and we still want to retain the variance
    '''
    whole = norm.fit_transform(whole)
    return whole
#%%
def hist_correlation(data, whole, labels, edge_names, big5):
    fig, ax = plt.subplots(5,3, figsize =(10,10))
    for j in range(len(labels)):
        label = np.array(data[labels[j]]).reshape(-1,1)
        # correlation of mean FA edges, mean str length, number of strl with Openness
        for i in range(3):
            map_o = np.concatenate((whole[:, i*7056:(i+1)*7056], label),axis =1)
            corr = np.cov(map_o, rowvar=False)
            ax[j][i].hist(corr[-1][:-1], log = True, bins=100)
            ax[j][i].set_title(big5[j]+' '+ edge_names[i])
            ax[j][i].set_ylabel('Num edges')
            ax[j][i].set_xlabel('Correlation coeff')
    plt.savefig('reports/correlation_distribution.png')
    plt.show()

def hist_fscore(whole, labels, big5, edge_names):
    #to return the fscore in order to get the best performing features according to fscore
    fscores = np.zeros((5,3,7056))
    fig, ax = plt.subplots(5,3, figsize = (15,15))
    for j in range(len(labels)):
        # thresholding for converting the data to binary format for classification
        bin_label = data[labels[j]] >= data[labels[j]].median()
        bin_label = bin_label.astype(int)
        bin_label.reset_index(drop=True, inplace=True)
        for i in range(3):
                data_edges = pd.DataFrame(whole[:, i * 7056:(i + 1) * 7056])
                data_edges.reset_index(inplace=True, drop= True)
                #print(bin_label.head())
                #print('x1', data[label])
                assert len(data_edges)== len(data[labels[j]])

                df = pd.concat([data_edges, bin_label], axis = 1)
                l1 = list(range(len(data_edges.columns)))
                l1.append(labels[j])
                df.columns = l1
                #print(df.head())
                #print(fscore(df, label))
                fscores[j][i] = fscore(df, labels[j])[:-1]
                ax[j][i].hist(np.log1p(fscores[j][i]), log = True, bins=100)
                ax[j][i].set_title(big5[j] + ' '+edge_names[i])
                ax[j][i].set_xlabel('F-Score')
                ax[j][i].set_ylabel('Number of edges')
    plt.savefig('reports/fscore_distribution.png')
    plt.show()
    return fscores

#%%
data = computed_subjects() # labels for the computed subjects
data.reset_index(inplace= True)
whole = generate_combined_matrix()
# Let us take the labels i.e. the ones from unrestricted_files!

#%%
labels = ['NEOFAC_A', 'NEOFAC_O', 'NEOFAC_C', 'NEOFAC_N', 'NEOFAC_E' ]
edge_names = ['mean_FA', 'mean strl', 'num streamlines']
big5 = ['Agreeableness', 'Openness', 'Conscientiousness', 'Neuroticism',
                        'Extraversion']
#%%
fscores = hist_fscore(whole, labels,big5, edge_names)
hist_correlation(data, whole, labels, edge_names, big5)
#%%
fscores
#%%
for label in labels:
    Y = np.array(data[label] >= data[label].median()).astype(int)
    X = whole
    X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.80)
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = sum(y_pred==y_test)/len(y_test)
    print(acc,'SVM', label)
    y_predX, y = make_classification(n_samples= len(data), n_features=len(data.columns),
                           n_informative=2, n_redundant=0,
                           random_state=0, shuffle=False)
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(X_train, y_train)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = sum(y_pred==y_test)/len(y_test)
    print(acc, 'RF', label)