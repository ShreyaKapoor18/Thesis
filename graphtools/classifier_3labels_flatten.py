import glob
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from readfiles import get_subj_ids
from readfiles import computed_subjects
import matplotlib.pyplot as plt
#%%
def f_score(data, feature, label):
    f1 = data.loc[data.loc[:,label]==0,feature]
    f2 = data.loc[data.loc[:,label]==1,feature]
    n1 = len(f1)
    n2 = len(f2)
    x1 = f1.mean()
    x2 = f2.mean()
    xw = data.loc[:,feature].mean()
    num = (x1-xw)**2+(x2-xw)**2
    s1 = 0
    for i in range(n1):
        s1+= (f1.iloc[i] - x1)**2
    s1=s1/(n1-1)
    s2 = 0
    for i in range(n2):
        s2+= (f2.iloc[i] - x2)**2
    s2=s2/(n2-1)
    deno = s1+s2
    if deno!=0:
        print(num, deno)
        return num/deno
    else:
        return 0

#%%
def fscore(data, class_col='class'):
    """ Compute the F-score for all columns in a DataFrame
    """
    grouped = data.groupby(by=class_col)
    means = data.mean()
    g_means = grouped.mean()
    g_vars = grouped.var()

    numerator = np.sum((g_means - means) ** 2, axis=0)
    denominator = np.sum(g_vars, axis=0)
    return numerator / denominator

#%%
whole = np.zeros((len(get_subj_ids()),7056*3))
j =0
'''
There are three features that we want to add to the files
1. Mean FA between the two nodes
2. The mean length of the streamlines between the two nodes
3. The number of streamlines between the two nodes
'''
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
#%%
np.shape(whole[0])
#%%
# Let us take the labels i.e. the ones from unrestricted_files!
data = computed_subjects()
data.reset_index(inplace= True)
#%%
labels = ['NEOFAC_A', 'NEOFAC_O', 'NEOFAC_C', 'NEOFAC_N', 'NEOFAC_E' ]
edge_names = ['mean_FA', 'mean strl', 'num streamlines']
big5 = ['Agreeableness', 'Openness', 'Conscientiousness', 'Neuroticism',
                        'Extraversion']
#%%
print(data['NEOFAC_O'].shape)
print(whole.shape)
#%%
fig, ax = plt.subplots(5,3, figsize =(10,10))
for j in range(len(labels)):
    label = np.array(data[labels[j]]).reshape(-1,1)
    # correlation of mean FA edges, mean str length, number of strl with Openness
    for i in range(3):
        map_o = np.concatenate((whole[:,0*i:(i+1)*7056], label),axis =1)
        corr = np.cov(map_o, rowvar=False)
        ax[j][i].hist(corr[-1][:-1], range=(0.0,0.4), log = True, bins=100)
        ax[j][i].set_title(big5[j]+' '+ edge_names[i])
        ax[j][i].set_ylabel('Num edges')
        ax[j][i].set_xlabel('Correlation coeff')
plt.show()
#%%
data_edges = pd.DataFrame(whole)
data_edges.reset_index(inplace=True)
#%%
fig, ax = plt.subplots(5,1, figsize = (15,15))
i = 0
for label in labels:
    bin_label = data[label] >= data[label].median()
    bin_label=  bin_label.astype(int)
    bin_label.reset_index(drop=True, inplace=True)
    #print(bin_label.head())
    #print('x1', data[label])
    assert len(data_edges)== len(data[label])
    '''
    df = pd.concat([data_edges, data[label]], axis=1, ignore_index=True)
    l1 = list(range(len(data_edges.columns)))
    l1.append(label)
    df.columns = l1
    '''
    df = pd.concat([data_edges, bin_label], axis = 1)
    l1 = list(range(len(data_edges.columns)))
    l1.append(label)
    df.columns = l1
    #print(df.head())
    #print(fscore(df, label))
    ax[i].hist(np.log1p(fscore(df,label)[:-1]), log = True, bins=100)
    ax[i].set_title(big5[i])
    ax[i].set_xlabel('F-Score')
    ax[i].set_title('Number of edges')

    i+=1
plt.show()
    #print(f_score(df, 200, label))
#%%
data_edges.isna().any()
#%%
'''arr = {}
for label in labels:
    arr[label] = {'mean_FA':[] ,'mean strl': [], 'num strl': []}
    for i in range(3):
        key = list(arr[label].keys())[i]
        for edge in range(i,(i+1)*7056):
            if edge not in labels:
                #print (edge, f_score(data_edges, edge, 'NEOFAC_A'))
                arr[label][key].append(f_score(data_edges,edge, label))'''
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