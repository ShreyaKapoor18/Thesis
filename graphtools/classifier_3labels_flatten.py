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
        ax[j][i].hist(corr[-1][:-1], range=(0,0.4), log = True)
        ax[j][i].set_title(big5[j]+' '+ edge_names[i])
        ax[j][i].set_ylabel('Num edges')
        ax[j][i].set_xlabel('Correlation coeff')
plt.show()