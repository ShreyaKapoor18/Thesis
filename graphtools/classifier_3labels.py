import glob
import numpy as np
import pandas as pd
from readfiles import computed_subjects
from readfiles import get_subj_ids
import matplotlib.pyplot as plt
#%%

'''
There are three features that we want to add to the feature matrix
1. Mean FA between the two nodes
2. The mean length of the streamlines between the two nodes
3. The number of streamlines between the two nodes
'''
whole = np.zeros((len(get_subj_ids()),3,7056))
j =0
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
        whole[j,i,:] = edge_feature
        #print(i,j)
        i+=1
    j+=1

#%%
# Let us take the labels i.e. the ones from unrestricted_files!
data = computed_subjects()
#%%
print(data['NEOFAC_A'].shape)
print(whole.shape)
#%%
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
norm = Normalizer()
#%%
correlations = np.zeros((7056, 3,3))
# pearson correlation per edge, lets try for one edge first
for j in range(7056):
    scaler =StandardScaler()
    edge = pd.DataFrame(whole[:,:,j], columns=['mean FA', 'mean strl length', 'num strl'])
    #print(edge.head())
    #print(edge.corr())
    edge = pd.DataFrame(norm.fit_transform(edge), columns=['mean FA', 'mean strl length', 'num strl'])
    #print(edge.head())
    correlations[j,:,:] = edge.corr()
#%%
#score histogram per type? What is meant by that from what I understood hisogram of FA in one edge across all subjects
fig, ax = plt.subplots(10,10, figsize=(20,20))
for i in range(10):
    for j in range(10):
        ax[i][j].hist(whole[:,0, i*10+j]) # across all subjects edge number index 3 with mean_FA
plt.title('mean FA along 100 edges for all subjects')
plt.show()
#%%
mean_FA_hist = { num_edge: np.histogram(whole[:,0,num_edge]) for num_edge in range(7056)}
mean_strl_hist = { num_edge: np.histogram(whole[:,1,num_edge]) for num_edge in range(7056)}
num_strl_hist = { num_edge: np.histogram(whole[:,2,num_edge]) for num_edge in range(7056)}