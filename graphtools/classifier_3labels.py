import glob
import numpy as np
import pandas as pd


def get_subj_ids():
    input_dir = '/data/skapoor/HCP/results'
    present_subj = []
    #Now we need to check for which all subjects the meam_FA_connectome exists!
    #we get this 1M file containing most biologically important features
    for s in glob.glob(f'{input_dir}/*/T1w/Diffusion/mean_FA_connectome_1M_SIFT.csv'):
        if s.split('/HCP/results/')[1][0] in ['1', '2']:
            #print(s)
            subject = s.split('/HCP/results/')[1].split('/')[0]
            present_subj.append(subject)
    return(present_subj)
#%%

whole = np.zeros((len(get_subj_ids()),3,7056))
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
        whole[j,i,:] = edge_feature
        #print(i,j)
        i+=1
    j+=1
#%%
np.shape(whole[0])
#%%
# Let us take the labels i.e. the ones from unrestricted_files!
data = pd.read_csv('present_subjects.csv')
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
'''
5 personality traits that we want to study are:
NEO-FFI Agreeableness (NEOFAC_A)
NEO-FFI Openness to Experience (NEOFAC_O)
NEO-FFI Conscientiousness (NEOFAC_C)
NEO-FFI Neuroticism (NEOFAC_N)
NEO-FFI Extraversion (NEOFAC_E)
'''
#%%
# cut and put it into three categories, namely low medium and high
agreeable = np.array(pd.cut(data['NEOFAC_A'],3,  labels=False, retbins=True, right=False)[0])
open = np.array(pd.cut(data['NEOFAC_O'],3,  labels=False, retbins=True, right=False)[0])
consci = np.array(pd.cut(data['NEOFAC_C'],3,  labels=False, retbins=True, right=False)[0])
neuroti = np.array(pd.cut(data['NEOFAC_N'],3,  labels=False, retbins=True, right=False)[0])
extrav = np.array(pd.cut(data['NEOFAC_E'],3,  labels=False, retbins=True, right=False)[0])

#%%
