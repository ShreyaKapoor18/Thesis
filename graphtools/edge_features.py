import numpy as np
import pandas as pd

def get_subj_ids():
    input_dir = '/data/skapoor/HCP/results'
    present_subj = []
    #Now we need to check for which all subjects the meam_FA_connectome exists!
    for s in glob.glob(f'{input_dir}/*/T1w/Diffusion/mean_FA_connectome_5M.csv'):
        if s.split('/HCP/results/')[1][0] in ['1', '2']:
            #print(s)
            subject = s.split('/HCP/results/')[1].split('/')[0]
            present_subj.append(subject)
    return(present_subj)
#%%

whole = np.zeros((len(get_subj_ids()),7056,5))
j = 0
for subject in get_subj_ids():
    out_diff = f'/data/skapoor/HCP/results/{subject}/T1w/Diffusion'
    all_5 = np.zeros((84, 84, 5))
    i = 0
    for file in glob.glob(f'{out_diff}/*.csv'):
        if 'mean_FA_per_streamline' not in file:
            #print(file) # we need to make all edges as a feature for each subject!
            # so we will have 84x84 features
            all_5[:,:, i] = np.array(pd.read_csv(file, sep = ' ', header= None))
            i+=1
    flatten = np.reshape(all_5, (7056,5))
    whole[j,:,:] = flatten
#%%
np.shape(whole[0])
#%%
#there are 7056 edges - features in each edge:5
# total subjects - 68
#%%
len(whole)
#%%
a = []
for i in range(len(whole)):
    all_cols = []
    for j in range(len(whole[0])):
        tup = tuple(whole[i,j,:])
        all_cols.append(tup)
    a.append(all_cols)
#%%
np.shape(a)
#%%
features = pd.DataFrame(a)
#%%
np.shape(whole)
#%%
# try getting only one feature for now and training classifier!