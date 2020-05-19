#%%
import pandas as pd
import os
import glob
import numpy as np
#%%
print(os.getcwd())
#%%
df = pd.read_csv('/home/skapoor/Thesis/Notes/HCP_data/unrestricted_mdkhatami_3_2_2017_5_48_20.csv')
input_dir = '/data/skapoor/HCP/results'
#%%
present_subj = []
for s in glob.glob(f'{input_dir}/*/T1w/Diffusion/mean_FA_connectome_5M.csv'):
    if s.split('/HCP/results/')[1][0] in ['1', '2']:
        #print(s)
        subject = s.split('/HCP/results/')[1].split('/')[0]
        #print(subject)
        if int(subject) in np.array(df['Subject']):
            #print(subject, ' present in file')
            #print(df.loc[df['Subject']==int(subject), :])
            present_subj.append(subject)
#%%
data = df.loc[df['Subject'].isin(present_subj), :] #reduced csv files containing data of only computed subj
#%%
'''
5 personality traits that we want to study are:
NEO-FFI Agreeableness (NEOFAC_A)
NEO-FFI Openness to Experience (NEOFAC_O)
NEO-FFI Conscientiousness (NEOFAC_C)
NEO-FFI Neuroticism (NEOFAC_N)
NEO-FFI Extraversion (NEOFAC_E)
'''
info = pd.read_excel('/home/skapoor/Thesis/Notes/HCP_data/HCP_S1200_DataDictionary_April_20_2018.xlsx', dtype ='str')
info.head()
#%%
labels = ['NEOFAC_A', 'NEOFAC_O', 'NEOFAC_C', 'NEOFAC_N', 'NEOFAC_E' ]
#%%
info = info.set_index('columnHeader')
#%%
info.loc[labels]['description']
#%%
# Extract these labels for our subjects!
big5 = data.loc[:, labels]
#%%
big5.describe()
#%%
import pandas_profiling
#%%
big5.profile_report()
#%%