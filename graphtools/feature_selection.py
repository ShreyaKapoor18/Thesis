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
profile = big5.profile_report()
#%%
#profile = pandas_profiling.ProfileReport(big5)
profile.to_file('./report.html')
#%%
data.select_dtypes(include = ['int']).dtypes
#%%
whole_profile = data.select_dtypes(include = ['bool']).profile_report()
whole_profile.to_file('./report-bool.html')
#%%
personality_labels = info.loc[info['category']=='Personality', :].index
personality_labels
#%%
computed_subj = data.loc[:, personality_labels]
personality_profile = computed_subj.profile_report()
personality_profile.to_file('./report-personality.html')
#%%
neuro_labels = info.loc[info['category']=='Psychiatric and Life Function', :].index

#%%
#computed_subj = data.loc[:, neuro_labels] # These labels are not present in the file

#%%
print(data.columns.shape) # has less labels than the given info file
info.shape # has more info labels
#%%
len(set(data.columns).intersection(set(info.index)))
# so we currently have 373 common labels
#%%
freesurfer_labels = info.loc[info['category']=='FreeSurfer', :].index
computed_subj = data.loc[:, freesurfer_labels]
personality_profile = computed_subj.profile_report()
personality_profile.to_file('./report-FreeSurfer.html') #notpresent
#%%
for category in np.unique(info['category']):

    labels = info.loc[info['category']==category, :].index
    if set(labels).issubset(set(data.columns)):
        print(category, ' is present in the data unresctricted')
        present_subj = data.loc[:, labels]
        profile = present_subj.profile_report()
        profile.to_file(f'./report-{category}.html')
