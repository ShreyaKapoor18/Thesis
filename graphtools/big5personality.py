#%%
import pandas as pd
import os
import glob
import numpy as np
from readfiles import computed_subjects
from readfiles import precomputed_subjects
import pandas_profiling

# %%
def report_cut(data, cut_type):
    # cut and put it into three categories, namely low medium and high
    if cut_type == 'cut':
        for label in labels:
            #print('before cut', data[label].head())
            data[label] = pd.Series(pd.cut(data[label], 3, labels=False, retbins=True, right=False)[0])
            #print('after cut', data[label].head())
        # how balanced are these cuts?
        big5 = pd.concat([data[label] for label in labels], axis=1)
        big5.columns = ['Agreeableness', 'Openness', 'Conscientiousness', 'Neuroticism',
                        'Extraversion']
        binned_traits = big5.profile_report()
        binned_traits.to_file('reports/big5-binned.html')
    if cut_type == 'qcut':
        for label in labels:
            data[label] = pd.Series(pd.qcut(data[label], 3, labels=False, retbins=True)[0])
        # how balanced are these cuts?
        big5 = pd.concat([data[label] for label in labels], axis=1)

        big5.columns = ['Agreeableness', 'Openness', 'Conscientiousness', 'Neuroticism',
                        'Extraversion']
        binned_traits = big5.profile_report()
        binned_traits.to_file('reports/big5-binned_qcut.html')
#%%
'''
5 personality traits that we want to study are:
NEO-FFI Agreeableness (NEOFAC_A)
NEO-FFI Openness to Experience (NEOFAC_O)
NEO-FFI Conscientiousness (NEOFAC_C)
NEO-FFI Neuroticism (NEOFAC_N)
NEO-FFI Extraversion (NEOFAC_E)
'''
print(os.getcwd())
data = computed_subjects()
data.to_csv('present_subjects.csv')
info = pd.read_excel('/home/skapoor/Thesis/Notes/HCP_data/HCP_S1200_DataDictionary_April_20_2018.xlsx', dtype ='str')
info.head()

labels = ['NEOFAC_A', 'NEOFAC_O', 'NEOFAC_C', 'NEOFAC_N', 'NEOFAC_E' ]

info = info.set_index('columnHeader')
info.loc[labels]['description']
# Extract these labels for our subjects!
big5 = data.loc[:, labels]
big5.describe()
profile = big5.profile_report()
profile.to_file('./report-big5.html')
#%%
data = precomputed_subjects() #reduced csv files containing target values from regina's folder
data.to_csv('present_subjects_regina.csv')
report_cut(data, 'cut')
#%%
data = precomputed_subjects()
report_cut(data, 'qcut')