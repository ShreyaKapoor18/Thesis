#%%
import pandas as pd
import os
import glob
import numpy as np
from readfiles import computed_subjects
from readfiles import precomputed_subjects
#%%
print(os.getcwd())
#%%
data = computed_subjects()
data.to_csv('present_subjects.csv')
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
info.loc[labels]['description']
# Extract these labels for our subjects!
big5 = data.loc[:, labels]
big5.describe()
import pandas_profiling
profile = big5.profile_report()
#%%
#profile = pandas_profiling.ProfileReport(big5)
profile.to_file('./report-big5.html')
#%%
data = precomputed_subjects() #reduced csv files containing target values from regina's folder
data.to_csv('present_subjects_regina.csv')
# cut and put it into three categories, namely low medium and high
agreeable = pd.Series(pd.cut(data['NEOFAC_A'],3,  labels=False, retbins=True, right=False)[0])
open =pd.Series(pd.cut(data['NEOFAC_O'],3,  labels=False, retbins=True, right=False)[0])
consci = pd.Series(pd.cut(data['NEOFAC_C'],3,  labels=False, retbins=True, right=False)[0])
neuroti = pd.Series(pd.cut(data['NEOFAC_N'],3,  labels=False, retbins=True, right=False)[0])
extrav = pd.Series(pd.cut(data['NEOFAC_E'],3,  labels=False, retbins=True, right=False)[0])
#%%
#how balanced are these cuts?
big5 = pd.concat([agreeable, open, consci, neuroti, extrav], axis=1)
big5.columns = ['Agreeableness', 'Openness', 'Conscientiousness', 'Neuroticism',
                            'Extraversion']
binned_traits = big5.profile_report()
binned_traits.to_file('./big5-binned.html')
