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

whole = np.zeros((len(get_subj_ids()),7056))
j =0
for subject in get_subj_ids():
    out_diff = f'/data/skapoor/HCP/results/{subject}/T1w/Diffusion'

    file = f'{out_diff}/mean_FA_connectome_1M_SIFT.csv'
    if 'mean_FA_per_streamline' not in file:
        #print(file) # we need to make all edges as a feature for each subject!
        # so we will have 84x84 features
        all_5 = np.array(pd.read_csv(file, sep = ' ', header= None))

    flatten = np.reshape(all_5, (7056,))
    whole[j,:] = flatten
#%%
np.shape(whole[0])
#%%
# Let us take the labels i.e. the ones from unrestricted_files!
data = pd.read_csv('present_subjects.csv')
#%%
print(data['Gender'].shape)
print(whole.shape)
#%%
from sklearn.model_selection import train_test_split
import numpy as np
#%%

Y = np.array(data['Gender'])
X = whole
X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.80)
#%%
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(X_train, y_train)
#%%
y_pred = clf.predict(X_test)
#%%
acc = sum(y_pred==y_test)/len(y_test)
#%%
print(acc)
y_pred
#%%
sum(data['Gender']=='M')/68