import glob
import numpy as np
import pandas as pd

#checking in reginas folder if the subjects we have the labels for them are balanced or not
def get_subj_ids():
    input_dir = '/data/skapoor/HCP/results'
    present_subj = []
    #Now we need to check for which all subjects the meam_FA_connectome exists!
    #we get this 1M file containing most biologically important features
    for s in glob.glob(f'{input_dir}/*/T1w/Diffusion/'):
        if s.split('/HCP/results/')[1][0] in ['1', '2']:
            #print(s)
            subject = s.split('/HCP/results/')[1].split('/')[0]
            present_subj.append(subject)
    return(present_subj)
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
    out_diff = f'/data/regina/HCP/{subject}/T1w/Diffusion'
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
data = pd.read_csv('present_subjects.csv')
#%%
print(data['NEOFAC_A'].shape)
print(whole.shape)
#%%
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
norm = Normalizer()
#%%
from sklearn.model_selection import train_test_split

#%%
Y = data['NEOFAC_A']
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
#%%
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
X, y = make_classification(n_samples= len(data), n_features=len(data.columns),
                           n_informative=2, n_redundant=0,
                           random_state=0, shuffle=False)
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X_train, y_train)
#%%
y_pred = clf.predict(X_test)
acc = sum(y_pred==y_test)/len(y_test)
print(acc)