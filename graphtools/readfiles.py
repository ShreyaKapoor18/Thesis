import pandas as pd
import numpy as np
import glob
import pandas_profiling

#notes_path = '/home/skapoor/Thesis/Notes/HCP_data'
#input_dir = '/data/skapoor/HCP/results'
def computed_subjects():
    '''

    :param notes_path: the notes directory
    :param input_dir: the input directory containing the results that have been pre-processed
    :param edge_file: the connectome files that we have already computed
    :return data: the data of only the subjects whose data has been pre-processed
    '''
    notes_path = '/home/skapoor/Thesis/Notes/HCP_data'
    input_dir = '/data/skapoor/HCP/results'
    df = pd.read_csv(f'{notes_path}/unrestricted_mdkhatami_3_2_2017_5_48_20.csv')
    present_subj = []
    #Now we need to check for which all subjects the meam_FA_connectome exists!
    for s in glob.glob(f'{input_dir}/*/T1w/Diffusion/mean_FA_connectome_5M.csv'):
        if s.split('/HCP/results/')[1][0] in ['1', '2']:
            #print(s)
            subject = s.split('/HCP/results/')[1].split('/')[0]
            #print(subject)
            if int(subject) in np.array(df['Subject']):
                present_subj.append(subject)
    data = df.loc[df['Subject'].isin(present_subj), :]  # reduced csv files containing data of only computed subj
    return data

def precomputed_subjects():
    '''

    :param notes_path: the notes directory
    :param input_dir: the input directory containing the results that have been pre-processed
    :param edge_file: the connectome files that we have already computed
    :return data: the data of only the subjects whose data has been pre-processed
    '''
    notes_path = '/home/skapoor/Thesis/Notes/HCP_data'
    input_dir = '/data/regina/HCP'
    df = pd.read_csv(f'{notes_path}/unrestricted_mdkhatami_3_2_2017_5_48_20.csv')
    present_subj = []
    #Now we need to check for which all subjects the meam_FA_connectome exists!
    for s in glob.glob(f'{input_dir}/*/T1w/Diffusion/'):
        if s.split('/HCP/')[1][0] in ['1', '2']:
            #print(s)
            subject = s.split('/HCP/')[1].split('/')[0]
            #print(subject)
            if int(subject) in np.array(df['Subject']):
                present_subj.append(subject)
    data = df.loc[df['Subject'].isin(present_subj), :]  # reduced csv files containing data of only computed subj
    return data


def make_profiles():
    '''
    Make html files for the profiles of all labels
    :return:
    '''
    info = pd.read_excel(f'{notes_path}/HCP_S1200_DataDictionary_April_20_2018.xlsx', dtype='str')
    for category in np.unique(info['category']):
        labels = info.loc[info['category'] == category, :].index
        if set(labels).issubset(set(data.columns)):
            print(category, ' is present in the data unresctricted')
            present_subj = data.loc[:, labels]
            profile = present_subj.profile_report()
            profile.to_file(f'./report-{category}.html')

def get_subj_ids():
    input_dir = '/data/skapoor/HCP/results'
    present_subj = []
    #Now we need to check for which all subjects the meam_FA_connectome exists!
    for s in glob.glob(f'{input_dir}/*/T1w/Diffusion/mean_FA_connectome_5M.csv'):
        if s.split('/HCP/results/')[1][0] in ['1', '2']:
            #print(s)
            subject = s.split('/HCP/results/')[1].split('/')[0]
            present_subj.append(suject)
    return(present_subj)