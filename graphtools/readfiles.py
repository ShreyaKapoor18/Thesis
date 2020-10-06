import pandas as pd
import numpy as np
import glob
import re
import os
import pandas_profiling

def computed_subjects():
    """
    :param notes_path: the notes directory
    :param input_dir: the input directory containing the results that have been pre-processed
    :param edge_file: the connectome files that we have already computed
    :return data: the data of only the subjects whose data has been pre-processed
    """
    notes_path = '/home/skapoor/Thesis/Notes/HCP_data'
    input_dir = '/data/skapoor/HCP/results'
    df = pd.read_csv(f'{notes_path}/unrestricted_mdkhatami_3_2_2017_5_48_20.csv')
    present_subj = []
    #Now we need to check for which all subjects the meam_FA_connectome exists!
    for s in sorted(glob.glob(f'{input_dir}/*/T1w/Diffusion/connectivity_matrices_whole.png')):
        if s.split('/HCP/results/')[1][0] in ['1', '2']:
            #print(s)
            subject = s.split('/HCP/results/')[1].split('/')[0]
            #print(subject)
            if int(subject) in np.array(df['Subject']):
                present_subj.append(subject)
    data = df.loc[df['Subject'].isin(present_subj), :] # reduced csv files containing data of only computed subj
    data.reset_index(drop=True, inplace=True) # so that the index goes from 0 to 140
    data.set_index(data['Subject'], inplace=True)

    return data

def test_subjects():
    notes_path = '/home/skapoor/Thesis/Notes/HCP_data'
    input_dir = '/data/skapoor/test_data'
    df = pd.read_csv(f'{notes_path}/unrestricted_mdkhatami_3_2_2017_5_48_20.csv')
    present_subj = []
    # Now we need to check for which all subjects the meam_FA_connectome exists!
    for s in sorted(glob.glob(f'{input_dir}/*/T1w/Diffusion/mean_FA_connectome_1M_SIFT.csv')):
        #print(s)
        subject = s.split('/')[-4]
        #print(subject)
        if int(subject) in np.array(df['Subject']):
            present_subj.append(subject)
    data = df.loc[df['Subject'].isin(present_subj), :] # reduced csv files containing data of only computed subj
    data.reset_index(drop=True, inplace=True) # so that the index goes from 0 to 140
    data.set_index(data['Subject'], inplace=True)

    return data



def make_profiles(filename, notes_path):
    """
    Make html files for the profiles of all label. Filename is either the one from
    my folder or the one from Regina's folder. present_subjects.csv or present_subjects_regina.csv
    :param filename the name of the csv file you want to check subject data
    :return:
    """
    info = pd.read_excel(f'{notes_path}/HCP_S1200_DataDictionary_April_20_2018.xlsx', dtype='str')
    data =pd.read_csv(filename)
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
    for s in sorted(glob.glob(f'{input_dir}/*/T1w/Diffusion/connectivity_matrices_whole.png')):
        if s.split('/HCP/results/')[1][0] in ['1', '2']:
            #print(s)
            subject = s.split('/HCP/results/')[1].split('/')[0]
            present_subj.append(subject)
    return present_subj


def corresp_label_file(file):
    f = open('/home/skapoor/Thesis/Notes/HCP_data/labels/' + file)
    dat = f.read()
    if file == 'FreeSurferColorLUT.txt':
        dat = [x.split(' ')[::2] for x in dat.split('\n')[1:-1]]
        dat = [x for x in dat if len(x) >= 3 and x[0] != '#']
        dict_data = {int(x[0]): x[1] for x in dat}
        """for x in np.unique(labels).astype(int):
            for y in dat: 
                if y[0]!=[''] and y[0]!='#':
                    #print(str(x), y)
                    if str(x) == y[0]: 
                        for a in y: 
                            if 'white-matter' in a.lower():
                                print (y)
                            if 'corpus' in a.lower(): 
                                print (y)
                            if 'cc' in a.lower(): 
                                print (y)"""
    elif file == 'fs_default.txt':
        dat = [re.sub(' +', ' ', x).split(' ') for x in dat.split('\n')[2:-2]]
        dat = [x for x in dat if len(x) >= 3 and int(x[0])]
        dict_data = {int(x[0]): x[2] for x in dat}

    return dict_data

def clean_dirs(mews):  # make a separate directory for each label, easier to do comparisons

    if os.path.exists(f'{mews}/outputs/nodes'):
        os.system(f'rm {mews}/outputs/nodes/*')
    if os.path.exists(f'{mews}/outputs/edges'):
        os.system(f'rm {mews}/outputs/edges/*')
    if os.path.exists(f'{mews}/outputs/solver'):
        os.system(f'rm {mews}/outputs/solver/*')

