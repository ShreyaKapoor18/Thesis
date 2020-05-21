import networkx as nx
import pandas as pd
import numpy as np
import glob
#%%
subject = '128127'
out_diff = f'/data/skapoor/HCP/results/{subject}/T1w/Diffusion'
G = nx.MultiGraph()
G.add_nodes_from(range(84))
edge_list = []
for i in range(84):
    row_list = []
    for j in range(84):
        row_list.append([i, j , {}])
    edge_list.append(row_list)
#%%
for file in glob.glob(f'{out_diff}/*.csv'):
    if 'mean_FA_per_streamline' not in file:
        # print(file) # we need to make all edges as a feature for each subject!
        # so we will have 84x84 features
        filename = file.split('Diffusion/')[1].strip('.csv')
        print(filename)
        df = pd.read_csv(file, sep = ' ', header=None)
        adj = df>0
        for i in range(84):
            for j in range(84):
                edge_list[i][j][2][f'{filename}'] = df.iloc[i,j]
#%%
edge_arr = np.array(edge_list).reshape(7056,3)
G.add_edges_from(edge_arr)
#%%
G[0][0][0]
#%%
G.size()
G.degree()
#%%
# No thresholding done as yet!
