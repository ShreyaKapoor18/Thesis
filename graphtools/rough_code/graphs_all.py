#%%
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import os.path
import numpy as np
#%%
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
subject = '128127'

df = pd.read_csv(os.path.join(parent_dir, 'preprocessing/result_files', subject,
                              'T1w/Diffusion/mean_FA_connectome_5M.csv'), sep = ' ')
#%%
print(df.shape)
df = np.array(df)
df =df[:,1:]
adj = np.array(df==1).astype(int)
#%%
G = nx.from_numpy_matrix(adj,create_using=nx.Graph) #we want an undirected graph
#nx.draw(G, pos = nx.spring_layout(G))
print(G.edges)
nodes=nx.draw_networkx_nodes(G,pos=nx.random_layout(G))
#nx.draw_networkx_edge_labels(G, pos=nx.spring_layout(G))

plt.show()
#%%
G = nx.from_numpy_matrix(df,create_using=nx.Graph) #we want an undirected graph
#nx.draw(G, pos = nx.spring_layout(G))
#print(G.edges)
nodes=nx.draw_networkx_nodes(G,pos=nx.spring_layout(G))
#nx.draw_networkx_edge_labels(G, pos=nx.spring_layout(G))
print(nx.number_of_nodes(G))
print(nx.average_node_connectivity(G))
plt.show()
#%%

G = nx.from_numpy_array(df>=0.52)
G.edges(data=True)
print(len(nx.nodes(G)))
nx.draw(G, pos= nx.fruchterman_reingold_layout(G))
plt.show()
#%%
np.ptp(df)
#%%
nx.draw(G, pos= nx.kamada_kawai_layout(G))
plt.show()
nx.draw(G, pos= nx.spectral_layout(G))
plt.show()
nx.draw(G, pos= nx.shell_layout(G))
plt.show()
#%%
plt.show()
#%%
df = pd.read_csv(os.path.join(parent_dir, 'preprocessing/result_files', subject,
                           'T1w/Diffusion/distances_mean_5M.csv'), sep=' ')
print(df.shape)
df = np.array(df)
df =df[:,1:]
adj = np.array(df==1).astype(int)

G = nx.from_numpy_array(df>150)
G.edges(data=True)
print(len(nx.nodes(G)))
nx.draw(G, pos= nx.fruchterman_reingold_layout(G))
plt.show()
#%%
np.ptp(df)
#%%
nx.draw(G, pos= nx.kamada_kawai_layout(G))
plt.show()
nx.draw(G, pos= nx.spectral_layout(G))
plt.show()
nx.draw(G, pos= nx.shell_layout(G))
plt.show()
