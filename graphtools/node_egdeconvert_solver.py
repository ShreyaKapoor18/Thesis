#%%
import pandas as pd
import networkx as nx
import os.path
import numpy as np
from random import randint
#%%
def make_nodes_edges(filename, subject):
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    df = pd.read_csv(os.path.join(parent_dir, 'preprocessing/result_files', subject,
                                  f'T1w/Diffusion/{filename}.csv'), sep = ' ', header = None)
    #print(df.shape)
    df = np.array(df)
    #adj = np.array(df==1).astype(int)
    G = nx.from_numpy_array(df)
    #write these in the format given for the mews file!
    nodes_file = open(f'{filename}_nodes', 'w+')
    edges_file = open(f'{filename}_edges', 'w+')
    nodes = []
    count = 0
    for node in nx.nodes(G):
        #print(node)
        if G.degree(node) >=2:
            print(str(node) + ' ' + str(1), file=nodes_file)
            #print(str(node) + ' ' + str(0), file=nodes_file)
            nodes.append(node)
            count +=1
            #print(node, 'has degree >=2')
    print('Number of nodes having degree>=2', count)
    #print(len(nodes))

    for edge in nx.edges(G):
        if edge[0] in nodes and edge[1] in nodes:
             print(str(edge[0]) + ' '+ str(edge[1])+ ' '+ str(G.get_edge_data(edge[0], edge[1])['weight']),
                  file=edges_file)
             #print(str(edge[0]) + ' ' + str(edge[1]) + ' ' + str(randint(-5,5)), file = edges_file)
    nodes_file.close()
    edges_file.close()
#%%
if __name__ == '__main__':
    mews = '/home/shreya/Desktop/Thesis/gmwcs-solver'
    subject = '128127'
    files_list = ['mean_FA_connectome_5M']
    '''files_list = ['mean_FA_connectome_5M', 'mean_FA_connectome_1M_SIFT',
                  'distances_mean_5M', 'distances_mean_1M_SIFT',
                  'connectome_1M']'''
    for file in files_list:
        print(file, '*'*100)
        make_nodes_edges(file, subject)
        cmd = (f' java -Xss4M -Djava.library.path=/opt/ibm/ILOG/CPLEX_Studio_Community129/cplex/bin/x86-64_linux/ '
              f'-cp /opt/ibm/ILOG/CPLEX_Studio_Community129/cplex/lib/cplex.jar:{mews}/target/gmwcs-solver.jar '
              f'ru.ifmo.ctddev.gmwcs.Main -e {file}_edges '
              f'-n {file}_nodes ')
        os.system(cmd)

