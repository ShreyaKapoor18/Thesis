import pandas as pd
from graphclass import BrainGraph
import numpy as np
from readfiles import corresp_label_file
'''
Trying to know which nodes are getting selected in most of the cases.
The intensity in each of the cells of the array tells us about which node is
being selected in most of the use cases. We have one matrix for each personality trait and 
each maximum number of nodes, using a visualized heatmap we can see it very easily. 
'''
mews = '/home/skapoor/Thesis/gmwcs-solver'
# grouping the networks based on personality traits and number of nodes
big5 = ['Agreeableness', 'Openness', 'Conscientiousness', 'Neuroticism',
        'Extraversion']

for trait in big5:
    for max_num_nodes in [5, 10, 15, 20, 25, 30]:
        a = np.zeros(84)
        edges = np.zeros((84, 84))
        for thresh in [0.01, 0.001, 0.005, 0.0055]:
            output_graph = BrainGraph('pearson', 'mean_FA', 'const', trait, max_num_nodes, val=-0.01, thresh=thresh)
            output_graph.read_from_file(mews, input_graph=False) # as it is not an input graph
            for node in output_graph.nodes:
                a[node] += 1
            for u,v in output_graph.edges:
                edges[u, v] += 1
        a = pd.Series(a, index=corresp_label_file('fs_default.txt').keys())
        edges = pd.DataFrame(edges,columns=corresp_label_file('fs_default.txt').keys(),
                             index=corresp_label_file('fs_default.txt').keys())
        a.to_csv(f'{mews}/outputs/csvs/{trait}_{max_num_nodes}_nodes.csv')
        edges.to_csv(f'{mews}/outputs/csvs/{trait}_{max_num_nodes}_edges.csv')
