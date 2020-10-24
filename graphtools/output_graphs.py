#%%
from graphclass import *
from readfiles import corresp_label_file
import numpy as np
import matplotlib.pyplot as plt
'''
Try to analyze the importance of each node based
on the output nodes obtained from the solver based approach for 
gender
We can hence get the gender importance
Starting from 3 nodes it actually gives us results so we can get iterative 
node importance
'''
mews = '/home/skapoor/Thesis/gmwcs-solver'
d1 = corresp_label_file('fs_default.txt')
counter = np.zeros((84)) # counter increases as soon as the node is found in this case
for max_num_nodes in range(4,31):
    output_graph = BrainGraph('t_test', 'mean_FA', 'const', 'Gender', max_num_nodes, -0.01, 0)
    output_graph.read_from_file(mews, input_graph=False)
    #print(output_graph.nodes)
    for node in output_graph.nodes:
        print(d1[node+1])
        counter[node] +=1

#%%
np.random.seed(19680801)
plt.rcdefaults()
fig, ax = plt.subplots(figsize=(10,15))
people = [x for x,i in zip(list(d1.values()), range(84)) if counter[i]!=0]
y_pos = [x for x in np.arange(84) if counter[x]!=0]
performance = [x for x in counter if x!=0]

ax.barh(y_pos, performance, align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(people)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Importance')
ax.set_title('Importance of nodes for Gender classification')
plt.tight_layout(pad=1)
plt.show()