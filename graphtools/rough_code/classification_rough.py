# For generating a dictionary structure
#%%
'''arr = {}
for label in labels:
    arr[label] = {'mean_FA':[] ,'mean strl': [], 'num strl': []}
    for i in range(3):
        key = list(arr[label].keys())[i]
        for edge in range(i,(i+1)*7056):
            if edge not in labels:
                #print (edge, f_score(data_edges, edge, 'NEOFAC_A'))
                arr[label][key].append(f_score(data_edges,edge, label))'''
#%%