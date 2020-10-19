'''
import pandas as pd
import copy
from itertools import product
from classification_refined import classify
from processing import *
from readfiles import *
from decision import filter_summary
from subgraphclass import make_solver_summary
from metrics import *
 Data computed for all 5 personality traits at once
# labels for the computed subjects, data.index is the subject id
num = 84  # number of nodes in the graph
tri = int(num * (num + 1) * 0.5)  # we want only the upper diagonal parts since everything below diagonal is 0
labels = ['NEOFAC_A', 'NEOFAC_O', 'NEOFAC_C', 'NEOFAC_N', 'NEOFAC_E']
edge_names = ['mean_FA']
big5 = ['Agreeableness', 'Openness', 'Conscientiousness', 'Neuroticism',
       'Extraversion']
mapping = {k: v for k, v in zip(big5, labels)}
mat = np.triu_indices(84)
mews = '/home/skapoor/Thesis/gmwcs-solver'
metrics = ['balanced_accuracy', 'accuracy', 'f1_weighted', 'roc_auc_ovr_weighted']
# note: right now the matrix whole is not scaled, for computing the fscores and correlation coeff it has to be so.
y_train = computed_subjects()
X_train = generate_combined_matrix(tri, list(
   y_train.index))  # need to check indices till here then convert to numpy array
num_strls = X_train.iloc[:, 2 * tri:]
# make_solver_summary(mapping, y_train, big5, mews, X_train, tri, num_strls, avg_thresh=False)
# filter_summary()
y_test = test_subjects()
X_test = generate_test_data(tri, y_test.index)
mean_FA = X_train.iloc[:, :tri]
mean_FA[:, diag_flattened_indices(84)]
mean_FA.iloc[:, diag_flattened_indices(84)].mean()
mean_FA.iloc[:, set(range(3570)).difference(set(diag_flattened_indices(84)))].mean().mean()
mean_FA.max().max()
mean_FA.mean().max()
#%%
'''
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('outputs/csvs/summary.csv')
fig = plt.figure(figsize=(50,40))
target = 'Gender'
g = sns.FacetGrid(df, col='Feature_type', row ='Edge', sharex=True, margin_titles=True,
                  legend_out=True)
g = g.map(plt.plot,'Output_Graph_nodes', 'Output_Graph_edges' )
g= (g.map(sns.scatterplot, 'Output_Graph_nodes', 'Output_Graph_edges')).add_legend()
#[plt.setp(ax.texts, text="") for ax in g.axes.flat]
# remove the original texts
# important to add this before setting titles
#g.set_titles(row_template = '{row_name}', col_template = '{col_name}')
fig.suptitle('Gender Classification - Solver')
plt.savefig(f'outputs/figures/{target}_nodes_preserved.png')
plt.show()
#%%
solver = pd.read_csv('outputs/csvs/solver.csv')
s1 = solver.groupby(by=['Type of feature', 'Edge'])

def multi_plot(x, y1,y2, y3, y4, **kwargs):
    plt.ylabel('metrics')
    plt.scatter(x, y1, c='red', label = 'test_roc_auc_ovr_weighted',alpha =0.5 )
    plt.scatter(x,y2, c='blue', label = 'test_accuracy', alpha =0.5 )
    plt.scatter(x,y3, c='pink', label= 'test_balanced_accuracy', alpha =0.5 )
    plt.scatter(x,y4, c='orange', label= 'test_f1_weighted', alpha =0.5)
    plt.legend()


#%%
fig = plt.figure(figsize=(50,40))
g = sns.FacetGrid(solver, col='Type of feature', row ='Edge', hue='Classifier', sharex=True, margin_titles=True,
                  legend_out=True,hue_kws=dict(color=['red','blue','pink','orange', "1","2","3","4"]))
g= (g.map(multi_plot, 'Num edges','test_roc_auc_ovr_weighted', 'test_accuracy',
      'test_balanced_accuracy', 'test_f1_weighted')).add_legend()
#[plt.setp(ax.texts, text="") for ax in g.axes.flat]
# remove the original texts
                                                    # important to add this before setting titles
#g.set_titles(row_template = '{row_name}', col_template = '{col_name}')
g.axes[0,0].set_ylabel('metrics')
g.axes[1,0].set_ylabel('metrics')
fig.suptitle('Gender Classification - Solver')
plt.savefig('outputs/figures/solver_results.png')
plt.show()
#%%
base=pd.read_csv('outputs/csvs/base.csv')
base = base[base['Num_features']<=1000]
fig = plt.figure(figsize=(50,40))
g = sns.FacetGrid(base, col='Type of feature', row ='Edge', sharex=True, margin_titles=True,
                  legend_out=True,hue_kws=dict(color=['red','blue','pink','orange', "1","2","3","4"]))
g= (g.map(multi_plot, 'Num_features','test_roc_auc_ovr_weighted', 'test_accuracy',
      'test_balanced_accuracy', 'test_f1_weighted')).add_legend()
#[plt.setp(ax.texts, text="") for ax in g.axes.flat]
# remove the original texts
                                                    # important to add this before setting titles
#g.set_titles(row_template = '{row_name}', col_template = '{col_name}')
g.axes[0,0].set_ylabel('metrics')
g.axes[1,0].set_ylabel('metrics')
fig.suptitle('Gender classification- Baseline')
plt.savefig('outputs/figures/baseline_results.png')
plt.show()
#%%
fig = plt.figure(figsize=(50,40))
g = sns.FacetGrid(solver, col='Type of feature', row ='Classifier', sharex=True, margin_titles=True,
                  legend_out=True,hue_kws=dict(color=['red','blue','pink','orange', "1","2","3","4"]))
g= (g.map(multi_plot, 'Num edges','test_roc_auc_ovr_weighted', 'test_accuracy',
      'test_balanced_accuracy', 'test_f1_weighted')).add_legend()
#[plt.setp(ax.texts, text="") for ax in g.axes.flat]
# remove the original texts
                                                    # important to add this before setting titles
#g.set_titles(row_template = '{row_name}', col_template = '{col_name}')
g.axes[0,0].set_ylabel('metrics')
g.axes[1,0].set_ylabel('metrics')
fig.suptitle('Gender Classification - Solver')
plt.savefig('outputs/figures/solver_results_classifier.png')
plt.show()
#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
base = pd.read_csv('outputs/csvs/base.csv')
base = base[base['Num_features']<=1000]
solver = pd.read_csv('outputs/csvs/solver.csv')
solver = solver.drop(columns=['Num_nodes', '% Positive edges', 'ROI_strl_thresh'])
base = base.drop(columns=['Self_loops', 'ROI_strl_thresh'])
base = base.rename(columns={'Num_features':'Num edges'})
base.columns
solver.columns
set(base.columns) == set(solver.columns)
combined = pd.concat([solver, base], axis=0)
combined.to_csv('combined_result_solver_base.csv')
#%%
for clf in ['SVC', 'RF', 'MLP']:
    svc = combined[combined['Classifier'] == clf]
    fig = plt.figure(figsize=(50,40))
    g = sns.FacetGrid(svc, col='Type of feature', row ='Edge', margin_titles=True,
                      legend_out=True, hue='Feature Selection')

    g = g.map(plt.plot,'Num edges','test_roc_auc_ovr_weighted' )
    g= (g.map(sns.scatterplot, 'Num edges','test_roc_auc_ovr_weighted')).add_legend()
    #[plt.setp(ax.texts, text="") for ax in g.axes.flat]
    # remove the original texts
                                                        # important to add this before setting titles
    #g.set_titles(row_template = '{row_name}', col_template = '{col_name}')
    g.fig.subplots_adjust(top=0.5)
    g.fig.suptitle(f'Gender classification- {clf}')
    plt.savefig(f'outputs/figures/comparison_roc_auc_{clf}.png')
    plt.show()