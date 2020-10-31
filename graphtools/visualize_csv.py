import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
#%%
def sns_plot(**kwargs):
    for clf in ['SVC', 'RF', 'MLP']:
        svc = combined[combined['Classifier'] == clf]
        fig = plt.figure(figsize=(50,40))
        g = sns.FacetGrid(svc, col=column, row = row, margin_titles=True,
                          legend_out=True, hue=hue)

        g = g.map(plt.plot, x_ax,y_ax, marker='.', markersize=12)
        #[plt.setp(ax.texts, text="") for ax in g.axes.flat]
        # remove the original text
        # important to add this before setting titles
        g.axes[0, 0].set_ylabel(label)
        g.axes[1, 0].set_ylabel(label)
        for i in range(g.axes.shape[0]):
            for j in range(g.axes.shape[1]):
                g.axes[i][j].grid(which='minor', alpha=0.2)
                g.axes[i][j].grid(which='major', alpha=0.5)
        #g.set_titles(row_template = '{row_name}', col_template = '{col_name}')
        g.fig.suptitle(f'Gender classification on test set- {clf}')
        g.fig.subplots_adjust(top=0.9)
        plt.savefig(f'outputs/figures/{filename}_{clf}.png')
        plt.show()


#%%
x_ax = 'Num edges'
y_ax = 'test_roc_auc_ovr_weighted'
column = 'Type of feature'
row = 'Edge'
hue = 'Feature Selection'
label ='Area under ROC curve'
filename = 'Gender_classification_comparison'
base = pd.read_csv('outputs/csvs/base.csv')
base = base[base['Num_features']<=1000]
solver = pd.read_csv('outputs/csvs/solver.csv')
solver = solver.drop(columns=['Num_nodes', '% Positive edges', 'ROI_strl_thresh'])
base = base.drop(columns=['Self_loops', 'ROI_strl_thresh'])
base = base.rename(columns={'Num_features':'Num edges'})
solver.columns
set(base.columns) == set(solver.columns)
combined = pd.concat([solver, base], axis=0)
#%%
combined.to_csv('outputs/csvs/combined_result_solver_base.csv')
#%%
#sns_plot()
fig, ax = plt.subplots(3,2, figsize=(10, 10))
for (feature , i) in zip(combined['Type of feature'].unique(), range(3)):
    for (edge, j) in zip(combined['Edge'].unique(), range(2)):
        slice = combined[(combined['Type of feature']== feature) & (combined['Edge']==edge)]
        #ax[i][j].plot(slice['Num edges'], slice['test_roc_auc'])
        if feature  =='num_streamlines':
            f = 'Number of streamlines'
        elif feature == 'mean_strl':
            f = 'Mean streamline length'
        else:
            f = 'Mean FA'
        ax[i][j].set_title(f'{f}, {edge}')
        lines = []
        label = []
        for sel in slice['Feature Selection'].unique():
            sl = slice[slice['Feature Selection']==sel]
            if sel == 'solver':
                linestyle = '-'
                trans = 0.5
                ll = 2
            elif sel =='baseline':
                linestyle = ':'
                trans = 0.8
                ll = 3
            for clf, color_line  in zip(sl['Classifier'].unique(), ['blue', 'green', 'orange']):
                sl2 = sl[sl['Classifier']==clf]
                l1 = ax[i][j].plot(sl2['Num edges'],sl2['test_roc_auc_ovr_weighted'],
                                  linestyle=linestyle, marker = '.', markersize=12, c='blue',
                              label=f'{clf}_{sel}', color=color_line, alpha =trans, linewidth=ll)[0]
                label.append(f'{clf}_{sel}')
                lines.append(l1)
        ax[i][j].grid(which='minor')
        ax[i][j].grid(which='major')
#plt.legend(lines, label)
#fig.subplots_adjust(top=0.95)
#fig.suptitle('Gender Classification')
#plt.xlabel('Number of edges')
#plt.ylabel('Area under ROC Curve')
ax[1,0].set_ylabel('Area under ROC curve')
fig.text(0.5, 0.002, 'Number of edges', ha='center', fontsize=12)

#fig.subplots_adjust(top=0.2)
plt.legend(handles=lines, bbox_to_anchor=(1,1), loc='upper left')
fig.tight_layout(pad=1)
plt.savefig('outputs/figures/combined_clf_auc_gender.png')
plt.show()
#%%
fig = plt.figure(figsize=(8,6))
slice = combined[(combined['Type of feature']== 'num_streamlines') & (combined['Edge']=='fscores')]
f = 'number of streamlines'
plt.title(f'Classification results based on {f} and fscores')
lines = []
label = []
for sel in slice['Feature Selection'].unique():
    sl = slice[slice['Feature Selection'] == sel]
    if sel == 'solver':
        linestyle = '-'
        trans = 0.5
        ll = 2
    elif sel == 'baseline':
        linestyle = ':'
        trans = 0.8
        ll = 3
    for clf, color_line in zip(sl['Classifier'].unique(), ['blue', 'green', 'orange']):
        sl2 = sl[sl['Classifier'] == clf]
        l1 = plt.plot(sl2['Num edges'], sl2['test_roc_auc_ovr_weighted'],
                           linestyle=linestyle, marker='.', markersize=12,
                           label=f'{clf}_{sel}', color=color_line, alpha=trans, linewidth=ll)[0]
        label.append(f'{clf}_{sel}')
        lines.append(l1)
plt.grid(which='minor')
plt.grid(which='major')
plt.ylabel('Area under ROC curve')
plt.xlabel('Number of edges')
plt.legend(handles=lines, bbox_to_anchor=(1,1), loc='upper left')
plt.savefig('outputs/figures/select_clf_auc_gender.png')
plt.show()
#%%
from scipy.optimize import curve_fit

def func(x, a, b, c):
  return a + b * x + c * x ** 2
#%%
df = pd.read_csv('outputs/csvs/summary.csv')
df.fillna(0, inplace=True)
fig, ax = plt.subplots(3,2, figsize=(8, 9), sharex=True, sharey=True)

for (feature , i) in zip(df['Feature_type'].unique(), range(3)):
    for (edge, j) in zip(df['Edge'].unique(), range(2)):
        slice =df[(df['Feature_type']== feature) & (df['Edge']==edge)]
        #ax[i][j].plot(slice['Num edges'], slice['test_roc_auc'])
        if feature  =='num_streamlines':
            f = 'Number of streamlines'
        elif feature == 'mean_strl':
            f = 'Mean streamline length'
        else:
            f = 'Mean FA'
        ax[i][j].set_title(f'{f}, {edge}')
        x = slice['Output_Graph_nodes']
        y = slice['Output_Graph_edges']
        ax[i][j].plot(x, y, linestyle='solid', marker = '.', markersize=12, c='blue',
                               color='black', alpha =0.5, linewidth=2, label= 'observed')
        popt, _ = curve_fit(func, x, y)
        print(popt)
        xnew = np.linspace(x.iloc[0], x.iloc[-1], 1000)
        ax[i][j].annotate(f'y= {popt[2].round(3)}x^2+{popt[1].round(3)}x  {popt[0].round(3)}',
                    xy=(0,0), xycoords='data',
                    xytext=(0,80), textcoords='offset points',
                    horizontalalignment='left', verticalalignment='top')
        ax[i][j].plot(xnew, func(xnew, *popt), 'r-', label='fitted model')
        ax[i][j].grid(which='minor')
        ax[i][j].grid(which='major')
        #ax[i][j].set_ýlabel('Number of edges')
#plt.legend(lines, label)
fig.subplots_adjust(top=0.4)
fig.suptitle('Output graphs based on gender', fontsize=12)
fig.text(0.5, 0.001, 'Number of nodes', ha='center', fontsize=12)
fig.text(0.001, 0.5, 'Number of edges', va='center', rotation='vertical')
plt.legend()
plt.tight_layout()
plt.savefig('outputs/figures/Gender_nodes_preserved.png')
plt.show()
#%%
df = pd.read_csv('outputs/csvs/summary.csv')
df.fillna(0, inplace=True)
fig, ax = plt.subplots(3,2, figsize=(8, 9), sharex=True, sharey=True)

for (feature , i) in zip(df['Feature_type'].unique(), range(3)):
    for (edge, j) in zip(df['Edge'].unique(), range(2)):
        slice =df[(df['Feature_type']== feature) & (df['Edge']==edge)]
        #ax[i][j].plot(slice['Num edges'], slice['test_roc_auc'])
        if feature  =='num_streamlines':
            f = 'Number of streamlines'
        elif feature == 'mean_strl':
            f = 'Mean streamline length'
        else:
            f = 'Mean FA'
        ax[i][j].set_title(f'{f}, {edge}')
        x = slice['Output_Graph_nodes']
        y = slice['Output_Graph_edges']
        ax[i][j].plot(x, y, linestyle='solid', marker = '.', markersize=12, c='blue',
                               color='black', alpha =0.5, linewidth=2, label= 'observed')
        popt, _ = curve_fit(func, x, y)
        print(popt)
        xnew = np.linspace(x.iloc[0], x.iloc[-1], 1000)
        ax[i][j].annotate(f'y= {popt[2].round(3)}x^2+{popt[1].round(3)}x  {popt[0].round(3)}',
                    xy=(0,0), xycoords='data',
                    xytext=(0,80), textcoords='offset points',
                    horizontalalignment='left', verticalalignment='top')
        ax[i][j].plot(xnew, func(xnew, *popt), 'r-', label='fitted model')
        ax[i][j].grid(which='minor')
        ax[i][j].grid(which='major')
        #ax[i][j].set_ýlabel('Number of edges')
#plt.legend(lines, label)
fig.subplots_adjust(top=0.4)
fig.suptitle('Output graphs based on gender', fontsize=12)
fig.text(0.5, 0.001, 'Number of nodes', ha='center', fontsize=12)
fig.text(0.001, 0.5, 'Number of edges', va='center', rotation='vertical')
plt.legend()
plt.tight_layout()
plt.savefig('outputs/figures/Gender_nodes_preserved.png')
plt.show()

#%%
df = pd.read_csv('outputs/csvs/summary.csv')
fig = plt.figure(figsize=(6,5))
target = 'Gender'
df = df[(df['Feature_type']=='num_streamlines') & (df['Edge']=='fscores')]
x = df['Output_Graph_nodes']
y = df['Output_Graph_edges']
plt.plot(x, y, linestyle='solid', marker='.', markersize=12, c='blue',
              color='black', alpha=0.5, linewidth=2, label='observed')
popt, _ = curve_fit(func, x, y)
xnew = np.linspace(x.iloc[0], x.iloc[-1], 1000)
plt.annotate(f'y= {popt[2].round(3)}x^2+{popt[1].round(3)}x  {popt[0].round(3)}',
                  xy=(0, 0), xycoords='data',
                  xytext=(0, 100), textcoords='offset points',
                  horizontalalignment='left', verticalalignment='top', fontsize=12)
plt.plot(xnew, func(xnew, *popt), 'r-', label='fitted curve')
plt.xlabel('Number of nodes')
plt.ylabel('Number of edges')
plt.xlim(0,30)
plt.grid(which='minor')
plt.grid(which='major')
plt.legend()
plt.tight_layout()
plt.savefig('outputs/figures/Gender_nodes_preserved.png')
plt.show()