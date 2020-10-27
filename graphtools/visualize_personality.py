import pandas as pd
import matplotlib.pyplot as plt

x_ax = 'Num edges'
y_ax = 'test_roc_auc_ovr_weighted'
column = 'Type of feature'
row = 'Edge'
hue = 'Feature Selection'
label ='Area under ROC curve'
filename = 'Gender_classification_comparison'
base = pd.read_csv('outputs/csvs/base_personality.csv')
#base = base[base['Num_features']<=250]
solver = pd.read_csv('outputs/csvs/solver_personality.csv')
solver = solver.drop(columns=['Num_nodes', '% Positive edges', 'ROI_strl_thresh'])
base = base.drop(columns=['Self_loops', 'ROI_strl_thresh'])
base = base.rename(columns={'Num_features':'Num edges'})
solver.columns
set(base.columns) == set(solver.columns)
'''combined = pd.concat([solver, base], axis=0)
combined.to_csv('outputs/csvs/combined_result_solver_base_personality.csv')
combined = combined[combined['Choice']=='test throw median']'''
combined = base
#sns_plot()
fig, ax = plt.subplots(3,5, figsize=(20, 10))
for (feature , i) in zip(combined['Type of feature'].unique(), range(3)):
    for (target, j) in zip(combined['Target'].unique(), range(5)):
        slice = combined[(combined['Type of feature']== feature) & (combined['Target']==target)&(combined['Choice']=='test throw median')]
        #ax[i][j].plot(slice['Num edges'], slice['test_roc_auc'])
        if feature  =='num_streamlines':
            f = 'Number of streamlines'
        elif feature == 'mean_strl':
            f = 'Mean streamline length'
        else:
            f = 'Mean FA'
        ax[i][j].set_title(f'{f}, {target}')
        lines = []
        label = []
        sel = 'baseline'
        sl = slice[slice['Feature Selection']==sel]
        if sel == 'solver':
            linestyle = '-'
            trans = 0.5
            ll = 2
        elif sel =='baseline':
            linestyle = '-'
            trans = 0.8
            ll = 3
        for clf, color_line  in zip(sl['Classifier'].unique(), ['blue', 'green', 'orange']):
            sl2 = sl[sl['Classifier']==clf]
            l1 = ax[i][j].plot(sl2['Num edges'],sl2['test_roc_auc_ovr_weighted'],
                              linestyle=linestyle, marker = '.', markersize=12,
                          label=f'{clf}_{sel}', color=color_line, alpha =trans, linewidth=ll)[0]
            label.append(f'{clf}_{sel}')
            lines.append(l1)
        ax[i][j].grid(which='minor')
        ax[i][j].grid(which='major')
fig.subplots_adjust(top=0.95)
ax[1,0].set_ylabel('Area under ROC curve', fontsize=16)
ax[2,2].set_xlabel('Number of edges', fontsize=16)
plt.legend(handles=lines, bbox_to_anchor=(1,1), loc='upper left')
fig.tight_layout()
plt.savefig('outputs/figures/combined_clf_auc_base_personality.png')
plt.show()