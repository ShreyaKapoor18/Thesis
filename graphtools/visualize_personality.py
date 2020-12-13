import pandas as pd
import matplotlib.pyplot as plt

x_ax = 'Num edges'
y_ax = 'test_roc_auc_ovr_weighted'
column = 'Type of feature'
row = 'Edge'
hue = 'Feature Selection'
label ='Area under ROC curve'
base = pd.read_csv('outputs/csvs/base_personality_3.csv')
#base = base[base['Num_features']<=250]
solver = pd.read_csv('outputs/csvs/solver_personality_3.csv')
solver = solver.drop(columns=['Num_nodes', '% Positive edges', 'ROI_strl_thresh'])
base = base.drop(columns=['Self_loops', 'ROI_strl_thresh'])
base = base.rename(columns={'Num_features':'Num edges'})
solver.columns
set(base.columns) == set(solver.columns)
'''combined = pd.concat([solver, base], axis=0)
combined.to_csv('outputs/csvs/combined_result_solver_base_personality.csv')
combined = combined[combined['Choice']=='test throw median']'''
#%%
comb1 = base[(base['Target']=='Extraversion') & (base['Type of feature']=='num_streamlines') & (base['Choice']=='test throw median')]
comb2 = base[(base['Target']=='Neuroticism') & (base['Type of feature']=='num_streamlines')& (base['Choice']=='test throw median')]
fig,ax  = plt.subplots(2,1, sharey=True, sharex=True)

for clf, color_line in zip(['SVC', 'RF', 'MLP'], ['blue', 'green', 'orange']):
    sl = comb1[comb1['Classifier']==clf]
    sl2 = comb2[comb2['Classifier']==clf]
    ax[0].plot(sl['Num edges'], sl['test_roc_auc_ovr_weighted'], marker='.', markersize=12,
                       label=f'{clf}', color=color_line, alpha=0.5)
    ax[1].plot(sl2['Num edges'], sl2['test_roc_auc_ovr_weighted'], marker='.', markersize=12,
                       label=f'{clf}', color=color_line, alpha=0.5)
    ax[0].set_title('Extraversion | Number of streamlines')
    ax[1].set_title('Neuroticism | Number of Streamlines')
    ax[0].grid(which='minor')
    ax[0].grid(which='major')
    ax[1].grid(which='minor')
    ax[1].grid(which='major')
plt.xlabel('Percentage')
#plt.ylabel('Area under ROC curve')
fig.text(0.015, 0.3, 'Area under ROC curve', ha='center', fontsize=12, rotation='vertical')
plt.legend()
plt.tight_layout(pad=2)
plt.savefig('outputs/figures/persona_2.png')
plt.show()

#%%
# compare solver and base
combined = pd.concat([solver, base], axis=0)
#combined.to_csv('outputs/csvs/combined_result_solver_base_personality.csv')
combined = combined[combined['Choice']=='test throw median']

#%%
#sns_plot()
fig, ax = plt.subplots(2,1, figsize=(8, 9), constrained_layout=True)
for (target, feature), i  in zip([('Extraversion', 'num_streamlines'), ('Neuroticism', 'num_streamlines')], range(2)):
        slice = combined[(combined['Type of feature']== feature) & (combined['Target']==target)]
        #ax[i][j].plot(slice['Num edges'], slice['test_roc_auc'])
        if feature  =='num_streamlines':
            f = 'Number of streamlines'
        elif feature == 'mean_strl':
            f = 'Mean streamline length'
        else:
            f = 'Mean FA'
        ax[i].set_title(f'{f}, {target}')
        lines = []
        label = []
        for sel in ['baseline', 'solver']:
            sl = slice[slice['Feature Selection']==sel]
            if sel == 'solver':
                linestyle = '-'
                trans = 0.5
                ll = 2
                cl = 'blue'
            elif sel =='baseline':
                linestyle = ':'
                trans = 0.8
                ll = 3
                cl = 'orange'
            for clf in ['SVC']:
                sl2 = sl[sl['Classifier']==clf]
                l1 = ax[i].plot(sl2['Num edges'],sl2['test_roc_auc_ovr_weighted'],
                                  linestyle=linestyle, marker = '.', markersize=12,
                              label=f'{sel}', color=cl, alpha =trans, linewidth=ll)[0]
                label.append(f'{sel}')
                lines.append(l1)
        ax[i].grid(which='minor')
        ax[i].grid(which='major')
        ax[i].set_xticks(range(20,320,10))
        ax[i].tick_params(labelrotation=60)

ax[1].set_xlabel('Number of edges', fontsize=12)
fig.text(0.01, 0.5, 'Area under ROC curve', ha='center', fontsize=12, rotation='vertical')
plt.suptitle('Classification based on number of streamlines and Pearson correlation with SVC')
#plt.subplots_adjust(top=0.2)
#fig.tight_layout(pad=0)
plt.legend(handles=lines, bbox_to_anchor=(0.95,1.08), loc='upper left')
plt.savefig('outputs/figures/persona_comp.png')
plt.show()
#%%