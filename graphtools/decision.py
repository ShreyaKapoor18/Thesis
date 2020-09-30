# %%
import pandas as pd
from matplotlib import pyplot
import seaborn

'''
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
from sklearn.tree import DecisionTreeClassifier
#%%
from six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
import os'''
'''
Aim is to design a decision tree classifier based on the fact if the 
solver produces an output or not
'''


# %%
def filter_summary():
    summary = pd.read_csv('/home/skapoor/Thesis/graphtools/outputs/csvs/summary.csv')
    summary.sort_values(by=['Features_preserved_per', 'Output_graph_posedge_per'],
                        ascending=[False, False], inplace=True)
    summary.fillna(0, inplace=True)
    summary.sort_values(['Features_preserved_per', 'Output_graph_posedge_per'], ascending=[False, False])
    filtered = summary[(summary.Features_preserved_per > 0) & (summary.Output_graph_posedge_per > 0) & (
                summary.Output_Graph_edges > 4)].copy()
    filtered.drop(filtered.columns[0], axis=1, inplace=True)
    filtered.drop_duplicates(subset=["Features_preserved_per", 'Output_Graph_edges'], inplace=True)
    filtered.index = range(len(filtered))
    filtered = filtered.round(3)  # round off to three decimal places
    filtered.to_csv('/home/skapoor/Thesis/graphtools/outputs/csvs/filtered.csv')  # make sure compiled with same version


# %%
'''
for target in summary['Target'].unique():
	df = summary[summary['Target']==target].iloc[:,:6]
	y = summary[summary['Target'] == target].Features_preserved_per
	y = y>0
	# encode all the categorical variables
	le = LabelEncoder()
	df.Feature_type = le.fit_transform(df.Feature_type)
	c1 = list(le.classes_)
	df.Edge = le.fit_transform(df.Edge)
	c2 = list(le.classes_)
	df.Node_weights = le.fit_transform(df.Node_weights)
	c4 = list(le.classes_)
	df.drop(['Target'], axis=1, inplace=True)
	columnTransformer = ColumnTransformer([('encoder',
											OneHotEncoder(),
											[0,2,3])],
										remainder='passthrough')
	data = np.array(columnTransformer.fit_transform(df), dtype=np.str)

	c1.extend(c2)
	c1.extend(c4)
	c1.extend(['Factor', 'Subtracted_value'])
	data = pd.DataFrame(data, columns=c1)

	dtree=DecisionTreeClassifier(max_depth=5)
	dtree.fit(data,y)

	dot_data = StringIO()
	export_graphviz(dtree, out_file=dot_data,
					filled=True, rounded=True,
					special_characters=True)
	graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
	Image(graph.create_png())
	export_graphviz(dtree, out_file=f'outputs/figures/{target}.dot', feature_names=data.columns)
	os.system(f'dot -Tpng outputs/figures/{target}.dot -o outputs/figures/{target}.png')
'''


# %%
# different decision trees for each personality tratiz
def read_summary_self_loops():
    df = pd.read_csv('outputs/csvs/base.csv')
    fg = seaborn.FacetGrid(data=df, hue='Self_loops', aspect=1.61, col='Edge', row='Percentage')
    fg.map(pyplot.plot, 'test_roc_auc_ovr_weighted').add_legend()
    pyplot.show()
