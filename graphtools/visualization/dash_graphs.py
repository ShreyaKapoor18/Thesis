# %% md
'''
In this example we show how to visualize a network graph created using `networkx`.

Install the Python library `networkx` with `pip install networkx`.
'''
import json

from scipy.stats import describe
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from classification import data_splitting
from graphclass import *
from metrics import fscore
from paramopt import graph_options
from processing import generate_combined_matrix
from readfiles import computed_subjects
from subgraphclass import nested_outputdirs
from subgraphclass import train_with_best_params

# %% md

### Create random graph
# %%
''' Data computed for all 5 personality traits at once'''
data = computed_subjects()  # labels for the computed subjects, data.index is the subject id
num = 84  # number of nodes in the graph
tri = int(num * (num + 1) * 0.5)  # we want only the upper diagonal parts since everything below diagonal is 0
whole = generate_combined_matrix(tri, list(data.index))  # need to check indices till here then convert to numpy array
assert list(whole.index) == list(data.index)
labels = ['NEOFAC_A', 'NEOFAC_O', 'NEOFAC_C', 'NEOFAC_N', 'NEOFAC_E']
edge_names = ['mean_FA', 'mean_strl', 'num_streamlines']
big5 = ['Agreeableness', 'Openness', 'Conscientiousness', 'Neuroticism',
        'Extraversion']
mapping = {k: v for k, v in zip(big5, labels)}
mat = np.triu_indices(84)
mews = '/home/skapoor/Thesis/gmwcs-solver'
# before
# this is what is supposed to be done
metrics = ['balanced_accuracy', 'accuracy', 'f1_weighted', 'roc_auc_ovr_weighted']
# note: right now the matrix whole is not scaled, for computing the fscores and correlation coeff it has to be so.
feature_type = 'mean_FA'
target = 'Agreeableness'
target_col = data[mapping[target]]
edge = 'pearson'
node_wts = 'max'
threshold = 85
plotting_options = graph_options(color='red', node_size=5, line_color='white', linewidhts=0.1, width=1)
# %%
if feature_type == 'mean_FA':
    whole = whole.iloc[:, :tri]
elif feature_type == 'mean_strl':
    whole = whole.iloc[:, tri:2 * tri]
elif feature_type == 'num_streamlines':
    whole = whole.iloc[:, 2 * tri:]  # input one feature at a time
print(f'the {feature_type} feature is being used; the shape of the matrix is:', whole.shape)
nested_outputdirs(mews='/home/skapoor/Thesis/gmwcs-solver')
f = open(f'/home/skapoor/Thesis/graphtools/outputs/dicts/{target}_combined_params.json', 'r')

best_params = json.load(f)
i = big5.index(target)
index = list(range(tri))  # if we are using only one type of feature at a time, lets say mean FA
# since we are using one feature at a time
# for choice in ['qcut', 'median', 'throw median']
# Let's say we only choose the throw median choice, because it is the one that makes more sense
choice = 'throw median'  # out of all these we will use these particular choices only!
X, y = data_splitting(choice, index, whole, target_col)  # this X is for random forests training
params = best_params['RF'][choice]["100"]  # maybe use the parameters that work the best for top 5%
feature_imp = train_with_best_params('RF', params, X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y)
scalar = StandardScaler()
X_train = pd.DataFrame(scalar.fit_transform(X_train), index=X_train.index)
X_test = pd.DataFrame(scalar.transform(X_test), index=X_test.index)
# train a graph based only on the training set
# get fscores for all the features, only based on the training set.
# maybe I can make a graph class based on this and initialize on the basis of the properties
stacked = pd.concat([X_train, y_train], axis=1)
cols = []
cols.extend(range(X_train.shape[1]))  # the values zero to the number of columns
cols.append(target_col.name)
stacked.columns = cols
if edge == 'fscores':
    arr = fscore(stacked, class_col=target_col.name)[:-1]  # take this only from the training data
if edge == 'pearson':
    arr = stacked.corr().iloc[:-1, -1]
if edge == 'feature_importance':
    arr = feature_imp
    # assert type(arr) == np.ndarray
input_graph = BrainGraph(edge, feature_type, 'const', target)
input_graph.make_graph(arr, 0.1)
input_graph.set_node_labels('const', 0)
# input_graph.normalize_node_attr()
input_graph.savefiles(mews)
input_graph.visualize_graph(mews, True, 0.1, plotting_options, 0)
print('Describing the node labels of the input graph', describe(input_graph.node_labels))
print('Describing the edge weights of the input graph', describe(input_graph.edge_weights))

output_graph = BrainGraph(edge, feature_type, node_wts, target)
reduced_feature_indices = output_graph.read_from_file(mews)
output_graph.visualize_graph(mews, False, 0.1, plotting_options, 0)
print('Describing the node labels of the input graph', describe(output_graph.node_labels))
print('Describing the edge weights of the input graph', describe(output_graph.edge_weights))
# get nodes and edges of this graph
# train algorithm accordingly
X_train = X_train.iloc[:, reduced_feature_indices]
X_test = X_test.iloc[:, reduced_feature_indices]

G = input_graph  # but we have to plot the reduced graph here and this needs to be solved
pos = nx.spring_layout(G)
for node in G.nodes.keys():
    G.nodes[node]['pos'] = pos[node]

# %%

import plotly.graph_objects as go

# %%  md

#### Create Edges
# Add edges as disconnected lines in a single trace and nodes as a scatter trace

# %%
color = []
if G.edge_weights != None and G.edge_weights != []:
    minima = min(G.edge_weights)
    maxima = max(G.edge_weights)
    norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima)
    mapper = cm.ScalarMappable(norm=norm, cmap=plt.get_cmap('Spectral'))
    color = []
    for v in G.edge_weights:
        color.append(tuple([int(round(x * 255.0)) for x in mapper.to_rgba(v)[:-1]]))

    # build a rectangle in axes coords

edge_trace = []
for edge, c in zip(G.edges(), color):
    edge_x = []
    edge_y = []
    x0, y0 = G.nodes[edge[0]]['pos']
    x1, y1 = G.nodes[edge[1]]['pos']
    edge_x.append(x0)
    edge_x.append(x1)
    edge_x.append(None)
    edge_y.append(y0)
    edge_y.append(y1)
    edge_y.append(None)
    edge_trace.append(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color=f'rgb{c}'),
        hoverinfo='none',
        mode='lines'))

node_x = []
node_y = []
for node in G.nodes():
    x, y = G.nodes[node]['pos']
    node_x.append(x)
    node_y.append(y)

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers',
    hoverinfo='text',
    marker=dict(
        showscale=True,
        # colorscale options
        # 'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
        # 'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
        # 'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
        colorscale='YlGnBu',
        reversescale=True,
        color=[],
        size=10,
        colorbar=dict(
            thickness=15,
            title='Node Connections',
            xanchor='left',
            titleside='right'
        ),
        line_width=2))

# %% md

#### Color Node Points
# Color node points by the number of connections.

# Another option would be to size points by the number of connections
# i.e. ```node_trace.marker.size = node_adjacencies```

# %%

node_adjacencies = []
node_text = []
'''for node, adjacencies in enumerate(G.adjacency()):
    node_adjacencies.append(len(adjacencies[1]))
    node_text.append('# of connections: '+str(len(adjacencies[1])))'''
for l in G.nodes.keys():
    if node_wts == 'max':
        node_adjacencies.append(max([dict(G[l])[k]['weight'] for k in dict(G[l]).keys()]))

node_trace.marker.color = node_adjacencies
node_trace.text = node_text

# %% md

#### Create Network Graph

# %%

fig = go.Figure(data=edge_trace + [node_trace],
                layout=go.Layout(
                    title='<br>Network graph made with Python',
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20, l=5, r=5, t=40),
                    annotations=[dict(
                        text="Python code: <a href='https://plotly.com/ipython-notebooks/network-graphs/'> https://plotly.com/ipython-notebooks/network-graphs/</a>",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002)],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
# fig.show()

# %% md

#### Reference
# See https://plotly.com/python/reference/scatter/ for more information and chart attribute options!


# %%

import dash
import dash_core_components as dcc
import dash_html_components as html

app = dash.Dash()
app.layout = html.Div([
    dcc.Graph(figure=fig)
])
if __name__ == '__main__':
    app.run_server(debug=True)  # Turn off reloader if inside Jupyter
