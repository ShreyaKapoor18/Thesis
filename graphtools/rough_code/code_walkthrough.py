
import copy
from itertools import product
from classification_refined import *
from processing import *
from readfiles import *
from decision import filter_summary
from subgraphclass import make_solver_summary
from paramopt import *


# In[5]:


tri = len(np.triu_indices(84)[0])

labels = ['NEOFAC_A', 'NEOFAC_O', 'NEOFAC_C', 'NEOFAC_N', 'NEOFAC_E']
edge_names = ['mean_FA']
big5 = ['Agreeableness', 'Openness', 'Conscientiousness', 'Neuroticism',
        'Extraversion']
mapping = {k: v for k, v in zip(big5, labels)}
# In[6]:


y_train = computed_subjects() #maybe make changes to the the names and name them just as training and test.
X_train = generate_combined_matrix(tri, list(y_train.index))
y_test = test_subjects()
X_test = generate_test_data(tri, y_test.index)


# In[7]:


X_train.head()  # mean_FA, mean_strl, num_strls


# In[8]:


y_train.head()
#%%

params = pd.read_csv('/outputs/csvs/filtered.csv', index_col=None)
#%%
def solver_viz(X_train, X_test, y_train, strls_num, feature, thresh, val, max_num_nodes, node_wts=None,
           target=None, edge=None):
    X_train, X_test, arr = process_raw(X_train, X_test, y_train, feature)
    arr = arr.abs()
    arr = pd.DataFrame(arr, index=arr.index)  # scale the array according to the index
    input_graph = BrainGraph(edge, feature, node_wts, target, max_num_nodes, val, thresh)
    if not os.path.exists(f'{mews}/outputs/edges/{input_graph.filename}.out')\
            and not os.path.exists(f'{mews}/outputs/nodes/{input_graph.filename}.out'):
        print ('the solver is being called since the reduced file does not exist')
        input_graph.make_graph(arr, strls_num, thresh)
        if node_wts == 'const':
            input_graph.set_node_labels(node_wts, const_val=val)
        else:
            input_graph.set_node_labels(node_wts)
        input_graph.savefiles(mews)
        input_graph.run_solver(mews, max_num_nodes=max_num_nodes)
    else:
        input_graph.read_from_file(mews, input_graph=True)
        if node_wts == 'const':
            input_graph.set_node_labels(node_wts, const_val=val)
        else:
            input_graph.set_node_labels(node_wts)

    input_graph.visualize_graph(mews, True, graph_options(color='red', node_size=5,
                                                          line_color='white', linewidhts=0.1, width=1))

    output_graph = BrainGraph(edge, feature,node_wts, target, max_num_nodes, val, thresh)
    reduced_feature_indices = output_graph.read_from_file(mews, input_graph=False)
    output_graph.visualize_graph(mews, False, graph_options(color='red', node_size=5,
                                                          line_color='white', linewidhts=0.1, width=1))

    print('The number of nodes in the Input graph', len(input_graph.nodes))
    print('The number of edges in the Input graph', len(input_graph.edges))
    print('The number of nodes in the output graph', len(output_graph.nodes))
    print('The number of edges in the output graph', len(output_graph.edges))
    return X_train, X_test, reduced_feature_indices, arr
# In[ ]:


par = params.iloc[0, :]
target = par['Target']
solver_edge = par['Edge']
edge = par['Feature_type']
val = par['Node_weights']
thresh = par['ROI_strl_thresh']
solver_node_wts = 'const'
max_num_nodes = par['Output_Graph_nodes']
choice = 'test throw median'
feature_selection = 'solver'
classifier = 'SVC'
refit_metric = 'balanced_accuracy'

metrics = ['balanced_accuracy', 'accuracy', 'f1_weighted', 'roc_auc_ovr_weighted']

strls_num = X_train.iloc[:, tri:2 * tri]

y_train_l = y_train[mapping[target]]
y_test_l = y_test[mapping[target]]
print('-' * 100)

X_train_l, X_test_l = edge_filtering(edge, X_train, X_test)
assert len(X_train) == len(y_train)
med = int(y_train_l.median())  # the median is tried based on the training set
y_train_l = pd.qcut(y_train_l, 5, labels=False, retbins=True)[0]

y_train_l = y_train_l[y_train_l != 2]
y_train_l = y_train_l // 3
X_train_l = X_train_l.loc[y_train_l.index]
assert len(X_train_l) == len(y_train_l)

if choice == 'test throw median':

    y_test_l = y_test_l[abs(y_test_l - med) > 1]
    y_test_l = y_test_l >= med
    X_test_l = X_test_l.loc[list(set(y_test_l.index))]

    assert len(X_test_l) == len(y_test_l)

elif choice == 'keep median':
    y_test_l = y_test_l >= med

if feature_selection == 'solver':
    print(classifier, feature_selection, choice, refit_metric, target, solver_edge, edge,
          solver_node_wts)
    strls_num_l = strls_num.loc[X_train_l.index, :]
    strls_num_l = strls_num.mean(axis=0, skipna=True)
    X_train_l, X_test_l, edge_wts, arr = solver_viz(X_train_l, X_test_l, y_train_l, strls_num_l, solver_edge, thresh,
                                                    val, max_num_nodes,node_wts=solver_node_wts, target=target, edge=edge)

    if len(edge_wts) != 0:
        train_res, test_res = cross_validation(classifier, X_train_l, y_train_l, X_test_l, y_test_l,
                                               metrics, refit_metric)

# In[ ]:


train_res

# In[ ]:


test_res

# In[30]:


import glob
import os
import tensorflow as tf
from tensorflow.keras.regularizers import l2

tf.keras.backend.clear_session()

# In[31]:


params = pd.read_csv('/outputs/csvs/filtered.csv', index_col=None)
#%%
def solver_viz(X_train, X_test, y_train, strls_num, feature, thresh, val, max_num_nodes, node_wts=None,
           target=None, edge=None):
    X_train, X_test, arr = process_raw(X_train, X_test, y_train, feature)
    arr = arr.abs()
    arr = pd.DataFrame(arr, index=arr.index)  # scale the array according to the index
    input_graph = BrainGraph(edge, feature, node_wts, target, max_num_nodes, val, thresh)
    if not os.path.exists(f'{mews}/outputs/edges/{input_graph.filename}.out')\
            and not os.path.exists(f'{mews}/outputs/nodes/{input_graph.filename}.out'):
        print ('the solver is being called since the reduced file does not exist')
        input_graph.make_graph(arr, strls_num, thresh)
        if node_wts == 'const':
            input_graph.set_node_labels(node_wts, const_val=val)
        else:
            input_graph.set_node_labels(node_wts)
        input_graph.savefiles(mews)
        input_graph.run_solver(mews, max_num_nodes=max_num_nodes)
    else:
        input_graph.read_from_file(mews, input_graph=True)
        if node_wts == 'const':
            input_graph.set_node_labels(node_wts, const_val=val)
        else:
            input_graph.set_node_labels(node_wts)

    input_graph.visualize_graph(mews, True, graph_options(color='red', node_size=5,
                                                          line_color='white', linewidhts=0.1, width=1))

    output_graph = BrainGraph(edge, feature,node_wts, target, max_num_nodes, val, thresh)
    input_graph.visualize_graph(mews, True, graph_options(color='red', node_size=5,
                                                          line_color='white', linewidhts=0.1, width=1))
    reduced_feature_indices = output_graph.read_from_file(mews, input_graph=False)
    print('The number of nodes in the Input graph', len(input_graph.nodes))
    print('The number of edges in the Input graph', len(input_graph.edges))
    print('The number of nodes in the output graph', len(output_graph.nodes))
    print('The number of edges in the output graph', len(output_graph.edges))
    return X_train, X_test, reduced_feature_indices, arr
# In[42]:


par = params.iloc[0, :]
target = par['Target']
solver_edge = par['Edge']
edge = par['Feature_type']
val = par['Node_weights']
thresh = par['ROI_strl_thresh']
solver_node_wts = 'const'
max_num_nodes = par['Output_Graph_nodes']
choice = 'test throw median'
feature_selection = 'solver'
classifier = 'SVC'
refit_metric = 'balanced_accuracy'

metrics = ['balanced_accuracy', 'accuracy', 'f1_weighted', 'roc_auc_ovr_weighted']

strls_num = X_train.iloc[:, tri:2 * tri]

y_train_l = y_train[mapping[target]]
y_test_l = y_test[mapping[target]]
print('-' * 100)

X_train_l, X_test_l = edge_filtering(edge, X_train, X_test)
assert len(X_train) == len(y_train)
med = int(y_train_l.median())  # the median is tried based on the training set
y_train_l = pd.qcut(y_train_l, 5, labels=False, retbins=True)[0]

y_train_l = y_train_l[y_train_l != 2]
y_train_l = y_train_l // 3
X_train_l = X_train_l.loc[y_train_l.index]
assert len(X_train_l) == len(y_train_l)

if choice == 'test throw median':

    y_test_l = y_test_l[abs(y_test_l - med) > 1]
    y_test_l = y_test_l >= med
    X_test_l = X_test_l.loc[list(set(y_test_l.index))]

    assert len(X_test_l) == len(y_test_l)

elif choice == 'keep median':
    y_test_l = y_test_l >= med

if feature_selection == 'solver':
    print(classifier, feature_selection, choice, refit_metric, target, solver_edge, edge,
          solver_node_wts)
    strls_num_l = strls_num.loc[X_train_l.index, :]
    strls_num_l = strls_num.mean(axis=0, skipna=True)
    X_train_l, X_test_l, edge_wts, arr = solver_viz(X_train_l, X_test_l, y_train_l, strls_num_l, solver_edge, thresh,
                                                    val,
                                                    max_num_nodes,
                                                    node_wts=solver_node_wts, target=target, edge=edge)

    if len(edge_wts) != 0:
        train_res, test_res = cross_validation(classifier, X_train_l, y_train_l, X_test_l, y_test_l,
                                               metrics, refit_metric)

# In[43]:


train_res

# In[44]:


test_res

# In[9]:


import glob
import os
import tensorflow as tf
from tensorflow.keras.regularizers import l2

tf.keras.backend.clear_session()

# In[10]:


X_train = X_train.iloc[:, :tri]
X_test = X_test.iloc[:, :tri]

# In[11]:


med = y_train['NEOFAC_A'].median()
y_train = y_train['NEOFAC_A'] >= med
y_test = y_test['NEOFAC_A'] >= med

# In[12]:


ks = [str(i) for i in range(3570)]

# In[13]:


tf.keras.backend.set_floatx('float64')
inputs = {key: tf.keras.layers.Input(shape=(), name=key) for key in ks}
x = tf.stack(list(inputs.values()), axis=-1)

# In[14]:


x = tf.keras.layers.Dense(50, activation='tanh', bias_regularizer=l2(0.01))(x)
x = tf.keras.layers.Dense(100, activation='tanh', bias_regularizer=l2(0.01))(x)
x = tf.keras.layers.Dense(100, activation='tanh', bias_regularizer=l2(0.01))(x)
x = tf.keras.layers.Dense(50, activation='tanh', bias_regularizer=l2(0.01))(x)
output = tf.keras.layers.Dense(1, activation='tanh')(x)

# In[15]:


model_func = tf.keras.Model(inputs=inputs, outputs=output)

model_func.compile(optimizer='adam',
                   loss='binary_crossentropy',
                   metrics=['accuracy'])

train_set = tf.data.Dataset.from_tensor_slices((X_train.to_dict('list'), y_train.values)).batch(20)
test_set = tf.data.Dataset.from_tensor_slices((X_test.to_dict('list'), y_test.values)).batch(10)

# In[16]:


model_func.fit(train_set, epochs=50)
model_func.evaluate(test_set)

# In[17]:


X_train.index = range(len(X_train))
y_train.index = range(len(y_train))

# In[18]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import StratifiedKFold
import numpy as np


def run_model(optimizer, X_train, y_train):
    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)
    #
    # define 5-fold cross validation test harness

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cvscores = []
    for train, validation in kfold.split(X_train, y_train):
        # create model
        model = Sequential()
        model.add(Dense(10, input_dim=3570, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        # Fit the model
        model.fit(np.array(X_train.loc[train]), np.array(y_train.loc[train]), epochs=150, batch_size=10, verbose=0)
        # evaluate the model
        scores = model.evaluate(np.array(X_train.loc[validation]), np.array(y_train.loc[validation]), verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
        cvscores.append(scores[1] * 100)
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
    return model


def evaluate(model, X_test, y_test):
    return model.evaluate(np.array(X_test), np.array(y_test))


# In[33]:


model = run_model('adam', X_train, y_train)
evaluate(model, X_test, y_test)

# In[ ]: