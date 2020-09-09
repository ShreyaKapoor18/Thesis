
from processing import *
from readfiles import *
from classification_refined import classify
import copy
from joblib import Parallel, delayed
import multiprocessing
import multiprocessing
from itertools import product
from functools import partial
from contextlib import contextmanager
#%%
@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()

# remember to automatically create the csv files beforehand
if __name__ == '__main__':
    # %%
    ''' Data computed for all 5 personality traits at once'''
    # labels for the computed subjects, data.index is the subject id
    num = 84  # number of nodes in the graph
    tri = int(num * (num + 1) * 0.5)  # we want only the upper diagonal parts since everything below diagonal is 0
    labels = ['NEOFAC_A', 'NEOFAC_O', 'NEOFAC_C', 'NEOFAC_N', 'NEOFAC_E']
    edge_names = ['mean_FA', 'mean_strl', 'num_streamlines']
    big5 = ['Agreeableness', 'Openness', 'Conscientiousness', 'Neuroticism',
            'Extraversion']
    mapping = {k: v for k, v in zip(big5, labels)}
    mat = np.triu_indices(84)
    mews = '/home/skapoor/Thesis/gmwcs-solver'
    metrics = ['balanced_accuracy', 'accuracy', 'f1_weighted', 'roc_auc_ovr_weighted']
    # note: right now the matrix whole is not scaled, for computing the fscores and correlation coeff it has to be so.
    y_train = computed_subjects()
    X_train = generate_combined_matrix(tri, list(y_train.index))  # need to check indices till here then convert to numpy array
    y_test = test_subjects()
    X_test = generate_test_data(tri, y_test.index)

    classifiers = ['SVC', 'RF','MLP']
    cols_base = ['Classifier', 'Target', 'Choice', 'Edge', 'Feature Selection', 'Type of feature', 'Percentage',
                 'Refit Metric']
    cols_solver = copy.deepcopy(cols_base)
    cols_solver.extend(['Node_weights', 'Factor', 'Subtracted_value', 'Num edges', '% Positive edges'])
    cols_base.extend(metrics)
    cols_solver.extend(metrics)
    l1 = [X_train, X_test, y_train, y_test]
    feature_selections = ['baseline', 'solver']
    choices = ['test throw median', 'keep median']
    s_params = pd.read_csv('/home/skapoor/Thesis/gmwcs-solver/outputs/solver/filtered.csv')
    s1 = pd.DataFrame([], columns=cols_solver)
    s1.to_csv('solver.csv')
    b1 = pd.DataFrame([], columns=cols_solver)
    b1.to_csv('base.csv')

    prod = list(product(classifiers,[s_params.iloc[:,:6]], feature_selections, choices, metrics))
    #print(len(prod)*len(s_params)*2, 'is the number of use cases the pipeline will run for')
    for classifier, params, feature_selection, choice, refit_metric in prod:
        resb, ress = classify(l1,classifier, params, feature_selection, choice, refit_metric)
        resb = pd.DataFrame(resb, columns=cols_base)
        ress = pd.DataFrame(ress, columns=cols_solver)
        ress.to_csv('solver.csv', mode='a', header=False)
        resb.to_csv('base.csv', mode='a', header=False)
        #results_base.extend(resb)
        #results_solver.extend(ress)


'''
Here we will need to take the threshold and degree as hyperparameters, change them and compute the result accordingly
maybe make the dictionary like dict['degree'] = val when the val is in a loop. We shall put all the options differently
'''
# %%
'''
To do:
def pipeline_summary():
    # fscores based on the label
    # pearson correlation based on the target variable
    # classification for the target variable
    # input to the solver
    # output from the solver
    # classification on the basis of the solver output
'''
