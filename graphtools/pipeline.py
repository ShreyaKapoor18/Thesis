
from processing import *
from readfiles import *
from classification_refined import classify
import copy
from itertools import product
from subgraphclass import make_solver_summary
from solver_decision import filter

# remember to automatically create the csv files beforehand
if __name__ == '__main__':
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
    #make_solver_summary(mapping, y_train, big5, mews, X_train, tri)
    filter()
    y_test = test_subjects()
    X_test = generate_test_data(tri, y_test.index)

    classifiers = ['SVC', 'RF','MLP']
    cols_base = ['Classifier', 'Target', 'Choice', 'Edge', 'Feature Selection', 'Type of feature', 'Percentage',
                 'Refit Metric']
    cols_solver = copy.deepcopy(cols_base)
    cols_solver.extend(['Node_weights', 'Factor', 'Subtracted_value', 'Num edges', '% Positive edges'])
    cols_base.extend([f'train_{metric}'for metric in metrics])
    cols_solver.extend([f'train_{metric}'for metric in metrics])
    cols_base.extend([f'test_{metric}'for metric in metrics])
    cols_solver.extend([f'test_{metric}'for metric in metrics])
    l1 = [X_train, X_test, y_train, y_test]
    feature_selections = ['baseline', 'solver']
    choices = ['test throw median', 'keep median']
    s_params = pd.read_csv('/home/skapoor/Thesis/graphtools/outputs/csvs/filtered.csv', index_col=None)#now it will be the one in graphtools outputs/csvs
    if len(s_params.columns)>13:
        s_params = s_params.iloc[:,1:]
    results_base = []
    results_solver=[]

    prod = product(classifiers,[s_params.iloc[:,:6]], feature_selections, choices, metrics)
    #print(len(prod)*len(s_params)*2, 'is the number of use cases the pipeline will run for')
    for classifier, params, feature_selection, choice, refit_metric in prod:
            resb, ress = classify(l1,classifier, params, feature_selection, choice, refit_metric)
            results_solver.extend(ress)
            results_base.extend(resb)
        #print(results_base)
        #results_base.extend(resb)
        #results_solver.extend(ress)
    results_base = pd.DataFrame(results_base, columns=cols_base)
    results_base = results_base.round(3)
    results_solver = pd.DataFrame(results_solver, columns=cols_solver)
    results_solver = results_solver.round(3)
    results_solver.to_csv('outputs/csvs/solver.csv')
    results_base.to_csv('outputs/csvs/base.csv')

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
