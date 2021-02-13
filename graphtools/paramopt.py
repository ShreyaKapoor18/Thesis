"""
This script is meant for parameter search in cross validation and option exploration.
"""
from sklearn.svm import SVC
from sklearn.utils.fixes import loguniform
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression


def get_distributions(classifier, search, train_params):
    """
    @param classifier: the classifier that we want to train
    @param search: if we are searching for the new parameters or have predefined parameters
    @param train_params: the training parameters
    @return: clf, distributions the classifier with the distributions on which CrossValidation will be run
    """
    if search:
        if classifier == 'SVC':
            clf = SVC(probability=True)
            distributions = {'C': loguniform(1e-4, 1e3),
                             'gamma': loguniform(1e-4, 1e-2),
                             'kernel': ['rbf', 'poly', 'sigmoid', 'linear'],
                             'class_weight': ['balanced', None]}

        elif classifier == 'RF':
            clf = RandomForestClassifier()
            distributions = {'bootstrap': [True, False],
                             'max_depth': [10, 30, 40, 50, 60],
                             'max_features': ['auto', 'sqrt'],
                             'min_samples_leaf': [1, 2, 4],
                             'min_samples_split': [2, 5, 10],
                             }

        elif classifier == 'GB':
            clf = GradientBoostingClassifier()
            distributions = {  # 'loss': ['deviance', 'exponential']
                'learning_rate': [0.8, 0.9, 1],
                'tol': [0.01, 0.1],
                'min_samples_leaf': [1, 2, 4],
                'min_samples_split': [2, 5, 10],
                # takes too long to converge if tolerance not specificied
            }
        elif classifier=='linear_reg':
           clf = LinearRegression()
           distributions = {
               'fit_intercept': ['True', 'False']
           }

        #multiclass cannot use loss exponential
        else:
            clf = MLPClassifier()
            distributions = {'hidden_layer_sizes': [(50, 100, 100, 50), (50, 100, 50)],
                             'activation': ['tanh', 'relu'],
                             'solver': ['sgd', 'adam'],
                             'alpha': [0.001, 0.05],
                             'learning_rate': ['adaptive']}  # doesn't converge even with maximum iterations!

        return clf, distributions
    else:
        if classifier == 'SVC':
            clf = SVC(probability=True, **train_params)
        elif classifier == 'RF':
            clf = RandomForestClassifier(**train_params)
        elif classifier == 'GB':
            clf = GradientBoostingClassifier(**train_params)
            # multiclass cannot use loss exponential
        elif classifier == 'linear_reg':
            clf = LinearRegression(**train_params)
        else:
            clf = MLPClassifier(**train_params)
        return clf


def graph_options(color, node_size, line_color, linewidhts, width):
    """
    @param color: the color we want to give the nodes
    @param node_size: the size of the nodes
    @param line_color: the color of the lines
    @param linewidhts: the width of the lines in the graph
    @param width: width
    """
    options = {
        'node_color': color,
        'node_size': node_size,
        'line_color': line_color,
        'linewidths': linewidhts,
        'width': width,
    }
    return options
