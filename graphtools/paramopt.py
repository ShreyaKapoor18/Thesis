from sklearn.svm import SVC
from sklearn.utils.fixes import loguniform
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


def get_distributions(classifier):
    if classifier == 'SVC':
        clf = SVC(probability=True)
        distributions = {'C': loguniform(1e0, 1e3),
                         'gamma': loguniform(1e-4, 1e-2),
                         'kernel': ['rbf', 'poly', 'sigmoid', 'linear'],
                         'class_weight': ['balanced', None]}

    elif classifier == 'RF':
        clf = RandomForestClassifier()
        distributions = {'bootstrap': [True, False],
                         'max_depth': [10, 20, 30, 40, 50, 60, 70],
                         'max_features': ['auto', 'sqrt'],
                         'min_samples_leaf': [1, 2, 4],
                         'min_samples_split': [2, 5, 10],
                         'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400]}

    elif classifier == 'GB':
        clf = GradientBoostingClassifier()
        distributions = {  # 'loss': ['deviance', 'exponential']
            'learning_rate': [0.8, 0.9, 1],
            'tol': [0.01, 0.1],
            'min_samples_leaf': [1, 2, 4],
            'min_samples_split': [2, 5, 10],
            'n_estimators': [200, 400]  # takes too long to converge if tolerance not specificied
        }
        # multiclass cannot use losss exponential
    elif classifier == 'MLP':
        clf = MLPClassifier()
        distributions = {'hidden_layer_sizes': [(50, 100, 100, 50), (50, 100, 50)],
                         'activation': ['tanh', 'relu'],
                         'solver': ['sgd', 'adam'],
                         'alpha': [0.001, 0.05],
                         'learning_rate': ['adaptive']}  # doesn't converge even with maximum iterations!

    return clf, distributions
