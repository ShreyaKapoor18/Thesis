from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
import numpy as np
import pandas as pd
from readfiles import computed_subjects
from processing import generate_combined_matrix, hist_fscore, hist_correlation
from sklearn.pipeline import make_pipeline
import time
# %%
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate, cross_val_predict


def plot_regression_results(ax, y_true, y_pred, title, scores, elapsed_time):
    """Scatter plot of the predicted vs true targets."""
    ax.plot([y_true.min(), y_true.max()],
            [y_true.min(), y_true.max()],
            '--r', linewidth=2)
    ax.scatter(y_true, y_pred, alpha=0.2)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    ax.set_xlim([y_true.min(), y_true.max()])
    ax.set_ylim([y_true.min(), y_true.max()])
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                          edgecolor='none', linewidth=0)
    ax.legend([extra], [scores], loc='upper left')
    title = title + '\n Evaluation in {:.2f} seconds'.format(elapsed_time)
    ax.set_title(title)


# %%
def dict_regressor(whole, metrics, labels, big5, data, new_corr):
    """
    :param metrics: the name of the metrics we want to calculate
    :param labels: the big5 personality labels
    :param data: the file with contains the labels for all features of all subjects
    :param new_fscores: flattened array of f scores: num_subjects x num edges
    :return: metric_scores: the values to be calculated using permitted keywords
    """

    for i in range(5):  # different labels

        for per in [5, 10]:
            val = np.percentile(new_corr[i], 100 - per)
            index = np.where(new_corr[i] >= val)
            print('Number of features', len(index[0]))
            y = data[labels[i]]
            X = whole[:, index[0]]
            lasso = LassoCV()
            rf = RandomForestRegressor()
            gradient = HistGradientBoostingRegressor(max_iter=100000)

            estimators = [('Random Forest', rf), ('Lasso', lasso), ('Gradient booster', gradient)]
            stacking_regressor = StackingRegressor(estimators=estimators, final_estimator=RidgeCV())

            fig, axs = plt.subplots(2, 2, figsize=(15, 15))
            axs = np.ravel(axs)

            for ax, (name, est) in zip(axs, estimators + [('Stacking Regressor',
                                                           stacking_regressor)]):
                start_time = time.time()
                score = cross_validate(est, X, y,
                                       scoring=metrics,
                                       n_jobs=-1, verbose=0)
                elapsed_time = time.time() - start_time

                y_pred = cross_val_predict(est, X, y, n_jobs=-1, verbose=0)

                plot_regression_results(
                    ax, y, y_pred,
                    name,
                    (r'$R^2={:.2f} \pm {:.2f}$' + '\n' + r'$MAE={:.2f} \pm {:.2f}$')
                        .format(np.mean(score['test_r2']),
                                np.std(score['test_r2']),
                                -np.mean(score['test_neg_mean_absolute_error']),
                                np.std(score['test_neg_mean_absolute_error'])),
                    elapsed_time)

            plt.suptitle(f'Single predictors versus stacked predictors for label {big5[i]}')
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            plt.savefig(f'outputs/{big5[i]}_{per}.png')


# %%
def make_csv(dict_score, filename):
    cv = pd.concat({
        k: pd.DataFrame.from_dict(v, 'index') for k, v in dict_score.items()
    },
        axis=0)
    cv.to_csv(filename)


# %%
data = computed_subjects()  # labels for the computed subjects
data.reset_index(inplace=True)
num = 84  # number of nodes in the graph
tri = int(num * (num + 1) * 0.5)  # we want only the upper diagonal parts since everything below diagonal is 0
whole = generate_combined_matrix(tri)
# The labels i.e. the ones from unrestricted_files!
# %%
labels = ['NEOFAC_A', 'NEOFAC_O', 'NEOFAC_C', 'NEOFAC_N', 'NEOFAC_E']
edge_names = ['mean_FA', 'mean strl', 'num streamlines']
big5 = ['Agreeableness', 'Openness', 'Conscientiousness', 'Neuroticism',
        'Extraversion']
# %%
corr = hist_correlation(data, whole, labels, edge_names, big5, tri)
new_corr = np.reshape(corr, (corr.shape[0], corr.shape[1]*corr.shape[2]))
# %%
# without taking the edge type into consideration
cv_metrics = ['r2', 'neg_mean_absolute_error']
dict_regressor(whole, cv_metrics, labels, big5, data, new_corr)
