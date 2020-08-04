from nested_cv import NestedCV
from sklearn.ensemble import RandomForestRegressor
from processing import *
from readfiles import computed_subjects
# %%
''' Data computed for all 5 personality traits at once'''
data = computed_subjects()  # labels for the computed subjects, data.index is the subject id
num = 84  # number of nodes in the graph
tri = int(num * (num + 1) * 0.5)  # we want only the upper diagonal parts since everything below diagonal is 0
X = generate_combined_matrix(tri, list(data.index))  # need to check indices till here then convert to numpy array
assert list(X.index) == list(data.index)
y = data['NEOFAC_A']
y = y>= y.median()

models_to_run = [RandomForestRegressor()]
models_param_grid = [
                    { # 1st param grid, corresponding to RandomForestRegressor
                            'max_depth': [3, None],
                            'n_estimators': [100,200,300,400,500,600,700,800,900,1000],
                            'max_features' : [50,100,150,200]
                    },
                    { # 2nd param grid, corresponding to XGBRegressor
                            'learning_rate': [0.05],
                            'colsample_bytree': np.linspace(0.3, 0.5),
                            'n_estimators': [100,200,300,400,500,600,700,800,900,1000],
                            'reg_alpha' : (1,1.2),
                            'reg_lambda' : (1,1.2,1.4)
                    },
                    { # 3rd param grid, corresponding to LGBMRegressor
                            'learning_rate': [0.05],
                            'n_estimators': [100,200,300,400,500,600,700,800,900,1000],
                            'reg_alpha' : (1,1.2),
                            'reg_lambda' : (1,1.2,1.4)
                    }
                    ]
for i, model in enumerate(models_to_run):
    nested_CV_search = NestedCV(model=model, params_grid=models_param_grid[i],
                                outer_kfolds=5, inner_kfolds=5,
                                cv_options={'sqrt_of_score': True, 'randomized_search_iter': 30})

    nested_CV_search.fit(X=X, y=y)
    model_param_grid = nested_CV_search.best_params

    print(np.mean(nested_CV_search.outer_scores))
    print(nested_CV_search.best_inner_params_list)
