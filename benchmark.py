import datasets
import glmnet
import numpy as np
import pandas as pd
import time
import xgboost as xgb
import sys
from scipy import sparse
from sklearn import metrics, linear_model, preprocessing

num_rounds = 1000
early_stopping_rounds = 5


def is_float(s):
    try:
        float(s)
        return 1
    except ValueError:
        return 0


def get_density(X):
    if sparse.issparse(X):
        return X.count_nonzero() / float(X.shape[0] * X.shape[1])
    return np.count_nonzero(X) / float(X.shape[0] * X.shape[1])


def xgb_get_weights(bst):
    return [float(s) for s in bst.get_dump()[0].split() if is_float(s)][1:]


def run_xgboost_regression(df, X, y, param, dataset, reg_alpha, reg_lambda, metric, density):
    tmp = time.time()
    dtrain = xgb.DMatrix(X, label=y, nthread=-1)
    bst = xgb.train(param, dtrain, num_rounds, [(dtrain, 'train')], early_stopping_rounds=early_stopping_rounds,
                    verbose_eval=False)
    xgb_time = time.time() - tmp
    xgb_score = np.sqrt(metrics.mean_squared_error(y, bst.predict(dtrain)))
    xgb_zero = X.shape[1] - np.count_nonzero(xgb_get_weights(bst))
    xgb_iterations = bst.best_iteration + early_stopping_rounds
    df.loc[len(df)] = [dataset, X.shape[1], X.shape[0], density, 'Regression', reg_alpha, reg_lambda, 'XGBoost', metric,
                       xgb_score,
                       xgb_time, xgb_iterations, xgb_zero]


def run_regression(df, dataset, X, y, reg_alpha, reg_lambda, standardize=True):
    if standardize:
        X = preprocessing.StandardScaler().fit_transform(X)
        y = preprocessing.scale(y)
    metric = 'RMSE'
    density = get_density(X)
    param = {'booster': 'gblinear', 'updater': 'coord_descent', 'reg_alpha': reg_alpha, 'reg_lambda': reg_lambda,
             'debug_verbose': 1}
    run_xgboost_regression(df, X, y, param, dataset, reg_alpha, reg_lambda, metric, density)

    tmp = time.time()
    enet = linear_model.ElasticNet(alpha=reg_alpha + reg_lambda, l1_ratio=reg_alpha / (reg_alpha + reg_lambda))
    enet.fit(X, y)
    enet_time = time.time() - tmp
    enet_score = np.sqrt(metrics.mean_squared_error(y, enet.predict(X)))
    enet_zero = X.shape[1] - np.count_nonzero(enet.coef_)
    df.loc[len(df)] = [dataset, X.shape[1], X.shape[0], density, 'Regression', reg_alpha, reg_lambda, 'Sklearn', metric,
                       enet_score,
                       enet_time, enet.n_iter_, enet_zero]

    tmp = time.time()
    glm = glmnet.ElasticNet(alpha=reg_alpha / (reg_alpha + reg_lambda), lambda_path=[reg_alpha + reg_lambda])
    glm.fit(X, y)
    glmnet_time = time.time() - tmp
    glmnet_score = np.sqrt(metrics.mean_squared_error(y, glm.predict(X, lamb=[reg_alpha + reg_lambda])))
    glmnet_zero = X.shape[1] - np.count_nonzero(glm.coef_)
    df.loc[len(df)] = [dataset, X.shape[1], X.shape[0], density, 'Regression', reg_alpha, reg_lambda, 'Glmnet', metric,
                       glmnet_score,
                       glmnet_time, '-', glmnet_zero]


def run_xgboost_classification(df, X, y, param, dataset, reg_alpha, reg_lambda, metric, density):
    dtrain = xgb.DMatrix(X, label=y, nthread=-1)
    tmp = time.time()
    bst = xgb.train(param, dtrain, num_rounds, [(dtrain, 'train')], early_stopping_rounds=early_stopping_rounds,
                    verbose_eval=True)
    xgb_time = time.time() - tmp
    xgb_score = metrics.accuracy_score(y, np.round(bst.predict(dtrain)))
    xgb_zero = X.shape[1] - np.count_nonzero(xgb_get_weights(bst))
    df.loc[len(df)] = [dataset, X.shape[1], X.shape[0], density, 'Classification', reg_alpha, reg_lambda, 'XGBoost',
                       metric,
                       xgb_score,
                       xgb_time, bst.best_iteration + early_stopping_rounds, xgb_zero]


def run_sklearn_classification(df, X, y, algo, dataset, reg_alpha, reg_lambda, metric, density):
    tmp = time.time()
    C = None
    penalty = None
    if reg_alpha>0 and reg_lambda>0:
        raise ValueError("Cannot perform l1 & l2 regularisation using sklearn.")
    if reg_alpha>0:
        penalty = 'l1'
        C = 1.0 / (reg_alpha*X.shape[0])
    if reg_lambda>0:
        penalty = 'l2'
        C = 1.0 / (reg_lambda*X.shape[0])
    sklearn = linear_model.LogisticRegression(C=C, solver=algo, penalty = penalty,intercept_scaling = 500)
    sklearn.fit(X, y)
    sklearn_time = time.time() - tmp
    sklearn_score = metrics.accuracy_score(y, sklearn.predict(X))
    sklearn_zero = X.shape[1] - np.count_nonzero(sklearn.coef_)
    df.loc[len(df)] = [dataset, X.shape[1], X.shape[0], density, 'Classification', reg_alpha, reg_lambda,
                       'Sklearn ({})'.format(algo),
                       metric, sklearn_score,
                       sklearn_time, sklearn.n_iter_, sklearn_zero]


def run_glmnet_classification(df, X, y, dataset, reg_alpha, reg_lambda, metric, density):
    tmp = time.time()
    glm = glmnet.LogitNet(alpha=reg_alpha / (reg_alpha + reg_lambda), lambda_path=[reg_alpha + reg_lambda])
    glm.fit(X, y)
    glmnet_time = time.time() - tmp
    glmnet_score = metrics.accuracy_score(y, glm.predict(X))
    glmnet_zero = X.shape[1] - np.count_nonzero(glm.coef_)

    df.loc[len(df)] = [dataset, X.shape[1], X.shape[0], density, 'Classification', reg_alpha, reg_lambda, 'Glmnet',
                       metric,
                       glmnet_score,
                       glmnet_time, '-', glmnet_zero]


def run_classification(df, dataset, X, y, reg_alpha, reg_lambda, standardize=True, solvers=None):
    if solvers is None:
        solvers = ['XGBoost', 'Sklearn (liblinear)', 'Sklearn (sag)', 'Sklearn (lbfgs)', 'Glmnet']
    if standardize:
        X = preprocessing.StandardScaler().fit_transform(X)
    metric = 'Accuracy score'
    density = get_density(X)
    param = {'booster': 'gblinear', 'objective': 'binary:logistic', 'updater': 'coord_descent', 'reg_alpha': reg_alpha,
             'reg_lambda': reg_lambda, 'debug_verbose': 1}
    if 'XGBoost' in solvers:
        run_xgboost_classification(df, X, y, param, dataset, reg_alpha, reg_lambda, metric, density)
    if 'Sklearn (liblinear)' in solvers:
        run_sklearn_classification(df, X, y, 'liblinear', dataset, reg_alpha, reg_lambda, metric, density)
    if 'Sklearn (sag)' in solvers:
        run_sklearn_classification(df, X, y, 'sag', dataset, reg_alpha, reg_lambda, metric, density)
    if 'Sklearn (lbfgs)' in solvers:
        run_sklearn_classification(df, X, y, 'lbfgs', dataset, reg_alpha, reg_lambda, metric, density)
    if 'Glmnet' in solvers:
        run_glmnet_classification(df, X, y, dataset, reg_alpha, reg_lambda, metric, density)


num_rows = None
if len(sys.argv) > 1:
    num_rows = 1000

df = pd.DataFrame([], columns=['Dataset', 'Num features', 'Num rows', 'Density', 'Type', 'L1 penalty', 'L2 penalty',
                               'Solver'
    , 'Metric', 'Accuracy', 'Time',
                               'Iterations', 'Zero Coefficients'])

X, y = datasets.get_year(num_rows)
run_regression(df, 'YearPredictionMSD', X, y, 0.05, 0.1)

X, y = datasets.get_higgs(num_rows)

run_classification(df, 'Higgs', X, y, 0.0, 0.1)

solvers = ['XGBoost', 'Sklearn (liblinear)','Glmnet']
run_classification(df, 'Higgs', X, y, 0.02, 0.0,True,solvers)

X, y = datasets.get_url(num_rows)

solvers = ['XGBoost', 'Sklearn (liblinear)', 'Sklearn (sag)', 'Sklearn (lbfgs)']
# solvers = ['XGBoost']
run_classification(df, 'URL Reputation', X, y, 0.0, 0.000001, False, solvers)

print(df.to_string())
df.to_csv('result.csv')
