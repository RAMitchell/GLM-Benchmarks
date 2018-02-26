import glmnet
import gzip
import numpy as np
import os
import pandas as pd
import shutil
import tarfile
import time
import urllib
import xgboost as xgb
import zipfile
from sklearn import datasets, metrics, linear_model, preprocessing
from sklearn.externals.joblib import Memory

mem = Memory("./mycache")

num_rounds = 1000
early_stopping_rounds = 5


@mem.cache
def get_higgs():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz'
    filename = 'HIGGS.csv'
    if not os.path.isfile(filename):
        urllib.urlretrieve(url, filename + '.gz')
        with gzip.open(filename + '.gz', 'rb') as f_in:
            with open(filename, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    higgs = pd.read_csv(filename)
    X = higgs.iloc[:, 1:].values
    y = higgs.iloc[:, 0].values

    return X, y


@mem.cache
def get_year():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00203/YearPredictionMSD.txt.zip'
    filename = 'YearPredictionMSD.txt'
    if not os.path.isfile(filename):
        urllib.urlretrieve(url, filename + '.zip')
        zip_ref = zipfile.ZipFile(filename + '.zip', 'r')
        zip_ref.extractall()
        zip_ref.close()

    year = pd.read_csv('YearPredictionMSD.txt', header=None)
    X = year.iloc[:, 1:].values
    y = year.iloc[:, 0].values

    return X, y


@mem.cache
def get_url():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/url/url_svmlight.tar.gz'
    filename = 'url_svmlight.tar.gz'
    if not os.path.isfile(filename):
        urllib.urlretrieve(url)
    tar = tarfile.open(filename, "r:gz")
    tar.extractall()
    tar.close()

    data = datasets.load_svmlight_file('url_svmlight/Day0.svm')
    X = data[0]

    y = data[1]
    y[y < 0.0] = 0.0
    return X, y


def is_float(s):
    try:
        float(s)
        return 1
    except ValueError:
        return 0


def xgb_get_weights(bst):
    return [float(s) for s in bst.get_dump()[0].split() if is_float(s)][1:]

def run_xgboost_regression(df,X, y, param, dataset,reg_alpha,reg_lambda):
    tmp = time.time()
    dtrain = xgb.DMatrix(X, label=y, nthread=-1)
    bst = xgb.train(param, dtrain, num_rounds, [(dtrain, 'train')], early_stopping_rounds=early_stopping_rounds,
                    verbose_eval=False)
    xgb_time = time.time() - tmp
    xgb_score = np.sqrt(metrics.mean_squared_error(y, bst.predict(dtrain)))
    xgb_zero = X.shape[1] - np.count_nonzero(xgb_get_weights(bst))
    xgb_iterations = bst.best_iteration + early_stopping_rounds
    df.loc[len(df)] = [dataset, X.shape[1], X.shape[0], 'Regression', reg_alpha, reg_lambda, 'XGBoost', metric,
                       xgb_score,
                       xgb_time, xgb_iterations, xgb_zero]

def run_regression(df, dataset, X, y, reg_alpha, reg_lambda, standardize=True):
    if standardize:
        X = preprocessing.StandardScaler().fit_transform(X)
        y = preprocessing.scale(y)
    metric = 'RMSE'
    param = {'booster': 'gblinear', 'updater': 'coord_descent', 'reg_alpha': reg_alpha, 'reg_lambda': reg_lambda,
             'debug_verbose': 1}
    run_xgboost_regression(df,X, y, param, dataset,reg_alpha,reg_lambda)

    tmp = time.time()
    enet = linear_model.ElasticNet(alpha=reg_alpha + reg_lambda, l1_ratio=reg_alpha / (reg_alpha + reg_lambda))
    enet.fit(X, y)
    enet_time = time.time() - tmp
    enet_score = np.sqrt(metrics.mean_squared_error(y, enet.predict(X)))
    enet_zero = X.shape[1] - np.count_nonzero(enet.coef_)
    df.loc[len(df)] = [dataset, X.shape[1], X.shape[0], 'Regression', reg_alpha, reg_lambda, 'Sklearn', metric,
                       enet_score,
                       enet_time, enet.n_iter_, enet_zero]

    tmp = time.time()
    glm = glmnet.ElasticNet(alpha=reg_alpha / (reg_alpha + reg_lambda), lambda_path=[reg_alpha + reg_lambda])
    glm.fit(X, y)
    glmnet_time = time.time() - tmp
    glmnet_score = np.sqrt(metrics.mean_squared_error(y, glm.predict(X, lamb=[reg_alpha + reg_lambda])))
    glmnet_zero = X.shape[1] - np.count_nonzero(glm.coef_)
    df.loc[len(df)] = [dataset, X.shape[1], X.shape[0], 'Regression', reg_alpha, reg_lambda, 'Glmnet', metric,
                       glmnet_score,
                       glmnet_time, '-', glmnet_zero]
    print(df.to_string())


def run_xgboost_classification(df,X, y, param, dataset,reg_alpha,reg_lambda, metric):
    dtrain = xgb.DMatrix(X, label=y, nthread=-1)
    tmp = time.time()
    bst = xgb.train(param, dtrain, num_rounds, [(dtrain, 'train')], early_stopping_rounds=early_stopping_rounds,
                    verbose_eval=True)
    xgb_time = time.time() - tmp
    xgb_score = metrics.accuracy_score(y, np.round(bst.predict(dtrain)))
    xgb_zero = X.shape[1] - np.count_nonzero(xgb_get_weights(bst))
    df.loc[len(df)] = [dataset, X.shape[1], X.shape[0], 'Classification', reg_alpha, reg_lambda, 'XGBoost', metric,
                       xgb_score,
                       xgb_time, bst.best_iteration + early_stopping_rounds, xgb_zero]

def run_sklearn_classification(df,X, y, algo ,dataset,reg_alpha,reg_lambda, metric):
    tmp = time.time()
    C = 1.0 / (reg_lambda)
    sklearn = linear_model.LogisticRegression(C=C, solver = algo)
    sklearn.fit(X, y)
    sklearn_time = time.time() - tmp
    sklearn_score = metrics.accuracy_score(y, sklearn.predict(X))
    sklearn_zero = X.shape[1] - np.count_nonzero(sklearn.coef_)
    df.loc[len(df)] = [dataset, X.shape[1], X.shape[0], 'Classification', reg_alpha, reg_lambda, 'Sklearn ({})'.format(algo),
                       metric, sklearn_score,
                       sklearn_time, sklearn.n_iter_, sklearn_zero]

def run_glmnet_classification(df,X, y, dataset,reg_alpha,reg_lambda, metric):
    tmp = time.time()
    glm = glmnet.LogitNet(alpha=reg_alpha / (reg_alpha + reg_lambda), lambda_path=[reg_alpha + reg_lambda])
    glm.fit(X, y)
    glmnet_time = time.time() - tmp
    glmnet_score = metrics.accuracy_score(y, glm.predict(X))
    glmnet_zero = X.shape[1] - np.count_nonzero(glm.coef_)

    df.loc[len(df)] = [dataset, X.shape[1], X.shape[0], 'Classification', reg_alpha, reg_lambda, 'Glmnet', metric,
                       glmnet_score,
                       glmnet_time, '-', glmnet_zero]

def run_classification(df, dataset, X, y, reg_alpha, reg_lambda, standardize=True, solvers = None):
    if solvers is None:
        solvers = ['XGBoost','Sklearn (liblinear)','Sklearn (sag)','Sklearn (lbfgs)','Glmnet']
    if standardize:
        X = preprocessing.StandardScaler().fit_transform(X)
    metric = 'Accuracy score'
    param = {'booster': 'gblinear', 'objective': 'binary:logistic', 'updater': 'coord_descent', 'reg_alpha': reg_alpha,
             'reg_lambda': reg_lambda, 'debug_verbose': 1}
    if 'XGBoost' in solvers:
        run_xgboost_classification(df,X,y,param,dataset,reg_alpha,reg_lambda, metric)
    if 'Sklearn (liblinear)' in solvers:
        run_sklearn_classification(df,X,y,'liblinear',dataset,reg_alpha,reg_lambda, metric)
    if 'Sklearn (sag)' in solvers:
        run_sklearn_classification(df,X,y,'sag',dataset,reg_alpha,reg_lambda, metric)
    if 'Sklearn (lbfgs)' in solvers:
        run_sklearn_classification(df,X,y,'lbfgs',dataset,reg_alpha,reg_lambda, metric)
    if 'Glmnet' in solvers:
        run_glmnet_classification(df,X,y,dataset,reg_alpha,reg_lambda, metric)



df = pd.DataFrame([], columns=['Dataset', 'Num features', 'Num rows', 'Type', 'L1 penalty', 'L2 penalty', 'Solver'
    , 'Metric', 'Accuracy', 'Time',
                               'Iterations', 'Zero Coefficients'])
'''
X, y = get_year()
run_regression(df, 'YearPredictionMSD', X, y, 0.05, 0.1)
X,y = get_higgs()
run_classification(df, 'Higgs', X, y, 0.0, 0.1)
'''
X, y = get_url()
X = X[:,0:1000000]
solvers = ['XGBoost','Sklearn (liblinear)','Sklearn (sag)','Sklearn (lbfgs)']
#solvers = ['XGBoost']
run_classification(df, 'URL Reputation', X, y, 0.0, 0.000001, False,solvers)
print(df.to_string())
df.to_csv('result.csv')
