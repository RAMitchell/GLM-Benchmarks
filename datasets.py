import shutil
import urllib
import tarfile
import gzip
import numpy as np
import zipfile
import os
import sys
import pandas as pd
from scipy.sparse import vstack
from sklearn.externals.joblib import Memory
from sklearn import datasets

if sys.version_info[0] >= 3:
    from urllib.request import urlretrieve
else:
    from urllib import urlretrieve

mem = Memory("./mycache")

@mem.cache
def get_higgs():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz'
    filename = 'HIGGS.csv'
    if not os.path.isfile(filename):
        urlretrieve(url, filename + '.gz')
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
        urlretrieve(url, filename + '.zip')
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
        urlretrieve(url, filename)
        tar = tarfile.open(filename, "r:gz")
        tar.extractall()
        tar.close()

    num_files = 120
    files = ['url_svmlight/Day{}.svm'.format(day) for day in range(num_files)]
    data = datasets.load_svmlight_files(files)
    X = vstack(data[::2])
    y = np.concatenate(data[1::2])

    y[y < 0.0] = 0.0
    return X, y

