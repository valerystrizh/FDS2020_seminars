import dask
import dask.array as da
import dask.dataframe as dd
import dask_xgboost
import joblib
import json
import numpy as np
import os
import pandas as pd
import tarfile
import time
import urllib.request

from glob import glob
from dask import persist
from dask.distributed import Client, progress
from dask_ml.model_selection import train_test_split as train_test_split_dask
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.model_selection import train_test_split as train_test_split_normal

def flights(url):
    flights_raw = os.path.join(data_dir, 'nycflights.tar.gz')
    flightdir = os.path.join(data_dir, 'nycflights')
    jsondir = os.path.join(data_dir, 'flightjson')

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    if not os.path.exists(flights_raw):
        print("- Downloading NYC Flights dataset... ", end='', flush=True)
        #url = "https://storage.googleapis.com/dask-tutorial-data/nycflights.tar.gz"
        urllib.request.urlretrieve(url, flights_raw)
        print("done", flush=True)

    if not os.path.exists(flightdir):
        print("- Extracting flight data... ", end='', flush=True)
        tar_path = os.path.join('data', 'nycflights.tar.gz')
        with tarfile.open(tar_path, mode='r:gz') as flights:
            flights.extractall('data/')
        print("done", flush=True)

    if not os.path.exists(jsondir):
        print("- Creating json data... ", end='', flush=True)
        os.mkdir(jsondir)
        for path in glob(os.path.join('data', 'nycflights', '*.csv')):
            prefix = os.path.splitext(os.path.basename(path))[0]
            # Just take the first 10000 rows for the demo
            df = pd.read_csv(path).iloc[:10000]
            df.to_json(os.path.join('data', 'flightjson', prefix + '.json'),
                       orient='records', lines=True)
        print("done", flush=True)

    print("** Finished! **")

def get_df(columns, num_rows=100):
    df_list = [
        pd.read_csv(path)[:num_rows][columns]
        for path 
        in glob(os.path.join('data', 'nycflights', '*.csv'))
    ]

    return pd.concat([pd.DataFrame(df_i) for df_i in df_list])

def get_data(df, target, is_dask=False, chunksize=200):
    if is_dask:
        df = dd.from_pandas(df, chunksize=chunksize)
        target_s = df[target]
        del df[target]
        df, target_s = persist(df, target_s)
        progress(df, target_s)
        df = dd.get_dummies(df.categorize()).persist()
        y = target_s.to_dask_array(lengths=True)
        X = df.to_dask_array(lengths=True)
    else:
        y = df[target].to_numpy()
        del df[target]
        
        df = pd.get_dummies(df)
        X = df.to_numpy()
        
    return train_test_split_normal(X, y, test_size=.1, random_state=18)


def main():
    print("Setting up data directory")
    print("-------------------------")

    #flights(args.url)
    columns = ['Year', 'Month', 'DayOfWeek', 'Distance', 'DepDelay', 'Origin']
    data_dir = 'data'
    target = 'DepDelay'
    log = ''
    results = {}

    df = get_df(columns).dropna()
    is_dask = True
    
    client = None
    if is_dask:
        client = Client(n_workers=20, threads_per_worker=20, memory_limit='1GB')

    model = GradientBoostingRegressor(random_state=18)
    params = {'max_depth': [2, 3], 'n_estimators': [1, 2, 3]}
    X_train, X_test, y_train, y_test = get_data(df.copy(), target, is_dask=False, chunksize=200)
    results = dict()
    clf_name = type(model).__name__

    clf_cv = GridSearchCV(model,
                          param_grid=params,
                          cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=18),
                          scoring='neg_mean_squared_error'
                         )



    with joblib.parallel_backend("dask" if is_dask else 'loky'):
        start = time.time()
        clf_cv.fit(X_train, y_train)
        end = time.time()

    y_predict_train = clf_cv.best_estimator_.predict(X_train)
    y_predict_test = clf_cv.best_estimator_.predict(X_test)

    train_error = mean_squared_error(y_train, y_predict_train, )
    test_error = mean_squared_error(y_test, y_predict_test, )
    best_params = clf_cv.best_params_
    
    results['Scikit XGBoost'] = {
        'train_error': train_error,
        'test_error': test_error,
        'time': end - start
    }
    log += 'Scikit XGBoost train_error: %.2f, test_error: %.2f, took: %.2f\n' % (train_error, test_error, end - start)
    
    is_dask = True
    X_train, X_test, y_train, y_test = get_data(df.copy(), target, is_dask=is_dask, chunksize=200)
    params = {'objective': 'reg:squarederror',
              'max_depth': 3, 'eta': 0.01, 'subsample': 0.5,
              'min_child_weight': 0.2}

    start = time.time()
    bst = dask_xgboost.train(client, params, X_train, y_train, num_boost_round=10)
    end = time.time()

    y_train_pred = dask_xgboost.predict(client, bst, X_train).persist()
    y_test_pred = dask_xgboost.predict(client, bst, X_test).persist()

    y_train, y_train_pred = dask.compute(y_train, y_train_pred)
    y_test, y_test_pred = dask.compute(y_test, y_test_pred)

    train_error = mean_squared_error(y_train, y_train_pred)
    test_error = mean_squared_error(y_test, y_test_pred)

    log += 'Dask XGBoost train_error: %.2f, test_error: %.2f, took: %.2f' % (train_error, test_error, end - start)
    results['Dask XGBoost'] = {
        'train_error': train_error,
        'test_error': test_error,
        'time': end - start
    }
        
    with open('results.txt', 'w') as outfile:
        json.dump(results, outfile)

    print('Finished!')


if __name__ == '__main__':
    main()


