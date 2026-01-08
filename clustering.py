from tsmoothie.smoother import *
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMinMax
from sklearn.cluster import KMeans
from rich.console import Console

import os
import time

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def clustering(data, scaled_data, smoother, n_clusters, symbol='EUR/USD', max_iter=20, random_state=123):
    print(f'Start to do clustering {symbol}...')
    start_time = time.time()
    kmeans = TimeSeriesKMeans(
        n_clusters=n_clusters,
        metric="dtw",
        max_iter=max_iter, 
        random_state=random_state,
        max_iter_barycenter=10,
        n_init=3,
        verbose=False
    ).fit(scaled_data)
    end_time = time.time()
    print(f'Clustering {symbol} has finished.')
    print(f'Time Taken: {(end_time - start_time) / 60.0:3f}m')

    start_index_in_original_data = len(data) - len(kmeans.labels_)
    dates_for_time = data['Date'][start_index_in_original_data : len(data)]
    time_column_names = [f'Time_{i:02d}' for i in range(1, smoother.Smoother.data.shape[1] + 1)]
    clustered_data = pd.DataFrame(smoother.Smoother.data, columns=time_column_names)
    clustered_data['Cluster_Label'] = kmeans.labels_
    filename = f'clustered_{symbol}_{n_clusters}cls.csv'
    clustered_data.to_csv(os.path.join('clustered_data', filename), index=False)

    print(f'Successfully saved the clustered data in {filename}.\n\n')

def main(args):
    data = pd.read_csv(args.path)
    window_shape = args.window_shape
    stride = args.stride
    n_clusters = args.n_clusters

    smoother = WindowWrapper(LowessSmoother(smooth_fraction=0.6, iterations=1, batch_size=1000), window_shape=window_shape)
    smoother.smooth(data['Close'].values)
    strided_data = smoother.Smoother.data[::stride]
    raw_scaled = TimeSeriesScalerMinMax().fit_transform(strided_data)

    clustering(
        data=data, 
        scaled_data=raw_scaled, 
        smoother=smoother, 
        n_clusters=n_clusters
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--path', type=str, required=True, help='path to dataset')
    parser.add_argument('--n_clusters', type=int, required=True, help='k: number of clusters')
    parser.add_argument('--symbol', type=str, default='EUR/USD', help='symbol (ex: EUR/USD)')
    parser.add_argument('--max_iter', type=int, default=10, help='Max Iteration')
    parser.add_argument('--random_state', type=int, default=123, help='Max Iteration')    

    args = parser.parse_args()

    main(args)