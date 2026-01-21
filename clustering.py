from tsmoothie.smoother import *
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMinMax
from rich.console import Console
from datetime import datetime

import os

import joblib
import argparse
import numpy as np
import pandas as pd

console = Console()

def export(kmeans, scaled_data, file_name, model_name):
    save_path = 'test_data/clustering'
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, 'model'), exist_ok=True)

    npz_file_name = os.path.join(save_path, file_name)
    model_file_name = os.path.join(os.path.join(save_path, 'model'), model_name)

    np.savez(
        npz_file_name,
        scaled_data=scaled_data,
        labels=kmeans.labels_,
        centers=kmeans.cluster_centers_,
        n_clusters=kmeans.n_clusters
    )

    joblib.dump(kmeans, model_file_name)

def clustering(scaled_data, n_clusters, symbol='EURUSD', max_iter=20, random_state=123, n_init=3):
    console.log(f'Start to do clustering {symbol}...')
    with console.status('[bold green] Clustering...') as status:
        kmeans = TimeSeriesKMeans(
            n_clusters=n_clusters,
            metric="dtw",
            max_iter=max_iter, 
            random_state=random_state,
            n_init=n_init,
            verbose=False
        ).fit(scaled_data)
    
    console.log(f'Clustering {symbol} has finished.')

    reshaped_data = scaled_data.reshape(scaled_data.shape[0], scaled_data.shape[1])
    time_column_names = [f'Time_{i:02d}' for i in range(1, reshaped_data.shape[1] + 1)]
    clustered_data = pd.DataFrame(reshaped_data, columns=time_column_names)
    clustered_data['Cluster_Label'] = kmeans.labels_
    filename = f'clustered_{symbol}_{n_clusters}cls.csv'
    path = 'data/processed'
    clustered_data.to_csv(os.path.join(path, filename), index=False)

    export(kmeans, reshaped_data, f'{symbol}_{n_clusters}k.npz', f'{symbol}_{n_clusters}k.joblib')

    console.log(f'Successfully saved the clustered data in {filename}.\n\n')

def run(args):
    data = pd.read_csv(args.path)
    window_shape = args.window_shape
    stride = args.stride
    n_clusters = args.n_clusters
    n_init = args.n_init
    symbol = args.symbol

    smoother = WindowWrapper(LowessSmoother(smooth_fraction=0.6, iterations=1, batch_size=1000), window_shape=window_shape)
    smoother.smooth(data['Close'].values)
    strided_data = smoother.Smoother.data[::stride]
    raw_scaled = TimeSeriesScalerMinMax().fit_transform(strided_data)

    clustering(
        scaled_data=raw_scaled,
        n_clusters=n_clusters,
        n_init=n_init,
        symbol=symbol
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--path', type=str, required=True, help='path to dataset')
    parser.add_argument('--n_clusters', type=int, required=True, help='$k$: number of clusters')
    parser.add_argument('--window_shape', type=int, required=True, help='$W$: window shape for clustering')
    parser.add_argument('--symbol', type=str, default='EURUSD', help='symbol (ex: EURUSD)')
    parser.add_argument('--max_iter', type=int, default=10, help='max iteration')
    parser.add_argument('--random_state', type=int, default=123, help='random state')    
    parser.add_argument('--n_init', type=int, default=3, help='n_init')
    parser.add_argument('--stride', type=int, default=12, help='stride for clustering')

    args = parser.parse_args()

    run(args)