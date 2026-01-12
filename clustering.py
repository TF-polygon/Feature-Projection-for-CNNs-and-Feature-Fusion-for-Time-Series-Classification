from tsmoothie.smoother import *
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMinMax
from rich.console import Console

import os

import argparse
import numpy as np
import pandas as pd

console = Console()

def export(kmeans, scaled_data, file_name):
    save_path = 'test_data/clustering'
    os.makedirs(save_path, exist_ok=True)

    final_file_name = os.path.join(save_path, file_name)

    np.savez(
        final_file_name,
        scaled_data=scaled_data,
        labels=kmeans.labels_,
        centers=kmeans.cluster_centers_,
        n_clusters=kmeans.n_clusters
    )

def clustering(data, scaled_data, smoother, n_clusters, symbol='EUR/USD', max_iter=20, random_state=123, n_init=3):
    console.log(f'Start to do clustering {symbol}...')
    with console.status('[bold green] Clustering...') as status:
        kmeans = TimeSeriesKMeans(
            n_clusters=n_clusters,
            metric="dtw",
            max_iter=max_iter, 
            random_state=random_state,
            max_iter_barycenter=10,
            n_init=n_init,
            verbose=False
        ).fit(scaled_data)
    
    console.log(f'Clustering {symbol} has finished.')

    # start_index_in_original_data = len(data) - len(kmeans.labels_)
    # dates_for_time = data['Date'][start_index_in_original_data : len(data)]
    time_column_names = [f'Time_{i:02d}' for i in range(1, smoother.Smoother.data.shape[1] + 1)]
    clustered_data = pd.DataFrame(smoother.Smoother.data, columns=time_column_names)
    clustered_data['Cluster_Label'] = kmeans.labels_
    filename = f'clustered_{symbol}_{n_clusters}cls.csv'
    clustered_data.to_csv(os.path.join('clustered_data', filename), index=False)

    export(kmeans, f'{symbol}.npz')

    console.log(f'Successfully saved the clustered data in {filename}.\n\n')

def run(args):
    data = pd.read_csv(args.path)
    window_shape = args.window_shape
    stride = args.stride
    n_clusters = args.n_clusters
    n_init = args.n_init

    smoother = WindowWrapper(LowessSmoother(smooth_fraction=0.6, iterations=1, batch_size=1000), window_shape=window_shape)
    smoother.smooth(data['Close'].values)
    strided_data = smoother.Smoother.data[::stride]
    raw_scaled = TimeSeriesScalerMinMax().fit_transform(strided_data)

    clustering(
        data=data, 
        scaled_data=raw_scaled, 
        smoother=smoother, 
        n_clusters=n_clusters,
        n_init=n_init
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--path', type=str, required=True, help='path to dataset')
    parser.add_argument('--n_clusters', type=int, required=True, help='$k$: number of clusters')
    parser.add_argument('--window_shape', type=int, required=True, help='$W$: window shape for clustering')
    parser.add_argument('--symbol', type=str, default='EUR/USD', help='symbol (ex: EUR/USD)')
    parser.add_argument('--max_iter', type=int, default=10, help='Max Iteration')
    parser.add_argument('--random_state', type=int, default=123, help='Max Iteration')    
    parser.add_argument('--n_init', type=int, default=3, help='n_init')

    args = parser.parse_args()

    run(args)