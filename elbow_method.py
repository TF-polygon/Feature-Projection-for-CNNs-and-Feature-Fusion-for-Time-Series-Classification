from tsmoothie.smoother import *
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMinMax
from rich.console import Console
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

import pandas as pd
import numpy as np
import datetime
import argparse
import time
import sys
import os

console = Console()

def export(k_range, inertia_arr, sc_arr, chi_arr, dbi_arr):
    save_path = 'experimental_data'
    file_name = f'elbow_method_results_{datetime.now().strftime("%m-%d_%H-%M")}.csv'
    os.makedirs(save_path, exist_ok=True)
    
    results_df = pd.DataFrame({
        'K': list(k_range),
        'Inertia': inertia_arr,
        'SC': sc_arr,
        'CHI': chi_arr,
        'DBI': dbi_arr
    })
    
    results_df.to_csv(os.path.join(save_path, file_name), index=False)

    print(f"Successfully save elbow method results! filename: {file_name}")

def elbow_method(args):
    data = pd.read_csv(args.path)
    window_shape = args.window_shape
    stride = args.stride

    smoother = WindowWrapper(LowessSmoother(smooth_fraction=0.6, iterations=1, batch_size=1000), window_shape=window_shape)
    smoother.smooth(data['Close'].values)
    strided_data = smoother.Smoother.data[::stride]

    raw_scaled = TimeSeriesScalerMinMax().fit_transform(strided_data)

    inertia_arr = []
    sc_arr = []
    chi_arr = []
    dbi_arr = []
    k_range = range(2, 16)

    start_time = time.time()

    
    console.log(f'Start to elbow method for {args.path}')

    with console.status('[bold green] Clustering...') as status:
        for k in k_range:
            status.update(f'[bold green] [{k} / 16] Clustering...')
            iter_start_time = time.time()
            kmeans = TimeSeriesKMeans(
                n_clusters=k,
                metric='dtw',
                max_iter=20,
                random_state=123,
                n_init=3,
                verbose=False
            )
            kmeans.fit(raw_scaled)
            
            inertia = kmeans.inertia_
            labels = kmeans.labels_

            sc = silhouette_score(raw_scaled.reshape(raw_scaled.shape[0], -1), labels, metric='euclidean')
            chi = calinski_harabasz_score(raw_scaled.reshape(raw_scaled.shape[0], -1), labels)
            dbi = davies_bouldin_score(raw_scaled.reshape(raw_scaled.shape[0], -1), labels)

            inertia_arr.append(inertia)
            sc_arr.append(sc)
            chi_arr.append(chi)
            dbi_arr.append(dbi)
            iter_end_time = time.time()

            console.log(f'K [{k}/16], cost : {inertia:.4f}, SC : {sc:.4f}, CHI : {chi:.4f}, DBI : {dbi:.4f}, Time : {(iter_end_time - iter_start_time) / 60.0:.3f}m')

    end_time = time.time()

    inertia_arr = np.array(inertia_arr)
    sc_arr = np.array(sc_arr)
    chi_arr = np.array(chi_arr)
    dbi_arr = np.array(dbi_arr)

    print('Inertia')
    print(inertia_arr)
    
    print('Silhouette Coefficient Score')
    print(sc_arr)
    
    print('Calinski Harabasz Index')
    print(chi_arr)

    print('Davies Bouldin Index')
    print(dbi_arr)

    print(f'Time Taken: {(end_time - start_time) / 60.0:3f}m')

    export(
        k_range=k_range,
        inertia_arr=inertia_arr,
        sc_arr=sc_arr,
        chi_arr=chi_arr,
        dbi_arr=dbi_arr
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--path', type=str, required=True, help='path to dataset')
    parser.add_argument('--window_shape', type=int, required=True, help='Window shape to clustering')
    parser.add_argument('--stride', type=int, required=True, help='Stride')

    args = parser.parse_args()

    elbow_method(args)