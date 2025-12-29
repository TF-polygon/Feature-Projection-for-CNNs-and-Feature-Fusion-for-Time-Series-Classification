from tsmoothie.smoother import *
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMinMax
from sklearn.cluster import KMeans
from tqdm import tqdm
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

import pandas as pd
import numpy as np
import datetime
import time
import sys
import os

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
    }).to_csv(os.path.join(save_path, file_name), index=False)

    print(f"Successfully save elbow method results! filename: {file_name}")

def main(path, window_size):
    data = pd.read_csv(path)
    window_shape = window_size

    smoother = WindowWrapper(LowessSmoother(smooth_fraction=0.6, iterations=1, batch_size=1000), window_shape=window_shape)
    smoother.smooth(data['Close'].values)

    raw_scaled = TimeSeriesScalerMinMax().fit_transform(smoother.Smoother.data)

    inertia_arr = []
    sc_arr = []
    chi_arr = []
    dbi_arr = []
    k_range = range(2, 16)

    progress_bar = tqdm(k_range, desc='Cluserting (K-means) : ')

    start_time = time.time()
    for k in progress_bar:
        iter_start_time = time.time()
        kmeans = TimeSeriesKMeans(
            n_clusters=k,
            metric='softdtw',
            metric_params={'gamma': 0.1},
            max_iter=30,
            max_iter_barycente=10,
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

        print(f'K [{k}/16],   cost : {inertia},   SC : {sc},   CHI : {chi},   DBI : {dbi},   Time : {(end_time - start_time) / 60.0:3f}m')

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

    print('Davies Bouldin Score')
    print(dbi_arr)

    print(f'Time Taken: {(end_time - start_time) / 60.0:3f}m')

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python elbow_method.py <path> <window_size>')
    
    else:
        path = str(sys.argv[1])
        window_size = int(sys.argv[2])
        main(path, window_size)