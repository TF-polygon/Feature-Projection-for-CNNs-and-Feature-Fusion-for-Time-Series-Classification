import pandas as pd
import numpy as np
import argparse
import joblib
import os

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def create_path_joblib(path):
    return os.path.join('test_data/clustering/model', path)

def create_path_dataset(path):
    return os.path.join('datasets', path)

def load(model_path, input_data_path, ground_truth_path):
    model = joblib.load(model_path)
    input_data = pd.read_csv(input_data_path)
    ground_truth = pd.read_csv(ground_truth_path)

    return model, input_data, ground_truth

def preprocess(raw_data, window_shape, stride=12):
    scaler = StandardScaler()
    scaler.fit(raw_data[['Close']])
    
    scaled_data = scaler.fit_transform(raw_data[['Close']]).flatten()

    windows = []
    for i in range(0, len(scaled_data) - window_shape + 1, window_shape):
        window = scaled_data[i : i + window_shape]
        windows.append(window)
    
    X_all = np.array(windows)
    split_idx = int(len(X_all) * 0.7)
    
    X_val = X_all[split_idx:]
    
    return X_val, split_idx

def map_labels(true, pred):    
    true = np.array(true).astype(int)
    pred = np.array(pred).astype(int)
    
    cm = confusion_matrix(true, pred)
    row_ind, col_ind = linear_sum_assignment(-cm)
    mapping = dict(zip(col_ind, row_ind))
    return np.array([mapping[l] for l in pred])

def visualize_comparison(X_val, y_true, y_pred, window_shape, n_samples=3):
    """
    y_true와 y_pred를 비교하여 시각화
    X_val: 스케일링된 윈도우 데이터 (N, Window_shape, 1)
    """
    clusters = np.unique(y_true)
    n_clusters = len(clusters)
    
    plt.figure(figsize=(15, n_clusters * 4))
    
    for i, cluster in enumerate(clusters):
        true_indices = np.where(y_true == cluster)[0]
        pred_indices = np.where(y_pred == cluster)[0]
        
        plt.subplot(n_clusters, 2, i*2 + 1)
        for j in range(min(n_samples, len(true_indices))):
            idx = true_indices[j]
            plt.plot(X_val[idx].flatten(), label=f'True Sample {j+1}', alpha=0.7)
        plt.title(f'Ground Truth: Cluster {cluster}')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(n_clusters, 2, i*2 + 2)
        for j in range(min(n_samples, len(pred_indices))):
            idx = pred_indices[j]
            plt.plot(X_val[idx].flatten(), label=f'Pred Sample {j+1}', alpha=0.7)
        plt.title(f'Model Prediction: Cluster {cluster}')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def run(args):
    model_path = create_path_joblib(args.model)
    input_data_path = create_path_dataset(args.dataset)
    ground_truth_path = create_path_dataset(args.ground_truth)

    model, input_data, ground_truth = load(
        model_path=model_path, 
        input_data_path=input_data_path, 
        ground_truth_path=ground_truth_path
    )

    X_val = input_data.filter(regex='^Time').values
    X_val_reshaped = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
    y_pred_raw = model.predict(X_val_reshaped).astype(int)
    y_true = ground_truth.iloc[:, -1].values.astype(int)
    y_pred_mapped = map_labels(y_true, y_pred_raw)

    print("=== Final Validation Result (Full-Scaled 30% Data) ===")
    print(f"Window Shape: {args.window_shape}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred_mapped):.4f}")
    print("\n[Classification Report]")
    print(classification_report(y_true, y_pred_mapped))

    visualize_comparison(X_val, y_true, y_pred_mapped, args.window_shape)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, required=True, help='path to model joblib')
    parser.add_argument('--dataset', type=str, required=True, help='path to dataset')
    parser.add_argument('--ground_truth', type=str, required=True, help='path to labeld data which has ground truths')
    parser.add_argument('--window_shape', type=int, required=True, help='window_shape')
    args = parser.parse_args()

    run(args)