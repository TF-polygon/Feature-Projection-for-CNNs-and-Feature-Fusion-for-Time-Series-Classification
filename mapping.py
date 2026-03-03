import pandas as pd
import numpy as np
import argparse
import joblib
import os

from rich.console import Console
from datetime import datetime
from tqdm import tqdm

console = Console()

def preprocess_and_label(model_path, raw_data, window_size=48):
    model = joblib.load(model_path)
    labeled_rows = []
    data_array = np.array(raw_data).astype(float)
    
    with console.status(f'Labeling...'):
        for i in range(0, len(raw_data) - window_size + 1, 12): # stride=12
            window = data_array[i : i + window_size] # .values.flatten()
            
            w_min = window.min()
            w_max = window.max()
            
            if w_max - w_min == 0:
                normalized_window = np.zeros_like(window)
            else:
                normalized_window = (window - w_min) / (w_max - w_min)

            input_data = normalized_window.reshape(1, window_size, 1)
            cluster_label = model.predict(input_data)[0]
            combined_row = np.append(normalized_window, cluster_label)
            labeled_rows.append(combined_row)
        
    cols = [f'Time_{i+1:02d}' for i in range(window_size)] + ['Cluster_Label']
    return pd.DataFrame(labeled_rows, columns=cols)

def export(df, symbol, n_clusters, data_type='valid'):
    save_path = 'data/processed'
    file_name = f'clustered_{symbol.upper()}_{n_clusters}cls_{data_type}.csv'
    os.makedirs(save_path, exist_ok=True)
    
    df.to_csv(os.path.join(save_path, file_name), index=False)

    print(f"Successfully save pseudo-labeled {data_type} dataset! filename: {file_name}")

def run(args):
    model_path = args.model_path
    window_size = args.window_size
    symbol = args.symbol
    n_clusters = args.n_clusters
    data_type = args.data_type
    
    df = pd.read_csv(args.data_path)
    raw_data = pd.to_numeric(df['Close'], errors='coerce').dropna()

    labeled_df = preprocess_and_label(model_path, raw_data, window_size)
    export(labeled_df, symbol, n_clusters, data_type)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--symbol', type=str, required=True)
    parser.add_argument('--n_clusters', type=int, required=True)
    parser.add_argument('--window_size', type=int, required=True)
    parser.add_argument('--data_type', type=str, default='valid')

    args = parser.parse_args()

    run(args)