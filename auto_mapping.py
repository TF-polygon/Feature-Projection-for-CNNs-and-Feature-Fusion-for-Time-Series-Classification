import pandas as pd
import numpy as np
import argparse
import joblib
import os

from tqdm import tqdm

def export(df, symbol, n_clusters, data_type='valid'):
    save_path = 'data/processed'
    file_name = f'clustered_{symbol.upper()}_{n_clusters}cls_{data_type}.csv'
    os.makedirs(save_path, exist_ok=True)
    
    df.to_csv(os.path.join(save_path, file_name), index=False)

    print(f"Successfully save pseudo-labeled {data_type} dataset! filename: {file_name}")

def run(args):
    symbol = args.symbol
    data_type = ['valid', 'test']
    n_clusters = [2, 3, 4]
    window_size = args.window_size
    
    for type in data_type:
        for num_clusters in range(2, 5):
            labeled_rows = []
            df = pd.read_csv(os.path.join(args.data_path, f'{symbol.lower()}1h_raw_100-15_{type}.csv'))
            raw_data = pd.to_numeric(df['Close'], errors='coerce').dropna()
            data_array = np.array(raw_data).astype(float)

            model_path = f'clustering_data/model/{symbol.upper()}_{num_clusters}k.joblib'
            model = joblib.load(model_path)

            progress_bar = tqdm(range(0, len(raw_data) - window_size + 1, 12), desc=f"Mapping {symbol}/{num_clusters}k/{type}", unit="window")

            for i in progress_bar:
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
            exported_data = pd.DataFrame(labeled_rows, columns=cols)
            export(exported_data, f"{symbol.upper()}", num_clusters, type)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, required=True, help='path to raw data directory') # --data_path
    parser.add_argument('--symbol', type=str, required=True, help='symbol (ex: EURUSD, GBPUSD)') # --symbol
    parser.add_argument('--window_size', type=int, required=True, help='window size for the symbol') # --window_size

    args = parser.parse_args()

    run(args)
