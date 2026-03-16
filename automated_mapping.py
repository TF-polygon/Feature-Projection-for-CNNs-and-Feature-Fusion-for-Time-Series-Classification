import pandas as pd
import numpy as np
import argparse
import joblib
import os

from rich.console import Console

def load_model(path, symbol):
    model_name_2k = symbol + '_2k.joblib'
    model_name_3k = symbol + '_3k.joblib'
    model_name_4k = symbol + '_4k.joblib'

    model_2k = joblib.load(os.path.join(path, model_name_2k))
    model_3k = joblib.load(os.path.join(path, model_name_3k))
    model_4k = joblib.load(os.path.join(path, model_name_4k))

    return [model_2k, model_3k, model_4k]

def export(df, symbol, n_clusters, data_type='valid'):
    save_path = 'data/processed'
    file_name = f'clustered_{symbol.upper()}_{n_clusters}cls_{data_type}.csv'
    os.makedirs(save_path, exist_ok=True)
    
    df.to_csv(os.path.join(save_path, file_name), index=False)

    print(f"Successfully save pseudo-labeled {data_type} dataset! filename: {file_name}")

def run(args):
    models = load_model(args.model_dir_path, args.symbol)
    data_type = ['valid', 'test']

    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument() # --model_dir_path
    parser.add_argument() # --symbol
    parser.add_argument() # --
    parser.add_argument() # --

    args = parser.parse_args()

    run(args)
