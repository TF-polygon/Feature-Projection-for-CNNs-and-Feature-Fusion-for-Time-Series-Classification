import argparse
import pandas as pd
import os
import numpy as np

def main(args):
    df = pd.read_csv(args.path)

    if args.label_type == 'fth':
        df['Future_Close'] = df['Close'].shift(-args.window_shape)
        df['Return'] = (df['Future_Close'] / df['Close']) - 1

        conditions = [
            (df['Return'] > args.threshold_rate),
            (df['Return'] < -args.threshold_rate)
        ]

        choices = [1, -1]
        df_len = len(df)
        df['Label'] = np.select(conditions, choices, default=0)
        labeled_df = df.iloc[:df_len - args.window_shape]

        labeled_df.to_csv(args.output, index=False)
    
    else: # tbl
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--path', type=str, required=True, help='path to raw data')
    parser.add_argument('--output', type=str, required=True, help='path to save')
    parser.add_argument('--label_type', type=str, required=True, help='labeling type ex) fth or tbl')
    parser.add_argument('--window_shape', type=int)
    parser.add_argument('--threshold_rate', type=float)
    
    args = parser.parse_args()

    main(args)