from tsmoothie.smoother import *
import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt
from pyts.image import GramianAngularField, RecurrencePlot
from tqdm import tqdm

def get_image_filter(type, x):
    if type == 'GASF':
        return GramianAngularField(method='summation', image_size=x.shape[1])
    elif type == 'GADF':
        return GramianAngularField(method='difference', image_size=x.shape[1])
    elif type == 'RP':
        return RecurrencePlot(threshold=None)
    
def run(args):
    symbol = args.symbol
    data_types = ['train', 'valid', 'test']
    image_types = ['GASF', 'GADF', 'RP']
    
    data_dir_path = 'data/processed'

    for d_type in data_types:
        for i in range(2, 5):
            file_name = f'clustered_{symbol}_{i}cls_{d_type}.csv'
            full_path = os.path.join(data_dir_path, file_name)

            df = pd.read_csv(full_path)
            X = df.drop('Cluster_Label', axis=1).values
            y = df['Cluster_Label'].values

            for it in image_types:
                image_filter = get_image_filter(it, x=X)
                label_map = {idx: f'class{idx}' for idx in range(i)}
                
                save_to_path = f'datasets/P-FXImageSet/{i}k/{symbol}/{d_type}/{it}'
                for label_name in label_map.values():
                    os.makedirs(os.path.join(save_to_path, label_name), exist_ok=True)
                
                for j, (series, label) in enumerate(tqdm(zip(X, y), total=len(X), desc=f'Running on image conversion ({it})'), start=1):
                    image = image_filter.fit_transform(series.reshape(1, -1))[0]
                    label_name = label_map[label]
                    path = os.path.join(save_to_path, label_name, f'{label_name.lower()}_{j}.png')
                    plt.imsave(path, image, cmap='rainbow')

                print(f"  └─ Finish: {d_type} {i}k {it} conversion completed.")
                    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--symbol', type=str, required=True, help='symbol (ex: EURUSD, GBPUSD)') # symbol

    args = parser.parse_args()

    run(args)