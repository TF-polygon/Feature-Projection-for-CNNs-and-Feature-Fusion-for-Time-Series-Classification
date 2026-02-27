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
    df = pd.read_csv(args.path)
    X = df.drop('Cluster_Label', axis=1).values
    y = df['Cluster_Label'].values

    image_filter = get_image_filter(type=args.image_type, x=X)
    label_map = {f'class{i}': i for i in range(args.num_classes)}

    dir = f'datasets/P-FXImageSet/{args.num_classes}k/{args.data_type}/{args.symbol}/{args.image_type}'
    for label_name in label_map.values():
        os.makedirs(os.path.join(dir, label_name), exist_ok=True)

    for i, (series, label) in enumerate(tqdm(zip(X, y), total=len(X), desc=f"Running on image conversion ({args.image_type}): "), start=1):
        image = image_filter.fit_transform(series.reshape(1, -1))[0]
        label_name = label_map[label]
        path = os.path.join(dir, label_name, f'{label_name.lower()}_{i}.png')
        plt.imsave(path, image, cmap='rainbow')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--path', type=str, required=True, help='path to dataset')
    parser.add_argument('--image_type', type=str, required=True, help='conversion type (ex: GASF, GADF, RP)')
    parser.add_argument('--data_type', type=str, required=True, help='purpose of data (ex: train, valid, test)')
    parser.add_argument('--num_classes', type=int, default=2, help='number of classes')
    parser.add_argument('--symbol', type=str, default='EURUSD', help='symbol (currency pair)')

    args = parser.parse_args()

    run(args)