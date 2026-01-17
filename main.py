from muffin.train import run
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, required=True, help='path to dataset')
    parser.add_argument('--input_size', type=int, required=True, help='input size of model')
    parser.add_argument('--num_features', type=int, required=True, default=3, help='number of features for fusion')
    parser.add_argument('--epochs', type=int, required=True, default=10)
    parser.add_argument('--batch_size', type=int, required=True, default=32)
    parser.add_argument('--test', type=bool, default=False)
    parser.add_argument('--num_classes', type=int, default=2, help='number of classes')
    
    args = parser.parse_args()

    run(args)