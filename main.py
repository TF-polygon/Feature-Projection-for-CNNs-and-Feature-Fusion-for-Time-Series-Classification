from muffin.train import run
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, required=True, help='path to dataset')
    parser.add_argument('--input_size', type=int, required=True, help='input size of model')
    parser.add_argument('--num_features', type=int, required=True, default=3, help='Number of features to input')
    parser.add_argument('--epochs', type=int, required=True, default=10)
    parser.add_argument('--batch_size', type=int, required=True, default=32)
    parser.add_argument('--test', type=bool, default=False)
    
    args = parser.parse_args()

    run(args)