from src.muffin.train import *

from datetime import datetime
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve

from glob import glob
from torch import nn, optim
from tqdm import tqdm

import pandas as pd
import numpy as np
import argparse
import torch
import os

def main(args):
    model = get_model(args.num_features, args.input_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    train_loader, valid_loader, test_loader = dataloader(args.num_features, args.input_size, args.batch_size, args.dataset)

    if args.num_features == 1:
        train_logs, valid_logs = train_singlefeature(
            model=model,
            epochs=args.epochs,
            criterion=criterion,
            optimizer=optimizer,
            train_loader=train_loader,
            valid_loader=valid_loader,
        )

    elif args.num_features == 2:
        train_logs, valid_logs = train_doublefeatures(
            model=model,
            epochs=args.epochs,
            criterion=criterion,
            optimizer=optimizer,
            train_loader=train_loader,
            valid_loader=valid_loader,
        )    
        
    elif args.num_features == 3:
        train_logs, valid_logs = train_multifeatures(
            model=model,
            epochs=args.epochs,
            criterion=criterion,
            optimizer=optimizer,
            train_loader=train_loader,
            valid_loader=valid_loader,
        )    

    test_labels, test_preds, test_results = test_model(
        model=model,
        num_features=args.num_features,
        test_loader=test_loader,
        criterion=criterion,
        device=device
    )

    export(
        model=model,
        train_logs=train_logs,
        valid_logs=valid_logs,
        test_results=test_results,
        test_labels=test_labels,
        test_preds=test_preds,
    )    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, required=True, help='path to dataset')
    parser.add_argument('--input_size', type=int, required=True, help='input size of model')
    parser.add_argument('--num_features', type=int, required=True, default=3, help='Number of features to input')
    parser.add_argument('--epochs', type=int, required=True, default=10)
    parser.add_argument('--batch_size', type=int, required=True, default=32)
    parser.add_argument('--test', type=bool, default=False)
    
    args = parser.parse_args()

    main(args)