from muffin.model import muffincnn_singlefeature, muffincnn_dualfusion, muffincnn_triplefusion
from muffin.dataset import MultiFeatureFusionDataset, DualFeatureFusionDataset, MultiFeatureNPZdataset, DualFeatureNPZdataset

from datetime import datetime
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from torch import nn, optim
from tqdm import tqdm

import pandas as pd
import numpy as np
import torch
import os

def get_model(num_features, num_classes, input_size):
    if num_features == 1:
        return muffincnn_singlefeature(input_size, num_classes)
    elif num_features == 2:
        return muffincnn_dualfusion(input_size, num_classes)
    elif num_features == 3:
        return muffincnn_triplefusion(input_size, num_classes)
    
def export(args, model, train_logs, valid_logs, test_results=None, test_labels=None, test_preds=None):
    train_data_path = 'data/results/train_history'
    weight_path = 'weights'
    os.makedirs(train_data_path, exist_ok=True)
    os.makedirs(weight_path, exist_ok=True)

    k = args.num_classes
    f = args.num_features
    symbol = args.dataset[-6:]

    if f == 1:
        img_map = args.f1.upper()
        symbol = f'{symbol}_{img_map}'
    if f == 2:
        img_map1 = args.f1.upper()
        img_map2 = args.f2.upper()
        symbol = f'{symbol}_{img_map1}_{img_map2}'

    log_df = pd.DataFrame(train_logs)
    log_valdf = pd.DataFrame(valid_logs)

    experimental_case = f'{k}k_{f}f_{symbol}'
    final_path = os.path.join(train_data_path, experimental_case)

    log_df.to_csv(final_path + "_train.csv", index=False)
    log_valdf.to_csv(final_path + "_valid.csv", index=False)    
    
    torch.save(model.state_dict(), os.path.join(weight_path, experimental_case + ".pt"))

    if test_results:
        log_testdf = pd.DataFrame(test_results)
        log_testdf.to_csv(final_path + "_test.csv", index=False)
        print(f"Successfully save training results! filename: [{final_path}_train.csv, {final_path}_valid.csv, {final_path}_test.csv]")     
        
        test_data_path = 'data/results/test_npz'
        os.makedirs(test_data_path, exist_ok=True)
        npz_file_path = os.path.join(test_data_path, experimental_case)

        test_labels_np = np.array(test_labels)
        test_preds_np = np.array(test_preds)

        np.savez_compressed(
            npz_file_path,
            labels=test_labels_np,
            preds=test_preds_np,
        )
        
        return
    
    print(f"Successfully save training results! filename: [{final_path}_train.csv, {final_path}_valid.csv]") 

def dataloader(num_features, input_size, batch_size, path, f1=None, f2=None, num_classes=2):
    train_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.RandomErasing(),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor()
    ])
    
    class_to_idx = {f'class{i}': i for i in range(num_classes)}
    train_path = os.path.join(path, 'train')
    valid_path = os.path.join(path, 'valid')
    test_path = os.path.join(path, 'test')

    def create_dataset(root, transform):
        if num_features == 1:
            return datasets.ImageFolder(root=os.path.join(root, f1.upper()), transform=transform)
        elif num_features == 2:
            return DualFeatureFusionDataset(
                root_dir=root, 
                class_to_idx=class_to_idx,                         
                f1=f1, 
                f2=f2, 
                transform=transform
            )
        else:
            return MultiFeatureFusionDataset(
                root_dir=root, 
                class_to_idx=class_to_idx, 
                transform=transform
            )

    train_dataset = create_dataset(train_path, train_transform)
    valid_dataset = create_dataset(valid_path, eval_transform)
    test_dataset = create_dataset(test_path, eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def run_epoch(model, data_loader, criterion, optimizer, total_epochs, current_epoch, device, is_training=True):
    if is_training: model.train()
    else: model.eval()

    running_loss = 0.0
    all_preds, all_labels = [], []

    epoch_type = 'Train' if is_training else 'Valid'
    desc = f'{epoch_type} Epoch [{current_epoch}/{total_epochs}]'
    progress_bar = tqdm(data_loader, desc=desc, leave=True)

    with torch.enable_grad() if is_training else torch.no_grad():
        for data in progress_bar:
            inputs = [d.to(device) for d in data[:-1]]
            labels = data[-1].to(device)

            if is_training:
                optimizer.zero_grad()
            
            outputs = model(*inputs)
            loss = criterion(outputs, labels)

            if is_training:
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            acc = accuracy_score(all_labels, all_preds)
            prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
            rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)

            progress_bar.set_postfix({
                'Loss': f"{loss.item():.4f}", 
                'Acc': f"{acc:.4f}",
                'Prec': f"{prec:.4f}",
                'Rec': f"{rec:.4f}"
            })

        avg_loss = running_loss / len(data_loader)
        final_acc = accuracy_score(all_labels, all_preds)
        final_prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        final_rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)

    epoch_type = 'Train' if is_training else 'Val'
    metrics = {
        "Epoch": current_epoch,
        f"{epoch_type} Loss": running_loss / len(data_loader),
        f"{epoch_type} Accuracy": final_acc,
        f"{epoch_type} Precision": final_prec,
        f"{epoch_type} Recall": final_rec
    }

    epoch_result_log = f"\nEpoch {current_epoch}, Summary: Loss = {avg_loss:.4f}, Accuracy = {final_acc:.4f}, Precision = {final_prec:.4f}, Recall = {final_rec:.4f}\n" if is_training else f"✅ Validation Summary: Loss = {avg_loss:.4f}, Acc = {final_acc:.4f}, Prec = {final_prec:.4f}, Rec = {final_rec:.4f}\n"
    print(epoch_result_log, flush=True)

    return metrics

def train(model, epochs, criterion, optimizer, train_loader, valid_loader, device=torch.device("cuda")):
    train_logs, valid_logs = [], []
    
    for epoch in range(epochs):
        t_m = run_epoch(
            model=model, 
            data_loader=train_loader, 
            criterion=criterion, 
            optimizer=optimizer,
            total_epochs=epochs, 
            current_epoch=epoch + 1, 
            device=device,
            is_training=True
        )
        v_m = run_epoch(
            model=model, 
            data_loader=valid_loader, 
            criterion=criterion, 
            optimizer=optimizer, 
            total_epochs=epochs, 
            current_epoch=epoch + 1, 
            device=device,
            is_training=False
        )
        train_logs.append(t_m)
        valid_logs.append(v_m)
    
    return train_logs, valid_logs

def test(model, test_loader, criterion, device):
    test_preds, test_labels, test_results = [], [], []
    test_loss = 0.0
    
    print("Start to test the model")
    progress_bar = tqdm(test_loader, desc="Testing", leave=True)

    with torch.no_grad():
        for data in progress_bar:
            inputs = [d.to(device) for d in data[:-1]]
            labels = data[-1].to(device)

            outputs = model(*inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    test_acc = accuracy_score(test_labels, test_preds)
    test_prec = precision_score(test_labels, test_preds, average='macro', zero_division=0)
    test_rec = recall_score(test_labels, test_preds, average='macro', zero_division=0)

    test_results.append({
        "Test Loss": test_loss,
        "Test Accuracy": test_acc,
        "Test Precision": test_prec,
        "Test Recall": test_rec
    })

    print(f"\n🏁 Test Set Evaluation Result:")
    print(f"Loss = {float(test_loss / len(test_loader)):.4f}, Accuracy = {test_acc:.4f}, Precision = {test_prec:.4f}, Recall = {test_rec:.4f} ")

    return test_labels, test_preds, test_results

def run(args):
    model = get_model(args.num_features, args.num_classes, args.input_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    train_loader, valid_loader, test_loader = dataloader(
        path=args.dataset,
        num_features=args.num_features, 
        input_size=args.input_size, 
        batch_size=args.batch_size, 
        num_classes=args.num_classes,
        f1=args.f1,
        f2=args.f2
    )

    train_logs, valid_logs = train(
        model=model,
        epochs=args.epochs,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=device
    )    

    test_labels, test_preds, test_results = test(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
    )

    export(
        args=args,
        model=model,
        train_logs=train_logs,
        valid_logs=valid_logs,
        test_results=test_results,
        test_labels=test_labels,
        test_preds=test_preds,
    )    

if __name__ == '__main__':
    dir()