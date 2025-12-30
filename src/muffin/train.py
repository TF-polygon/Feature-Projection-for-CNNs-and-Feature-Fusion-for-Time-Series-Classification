from muffin.model import single_feature_model, double_features_model, multi_features_model
from muffin.utils import DoubleFeatureNPZdataset, MultiFeatureNPZdataset

from datetime import datetime
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve

from torch import nn, optim
from tqdm import tqdm

import pandas as pd
import numpy as np
import torch
import os

def get_model(num_features, input_size):
    if num_features == 1:
        return single_feature_model(input_size)
    elif num_features == 2:
        return double_features_model(input_size)
    elif num_features == 3:
        return multi_features_model(input_size) # multi_features_mfct_net(input_size)
    
def export(model, train_logs, valid_logs, test_results=None, test_labels=None, test_preds=None):
    train_data_path = 'data/results'
    weight_path = 'weights'
    os.makedirs(train_data_path, exist_ok=True)
    os.makedirs(weight_path, exist_ok=True)
    weight_name = datetime.now().strftime("%m-%d_%H-%M")
    torch.save(model.state_dict(), os.path.join(weight_path, weight_name + ".pt"))
    log_df = pd.DataFrame(train_logs)
    log_valdf = pd.DataFrame(valid_logs)

    log_df.to_csv(os.path.join(train_data_path, weight_name + "_train.csv"), index=False)
    log_valdf.to_csv(os.path.join(train_data_path, weight_name + "_valid.csv"), index=False)    

    if test_results:
        log_testdf = pd.DataFrame(test_results)
        log_testdf.to_csv(os.path.join(train_data_path, weight_name + "_test.csv"), index=False)
        print(f"Successfully save training results! filename: [{weight_name}_train.csv, {weight_name}_valid.csv, {weight_name}_test.csv]")     
        
        test_data_path = 'data/results/test_data'
        os.makedirs(test_data_path, exist_ok=True)
        npz_file_path = os.path.join(test_data_path, weight_name)

        test_labels_np = np.array(test_labels)
        test_preds_np = np.array(test_preds)

        return np.savez_compressed(
            npz_file_path,
            labels=test_labels_np,
            preds=test_preds_np,
        )
    
    print(f"Successfully save training results! filename: [{weight_name}_train.csv, {weight_name}_valid.csv]") 

def dataloader(num_features, input_size, batch_size, path):
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.RandomErasing(),
    ])
    
    if num_features == 1:
        dataset = datasets.ImageFolder(root=path, transform=transform)
    elif num_features == 2:
        dataset = DoubleFeatureNPZdataset(path, transform)
    else:
        dataset = MultiFeatureNPZdataset(path, transform)

    total_size = len(dataset)
    train_size = int(total_size * 0.7)
    val_size = int(total_size * 0.15)
    test_size = total_size - train_size - val_size
    
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def train_singlefeature(model, epochs, criterion, optimizer, train_loader, valid_loader, device=torch.device("cuda")):
    train_logs, valid_logs = [], []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        progress_bar = tqdm(train_loader, desc=f"Train Epoch [{epoch + 1}/{epochs}]", leave=True)

        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
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

        avg_loss = running_loss / len(train_loader)
        final_acc = accuracy_score(all_labels, all_preds)
        final_prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        final_rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)

        train_logs.append({
            "Epoch": epoch + 1,
            "Train Loss": running_loss / len(train_loader),
            "Train Accuracy": final_acc,
            "Train Precision": final_prec,
            "Train Recall": final_rec
        })

        print(f"\nEpoch {epoch + 1}, Summary: Loss = {avg_loss:.4f}, Accuracy = {final_acc:.4f}, Precision = {final_prec:.4f}, Recall = {final_rec:.4f}\n", flush=True)

        model.eval()
        val_preds, val_labels = [], []
        val_loss = 0.0

        valid_progress_bar = tqdm(valid_loader, desc=f"Validation Epoch [{epoch + 1}/epochs]", leave=True)

        with torch.no_grad():
            for images, labels in valid_progress_bar:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

                valid_progress_bar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'Acc': f"{accuracy_score(val_labels, val_preds):.4f}",
                    'Prec': f"{precision_score(val_labels, val_preds, average='macro', zero_division=0):.4f}",
                    'Rec': f"{recall_score(val_labels, val_preds, average='macro', zero_division=0):.4f}"
                })

        val_acc = accuracy_score(val_labels, val_preds)
        val_prec = precision_score(val_labels, val_preds, average='macro', zero_division=0)
        val_rec = recall_score(val_labels, val_preds, average='macro', zero_division=0)
        avg_val_loss = val_loss / len(valid_loader)
        print(f"✅ Validation Summary: Loss = {avg_val_loss:.4f}, Acc = {val_acc:.4f}, Prec = {val_prec:.4f}, Rec = {val_rec:.4f}\n")
        
        valid_logs.append({
            "Epoch": epoch + 1,
            "Val Loss": val_loss / len(valid_loader),
            "Val Accuracy": val_acc,
            "Val Precision": val_prec,
            "Val Recall": val_rec
        })
    
    return train_logs, valid_logs

def train_doublefeatures(model, epochs, criterion, optimizer, train_loader, valid_loader, device=torch.device("cuda")):
    train_logs, valid_logs = [], []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        progress_bar = tqdm(train_loader, desc=f"Train Epoch [{epoch + 1}/{epochs}]", leave=True)

        for ch1, ch2, labels in progress_bar:
            ch1 = ch1.to(device)
            ch2 = ch2.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(ch1, ch2)
            loss = criterion(outputs, labels)
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

        avg_loss = running_loss / len(train_loader)
        final_acc = accuracy_score(all_labels, all_preds)
        final_prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        final_rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)

        train_logs.append({
            "Epoch": epoch + 1,
            "Train Loss": running_loss / len(train_loader),
            "Train Accuracy": final_acc,
            "Train Precision": final_prec,
            "Train Recall": final_rec
        })

        print(f"\nEpoch {epoch + 1}, Summary: Loss = {avg_loss:.4f}, Accuracy = {final_acc:.4f}, Precision = {final_prec:.4f}, Recall = {final_rec:.4f}\n", flush=True)

        model.eval()
        val_preds, val_labels = [], []
        val_loss = 0.0

        valid_progress_bar = tqdm(valid_loader, desc=f"Validation Epoch [{epoch + 1}/epochs]", leave=True)

        with torch.no_grad():
            for ch1, ch2, labels in valid_progress_bar:
                ch1 = ch1.to(device)
                ch2 = ch2.to(device)
                labels = labels.to(device)

                outputs = model(ch1, ch2)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

                valid_progress_bar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'Acc': f"{accuracy_score(val_labels, val_preds):.4f}",
                    'Prec': f"{precision_score(val_labels, val_preds, average='macro', zero_division=0):.4f}",
                    'Rec': f"{recall_score(val_labels, val_preds, average='macro', zero_division=0):.4f}"
                })

        val_acc = accuracy_score(val_labels, val_preds)
        val_prec = precision_score(val_labels, val_preds, average='macro', zero_division=0)
        val_rec = recall_score(val_labels, val_preds, average='macro', zero_division=0)
        avg_val_loss = val_loss / len(valid_loader)
        print(f"✅ Validation Summary: Loss = {avg_val_loss:.4f}, Acc = {val_acc:.4f}, Prec = {val_prec:.4f}, Rec = {val_rec:.4f}\n")
        
        valid_logs.append({
            "Epoch": epoch + 1,
            "Val Loss": val_loss / len(valid_loader),
            "Val Accuracy": val_acc,
            "Val Precision": val_prec,
            "Val Recall": val_rec
        })

    return train_logs, valid_logs

def train_multifeatures(model, epochs, criterion, optimizer, train_loader, valid_loader, device=torch.device("cuda")):
    train_logs, valid_logs = [], []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        progress_bar = tqdm(train_loader, desc=f"Train Epoch [{epoch + 1}/{epochs}]", leave=True)

        for ch1, ch2, ch3, labels in progress_bar:
            ch1 = ch1.to(device)
            ch2 = ch2.to(device)
            ch3 = ch3.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(ch1, ch2, ch3)
            loss = criterion(outputs, labels)
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

        avg_loss = running_loss / len(train_loader)
        final_acc = accuracy_score(all_labels, all_preds)
        final_prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        final_rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)

        train_logs.append({
            "Epoch": epoch + 1,
            "Train Loss": running_loss / len(train_loader),
            "Train Accuracy": final_acc,
            "Train Precision": final_prec,
            "Train Recall": final_rec
        })

        print(f"\nEpoch {epoch + 1}, Summary: Loss = {avg_loss:.4f}, Accuracy = {final_acc:.4f}, Precision = {final_prec:.4f}, Recall = {final_rec:.4f}\n", flush=True)

        model.eval()
        val_preds, val_labels = [], []
        val_loss = 0.0

        valid_progress_bar = tqdm(valid_loader, desc=f"Validation Epoch [{epoch + 1}/epochs]", leave=True)

        with torch.no_grad():
            for ch1, ch2, ch3, labels in valid_progress_bar:
                ch1 = ch1.to(device)
                ch2 = ch2.to(device)
                ch3 = ch3.to(device)
                labels = labels.to(device)

                outputs = model(ch1, ch2, ch3)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

                valid_progress_bar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'Acc': f"{accuracy_score(val_labels, val_preds):.4f}",
                    'Prec': f"{precision_score(val_labels, val_preds, average='macro', zero_division=0):.4f}",
                    'Rec': f"{recall_score(val_labels, val_preds, average='macro', zero_division=0):.4f}"
                })

        val_acc = accuracy_score(val_labels, val_preds)
        val_prec = precision_score(val_labels, val_preds, average='macro', zero_division=0)
        val_rec = recall_score(val_labels, val_preds, average='macro', zero_division=0)
        avg_val_loss = val_loss / len(valid_loader)
        print(f"✅ Validation Summary: Loss = {avg_val_loss:.4f}, Acc = {val_acc:.4f}, Prec = {val_prec:.4f}, Rec = {val_rec:.4f}\n")
        
        valid_logs.append({
            "Epoch": epoch + 1,
            "Val Loss": val_loss / len(valid_loader),
            "Val Accuracy": val_acc,
            "Val Precision": val_prec,
            "Val Recall": val_rec
        })
    
    return train_logs, valid_logs


def test_multifeatures(model, test_loader, criterion, device):
    test_preds, test_labels = [], []
    test_loss = 0.0
    
    with torch.no_grad():
        for f1, f2, f3, labels in test_loader:
            f1, f2, f3, labels = f1.to(device), f2.to(device), f3.to(device), labels.to(device)

            outputs = model(f1, f2, f3)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    test_acc = accuracy_score(test_labels, test_preds)
    test_prec = precision_score(test_labels, test_preds, average='macro', zero_division=0)
    test_rec = recall_score(test_labels, test_preds, average='macro', zero_division=0)

    return test_labels, test_preds, test_loss, test_acc, test_prec, test_rec

def test_doublefeatures(model, test_loader, criterion, device):
    test_preds, test_labels = [], []
    test_loss = 0.0
    
    with torch.no_grad():
        for f1, f2, labels in test_loader:
            f1, f2, labels = f1.to(device), f2.to(device), labels.to(device)

            outputs = model(f1, f2)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    test_acc = accuracy_score(test_labels, test_preds)
    test_prec = precision_score(test_labels, test_preds, average='macro', zero_division=0)
    test_rec = recall_score(test_labels, test_preds, average='macro', zero_division=0)

    return test_labels, test_preds, test_loss, test_acc, test_prec, test_rec

def test_singlefeature(model, test_loader, criterion, device):
    test_preds, test_labels = [], []
    test_loss = 0.0
    
    test_progress_bar = tqdm(test_loader, desc=f"Testing the model", leave=True)
    with torch.no_grad():
        for images, labels in test_progress_bar:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    test_acc = accuracy_score(test_labels, test_preds)
    test_prec = precision_score(test_labels, test_preds, average='macro', zero_division=0)
    test_rec = recall_score(test_labels, test_preds, average='macro', zero_division=0)
    
    return test_labels, test_preds, test_loss, test_acc, test_prec, test_rec

def test_model(model, num_features, test_loader, criterion, device):
    test_results = []
    print("Start to test the model")
    test_progress_bar = tqdm(test_loader, desc=f"Testing the model", leave=True)
    
    if num_features == 1:
        test_labels, test_preds, test_loss, test_acc, test_prec, test_rec = test_singlefeature(
            model=model, 
            test_loader=test_progress_bar, 
            criterion=criterion, 
            device=device
        )
        
        test_results.append({
            "Test Accuracy": test_acc,
            "Test Precision": test_prec,
            "Test Recall": test_rec
        })

    elif num_features == 2:
        test_labels, test_preds, test_loss, test_acc, test_prec, test_rec = test_doublefeatures(
            model=model, 
            test_loader=test_progress_bar, 
            criterion=criterion, 
            device=device
        )

        test_results.append({
            "Test Accuracy": test_acc,
            "Test Precision": test_prec,
            "Test Recall": test_rec
        })

    else:
        test_labels, test_preds, test_loss, test_acc, test_prec, test_rec = test_multifeatures(
            model=model, 
            test_loader=test_progress_bar, 
            criterion=criterion, 
            device=device
        )

        test_results.append({
            "Test Accuracy": test_acc,
            "Test Precision": test_prec,
            "Test Recall": test_rec
        })
    
    print(f"\n🏁 Test Set Evaluation Result:")
    print(f"Loss = {test_loss / len(test_loader):.4f}, Accuracy = {test_acc:.4f}, Precision = {test_prec:.4f}, Recall = {test_rec:.4f} ")
    
    return test_labels, test_preds, test_results

def run(args):
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
    dir()