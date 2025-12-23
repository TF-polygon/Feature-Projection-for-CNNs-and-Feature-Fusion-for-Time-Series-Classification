import torch
import os
import numpy as np

from glob import glob
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from tqdm import tqdm

class MultiFeatureFusionDataset(Dataset):
    def __init__(self, root_dir, class_to_idx, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.class_to_idx = class_to_idx
        self.data_list = self._make_data_list()

        if not self.data_list:
            raise RuntimeError(f"Error :: Not found a data in Path: {root_dir}. Check the directory structure.")
        
    def _make_data_list(self):
        data_list = []

        classes = list(self.class_to_idx.keys())

        for class_name in classes:
            class_idx = self.class_to_idx[class_name]
            dir = os.path.join(self.root_dir, 'GASF', class_name)
            files = sorted(glob(os.path.join(dir, '*.png')))

            for path in files:
                filename = os.path.basename(path)

                gadf_path = os.path.join(self.root_dir, 'GADF', class_name, filename)
                rp_path = os.path.join(self.root_dir, 'RP', class_name, filename)

                if os.path.exists(gadf_path) and os.path.exists(rp_path):
                    data_list.append({
                        'gasf': path,
                        'gadf': gadf_path,
                        'rp': rp_path,
                        'label': class_idx
                    })

        return data_list

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        item = self.data_list[index]

        img_gasf = Image.open(item['gasf']).convert('RGB') # L
        img_gadf = Image.open(item['gadf']).convert('RGB')
        img_rp = Image.open(item['rp']).convert('RGB')

        label = item['label']

        if self.transform:
            img_gasf = self.transform(img_gasf)
            img_gadf = self.transform(img_gadf)
            img_rp = self.transform(img_rp)

        return img_gasf, img_gadf, img_rp, torch.tensor(label, dtype=torch.long)

class DoubleFeatureFusionDataset(Dataset):
    def __init__(self, root_dir, f1, f2, class_to_idx, transform=None):
        self.root_dir = root_dir
        self.f1 = f1
        self.f2 = f2
        self.transform = transform
        self.class_to_idx = class_to_idx
        self.data_list = self._make_data_list()

        if not self.data_list:
            raise RuntimeError(f"Error :: Not found a data in Path: {root_dir}. Check the directory structure.")
        
    def _make_data_list(self):
        data_list = []

        classes = list(self.class_to_idx.keys())

        for class_name in classes:
            class_idx = self.class_to_idx[class_name]
            dir = os.path.join(self.root_dir, self.f1, class_name)
            files = sorted(glob(os.path.join(dir, '*.png')))

            for path in files:
                filename = os.path.basename(path)
                f2_path = os.path.join(self.root_dir, self.f2, class_name, filename)
                
                if os.path.exists(f2_path):
                    data_list.append({
                        'f1': path,
                        'f2': f2_path,
                        'label': class_idx
                    })

        return data_list

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        item = self.data_list[index]

        img_f1 = Image.open(item['f1']).convert('RGB') # L
        img_f2 = Image.open(item['f2']).convert('RGB')

        label = item['label']

        if self.transform:
            img_f1 = self.transform(img_f1)
            img_f2 = self.transform(img_f2)
        else:
            img1 = transforms.ToTensor()(img1)
            img2 = transforms.ToTensor()(img2)

        return img_f1, img_f2, torch.tensor(label, dtype=torch.long)
    
class MultiFeatureNPZdataset(Dataset):
    def __init__(self, npz_path, transform=None):
        self.transform = transform
        data = np.load(npz_path)

        self.gasf_data = data['gasf']
        self.gadf_data = data['gadf']
        self.rp_data = data['rp']
        self.labels = data['labels']

        self.num_samples = len(self.labels)

        print(f"Loaded NPZ Dataset: {self.num_samples} samples.")

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index):
        img_gasf = transforms.ToPILImage()(self.gasf_data[index])
        img_gadf = transforms.ToPILImage()(self.gadf_data[index])
        img_rp = transforms.ToPILImage()(self.rp_data[index])
        
        label = self.labels[index]

        if self.transform:
            img_gasf = self.transform(img_gasf)
            img_gadf = self.transform(img_gadf)
            img_rp = self.transform(img_rp)

        return img_gasf, img_gadf, img_rp, torch.tensor(label, dtype=torch.long)

class DoubleFeatureNPZdataset(Dataset):
    def __init__(self, npz_path, transform=None):
        self.transform = transform
        data = np.load(npz_path)
        self.f1 = data['f1']
        self.f2 = data['f2']
        self.labels = data['labels']
        self.num_samples = len(self.labels)
        print(f"Loaded NPZ Dataset: {self.num_samples} samples.")

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index):
        img1 = transforms.ToPILImage()(self.f1[index])
        img2 = transforms.ToPILImage()(self.f2[index])
        label = self.labels[index]

        if self.transform:
            img1 = self.transform(img1)            
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(label, dtype=torch.long)
    
def serialize_doublefeature(root_dir, output_npz_path, input_feature1, input_feature2):
    print(f"Search the dataset path: {root_dir}")
    class_to_idx = {
        'Downward': 0,
        'Sideway': 1,
        'Upward': 2
    }
    dataset_temp = DoubleFeatureFusionDataset(root_dir, input_feature1, input_feature2, class_to_idx)
    data_list = dataset_temp.data_list

    if not data_list:
        print("Warning: There is no data to serialize.")
        return
    
    f1_array = []
    f2_array = []
    labels = []

    print(f"Start to serialize {len(data_list)} samples...")

    for item in tqdm(data_list, desc='Serializing'):
        img1 = Image.open(item['f1']).convert('RGB')
        img2 = Image.open(item['f2']).convert('RGB')

        arr_img1 = np.array(img1, dtype=np.uint8)
        arr_img2 = np.array(img2, dtype=np.uint8)

        f1_array.append(arr_img1)
        f2_array.append(arr_img2)
        labels.append(item['label'])
    
    final_images1 = np.array(f1_array)
    final_images2 = np.array(f2_array)
    final_labels = np.array(labels, dtype=np.int64)

    np.savez_compressed(
        output_npz_path,
        f1=final_images1,
        f2=final_images2,
        labels=final_labels
    )
    
    print(f"Complete to serialize. Path to save the file: {output_npz_path}")    

def serialize_multifeature(root_dir, output_npz_path):
    print(f"Search the dataset path: {root_dir}")
    class_to_idx = {
        'Downward': 0,
        'Sideway': 1,
        'Upward': 2
    }
    dataset_temp = MultiFeatureFusionDataset(root_dir, class_to_idx)
    data_list = dataset_temp.data_list
    
    if not data_list:
        print("Warning: There is no data to serialize.")
        return

    gasf_arrays = []
    gadf_arrays = []
    rp_arrays = []
    labels = []
    
    print(f"Start to serialize {len(data_list)} samples...")
    
    for item in tqdm(data_list, desc="Serializing"):
        img_gasf = Image.open(item['gasf']).convert('RGB')
        img_gadf = Image.open(item['gadf']).convert('RGB')
        img_rp = Image.open(item['rp']).convert('RGB')

        arr_gasf = np.array(img_gasf, dtype=np.uint8)
        arr_gadf = np.array(img_gadf, dtype=np.uint8)
        arr_rp = np.array(img_rp, dtype=np.uint8)

        gasf_arrays.append(arr_gasf)
        gadf_arrays.append(arr_gadf)
        rp_arrays.append(arr_rp)
        labels.append(item['label'])

    final_gasf = np.array(gasf_arrays)
    final_gadf = np.array(gadf_arrays)
    final_rp = np.array(rp_arrays)
    final_labels = np.array(labels, dtype=np.int64)
    
    np.savez_compressed(
        output_npz_path,
        gasf=final_gasf,
        gadf=final_gadf,
        rp=final_rp,
        labels=final_labels
    )
    
    print(f"Complete to serialize. Path to save the file: {output_npz_path}")

def main(args):    
    if args.num_features == 2:
        print('Start to serialize for double features')
        serialize_doublefeature(args.dataset, args.path, args.f1, args.f2)
    else:
        print('Start to serialize for multi features')
        serialize_multifeature(args.dataset, args.path)