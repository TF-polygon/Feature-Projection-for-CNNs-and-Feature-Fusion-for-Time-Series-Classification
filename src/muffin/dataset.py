import torch
import os
import numpy as np
from glob import glob
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

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