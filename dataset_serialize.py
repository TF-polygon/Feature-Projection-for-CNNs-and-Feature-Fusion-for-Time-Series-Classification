import os
import numpy as np
import sys
import argparse
from glob import glob
from dataset import MultiFeatureFusionDataset, DoubleFeatureFusionDataset
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import transforms

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--num_features', type=int, required=True, help='if num_features for multi features is 3. The serialization for double features is 2.')
    parser.add_argument('--dataset', type=str, required=True, help='path to dataset')
    parser.add_argument('--path', type=str, required=True, help='save to path')
    parser.add_argument('--f1', type=str, help='Feature Projection Method ex) GASF, GADF, RP')
    parser.add_argument('--f2', type=str, help='Feature Projection Method ex) GASF, GADF, RP')
    args = parser.parse_args()

    main(args)