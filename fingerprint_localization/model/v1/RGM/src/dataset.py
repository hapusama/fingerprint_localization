import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from scipy.stats import truncnorm
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from collections import Counter


class ComplexDataset(Dataset):

    def __init__(self, 
                 amplitude_list, 
                 phase_list, 
                 labels):
        assert len(amplitude_list) == len(phase_list) == len(labels), "Mismatch in list lengths"
        self.amplitude_list = amplitude_list
        self.phase_list = phase_list
        self.labels = labels


    def __len__(self):
        return len(self.labels)


    def __getitem__(self, idx):
        amplitude = self.amplitude_list[idx]
        # phase = self.phase_list[idx]
        label = self.labels[idx]
        # real = amplitude * torch.cos(phase)
        # imaginary = amplitude * torch.sin(phase)
        
        # stacks the real and imaginary tensors along a new dimension, 
        # resulting in a new tensor with shape (2, 16).
        # complex_number = torch.stack((real, imaginary), dim=0)
        return amplitude, label


class ComplexDatasetLocs(Dataset):

    def __init__(self, 
                 rssi_list, 
                 snr_list, 
                 labels, 
                 label_feature_path 
                 ):
        assert len(rssi_list) == len(snr_list) == len(labels), "Mismatch in list lengths"

        self.rssi_list =rssi_list
        self.snr_list = snr_list
        self.labels = labels
        # 读取位置特征
        self.label_features = self.load_label_features(label_feature_path)
        # 检查位置特征是否和标签匹配38
        unique_ids = torch.unique(self.labels)
        # if len(self.label_features) != len(unique_ids):
        #     # 打印两者的label
        #     print(f"Label features: {list(self.label_features.keys())}")
        #     print(f"Unique labels: {unique_ids}")
        #     raise ValueError(f"Mismatch between label_features and labels. Only {len(self.label_features)} location feature were obtained, but there are a total of {len(unique_ids)} location IDs. \n\nPlease make sure to finish running '0_location_representation.py' before running this script.\n")        


    # 读取位置特征，如果为空就不要返回了
    def load_label_features(self, path):
        label_features = {}
        df = pd.read_csv(path)
        base_columns = ['x', 'y', 'distance', 'wall_nums', 'window', 'floor']
        extra_columns = [
            column for column in df.columns
            if column.startswith('multipath_') and column.endswith('_norm')
        ]
        feature_columns = base_columns + extra_columns

        for _, row in df.iterrows():
            if pd.notnull(row['x']) and pd.notnull(row['y']) and pd.notnull(row['distance']):
                mapped_id = int(row['idx'])
                values = pd.to_numeric(row[feature_columns], errors='coerce').fillna(0.0).values
                vector = torch.tensor(values, dtype=torch.float32)
                label_features[mapped_id] = vector
         
        return label_features



    def __len__(self):
        return len(self.labels)


    def __getitem__(self, idx):
        # rssi等传list的话stack之后cplx会变成（2，input_dim）的形状)
        # 但是这里的rssi和snr是一个数值，所以stack之后是(2,)的形状
        rssi = self.rssi_list[idx]
        snr = self.snr_list[idx]
        label = self.labels[idx]
        real = rssi
        imaginary = snr
        # 以上都不能为空
        # if real is None or imaginary is None or label is None: 
        #     raise ValueError(f"datasetloc part is None at index {idx}.")
        # complex_number = torch.stack((real, imaginary), dim=0)
        label_feature = self.label_features.get(label.item())
        if label_feature is None:
            raise KeyError(f"Missing label feature for label {label.item()}")
        # 将label_feature和snr cat在一起
        # 将label_feature和snr cat在一起
        label_feature = torch.cat((label_feature, snr), dim=0)        # 确保返回的值都是一维张量
        return rssi, label_feature, label


class ShiftedDataset(Dataset):

    def __init__(self, base_dataset):
        self.base_dataset = base_dataset


    def __getitem__(self, idx):
        data, label, label_int = self.base_dataset[idx]

        return data, label, label_int + 1   # 将标签从 0开始 改为从 1 开始 

    def __len__(self):
        return len(self.base_dataset)


class ComplexDataset_real_imagary_v2(Dataset):
    
    def __init__(self, 
                 features, 
                 features_label, 
                 labels, 
                 labels_int):
        assert len(features) == len(features_label) == len(labels) == len(labels_int), "Mismatch in list lengths"
        self.features = features
        self.features_label = features_label 
        self.labels = labels
        self.labels_int = labels_int


    def __len__(self):
        return len(self.labels)


    def __getitem__(self, idx):
        label = self.labels[idx]
        label_int = self.labels_int[idx]

        complex_number = self.features[idx]
        complex_number_label = self.features_label[idx]

        return complex_number, complex_number_label, label, label_int


def generate_three_loader_v2(dataset_t, 
                             n_batch_size, 
                             valid_size, 
                             test_size):
    
    labels = [label for _, label in dataset_t]
    
    indices = np.arange(len(dataset_t))
    
    train_temp_indices, test_indices = train_test_split(
        indices, test_size=test_size, stratify=labels, random_state=42)
    
    train_temp_labels = [labels[i] for i in train_temp_indices]
    
    train_indices, valid_indices = train_test_split(
        train_temp_indices, test_size=valid_size, stratify=train_temp_labels, random_state=42)
    
    # 创建分集
    train_dataset = Subset(dataset_t, train_indices)
    valid_dataset = Subset(dataset_t, valid_indices)
    test_dataset = Subset(dataset_t, test_indices)
    
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=n_batch_size, shuffle=True)
    valid_data_loader = DataLoader(dataset=valid_dataset, batch_size=len(valid_indices), shuffle=False)
    test_data_loader  = DataLoader(dataset=test_dataset, batch_size=len(test_indices), shuffle=False)
    
    print(f'Loaded {len(dataset_t)} samples, split {len(train_indices)}/{len(valid_indices)}/{len(test_indices)} for train/valid/test.')
    
    return train_data_loader, valid_data_loader, test_data_loader


def generate_three_loader_v3(dataset_t, 
                             n_batch_size, 
                             valid_size, 
                             test_size):
    labels = [label for _, _, _, label in dataset_t]
    
    indices = np.arange(len(dataset_t))
    
    train_temp_indices, test_indices = train_test_split(
        indices, test_size=test_size,random_state=42)
    
    train_temp_labels = [labels[i] for i in train_temp_indices]
    
    train_indices, valid_indices = train_test_split(
        train_temp_indices, test_size=valid_size, stratify=train_temp_labels, random_state=42)
    
    # 创建分集
    train_dataset = Subset(dataset_t, train_indices)
    valid_dataset = Subset(dataset_t, valid_indices)
    test_dataset = Subset(dataset_t, test_indices)
    
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=n_batch_size, shuffle=True)
    valid_data_loader = DataLoader(dataset=valid_dataset, batch_size=len(valid_indices), shuffle=False)
    test_data_loader  = DataLoader(dataset=test_dataset, batch_size=len(test_indices), shuffle=False)
    
    print(f'Loaded {len(dataset_t)} samples, split {len(train_indices)}/{len(valid_indices)}/{len(test_indices)} for train/valid/test.')
    
    return train_data_loader, valid_data_loader, test_data_loader


def generate_three_dataset_v2(dataset_t, 
                              valid_size, 
                              test_size):
    #complex number,label_feature(7维向量),label
    labels = [label for _, _, label in dataset_t]
    
    indices = np.arange(len(dataset_t))
    
    train_temp_indices, test_indices = train_test_split(
        indices, test_size=test_size, stratify=labels, random_state=42)
    
    train_temp_labels = [labels[i] for i in train_temp_indices]
    
    train_indices, valid_indices = train_test_split(
        train_temp_indices, test_size=valid_size, stratify=train_temp_labels, random_state=42)
    
    train_dataset = Subset(dataset_t, train_indices)
    valid_dataset = Subset(dataset_t, valid_indices)
    test_dataset = Subset(dataset_t, test_indices)

    print(f'Loaded {len(dataset_t)} samples, split {len(train_indices)}/{len(valid_indices)}/{len(test_indices)} for train/valid/test.')
    
    return train_dataset, valid_dataset, test_dataset


def generate_three_dataset_v3(dataset_t, 
                              ratios, 
                              valid_r, 
                              test_r):

    # labels需要是个list或者tensor数组
    labels = np.array([label for _, _, label in dataset_t])
    unique_labels = np.unique(labels)
    num_selected_labels = int(len(unique_labels) * ratios)
    selected_label_ids = np.random.choice(unique_labels, num_selected_labels, replace=False)
    print(f"Selected label IDs: {selected_label_ids}")
    # 筛选选定标签ID的索引
    filtered_indices = [i for i, label in enumerate(labels) if label in selected_label_ids]
    filtered_labels = labels[filtered_indices]

    # 将筛选后的索引转换为 NumPy 数组，以兼容train_test_split函数的stratify参数
    filtered_indices = np.array(filtered_indices)
    
    # 基于筛选后的数据集，计算验证集和测试集的实际大小。
    total_size = len(filtered_indices)
    test_size = int(total_size * test_r)
    valid_size = int(total_size * valid_r)

    # 分层划分以确保每个标签 ID 的代表性均等
    train_temp_indices, test_indices = train_test_split(
        filtered_indices, test_size=test_size, stratify=filtered_labels, random_state=42)
    
    # 更新用于验证的分层划分标签
    train_temp_labels = filtered_labels[[np.where(filtered_indices == i)[0][0] for i in train_temp_indices]]
    
    train_indices, valid_indices = train_test_split(
        train_temp_indices, test_size=valid_size, stratify=train_temp_labels, random_state=42)
    print("train_len, valid_len, test_len: ", len(train_indices), len(valid_indices), len(test_indices))

    # 为训练集、验证集和测试集创建子集
    train_dataset = Subset(dataset_t, train_indices)
    valid_dataset = Subset(dataset_t, valid_indices)
    test_dataset = Subset(dataset_t, test_indices)

    return train_dataset, valid_dataset, test_dataset


def generate_three_dataset_v4(dataset_t, 
                              ratios, 
                              valid_r, 
                              test_r):
    
    labels = np.array([label for _, _, label in dataset_t])
    unique_labels = np.unique(labels)

    # Calculate the number of labels to select based on ratios
    num_selected_labels = int(len(unique_labels) * ratios)

    # Randomly select label IDs for the first part
    selected_label_ids = np.random.choice(unique_labels, num_selected_labels, replace=False)

    # Identify remaining label IDs for the second part
    remaining_label_ids = np.setdiff1d(unique_labels, selected_label_ids)

    # Helper function to split dataset based on label IDs
    def split_dataset_by_labels(label_ids):
        filtered_indices = [i for i, label in enumerate(labels) if label in label_ids]
        filtered_labels = labels[filtered_indices]

        # Calculate actual sizes for validation and test sets based on the filtered dataset
        total_size = len(filtered_indices)
        test_size = int(total_size * test_r)
        valid_size = int(total_size * valid_r)

        # Stratified split to ensure equal representation of each label ID
        train_temp_indices, test_indices = train_test_split(
            filtered_indices, test_size=test_size, stratify=filtered_labels, random_state=42)
        
        train_temp_labels = filtered_labels[[np.where(filtered_indices == i)[0][0] for i in train_temp_indices]]
        
        train_indices, valid_indices = train_test_split(
            train_temp_indices, test_size=valid_size, stratify=train_temp_labels, random_state=42)

        # Create subsets for train, validation, and test datasets
        train_dataset = Subset(dataset_t, train_indices)
        valid_dataset = Subset(dataset_t, valid_indices)
        test_dataset = Subset(dataset_t, test_indices)

        return train_dataset, valid_dataset, test_dataset

    # Split datasets for selected and remaining label IDs
    train_dataset_1, valid_dataset_1, test_dataset_1 = split_dataset_by_labels(selected_label_ids)
    train_dataset_2, valid_dataset_2, test_dataset_2 = split_dataset_by_labels(remaining_label_ids)


    return train_dataset_1, valid_dataset_1, test_dataset_1, train_dataset_2, valid_dataset_2, test_dataset_2



