import torch
import copy
import os
import random
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as tfm
from imageio import imread
from skimage.color import rgb2hsv, hsv2rgb

# 1. ĐÃ ĐỔI TÊN CHUẨN KHỚP 100% VỚI 15 THƯ MỤC CỦA BẠN
labels_map = {
    'BAS': 0, 
    'EBO': 1, 
    'EOS': 2, 
    'KSC': 3, 
    'LYA': 4,
    'LYT': 5, 
    'MMZ': 6, 
    'MOB': 7, 
    'MON': 8, 
    'MYB': 9,
    'MYO': 10, 
    'NGB': 11, 
    'NGS': 12, 
    'PMB': 13, 
    'PMO': 14
}

class DatasetMarr(Dataset): 
    def __init__(self, dataroot, dataset_selection, labels_map, fold, transform=None, state='train', is_hsv=False, is_hed=False):
        super(DatasetMarr, self).__init__()
        
        self.dataroot = os.path.join(dataroot, '')  

        # Trỏ vào file CSV danh bạ
        metadata_path = os.path.join(self.dataroot, 'AML_metadata.csv') 
        try:
            metadata = pd.read_csv(metadata_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Không tìm thấy file CSV tại: {metadata_path}. Hãy chạy file make_csv.py trước.")

        set_fold = "kfold" + str(fold)  
        
        if isinstance(dataset_selection, list):
            dataset_index = metadata.dataset.isin(dataset_selection)
        else:
            dataset_index = metadata["dataset"] == dataset_selection

        # Lọc Train/Val/Test
        if state == 'train':
            dataset_index = dataset_index & metadata[set_fold].isin(["train"])
        elif state == 'validation':
            dataset_index = dataset_index & metadata[set_fold].isin(["val"])
        elif state == 'test':
            dataset_index = dataset_index & metadata[set_fold].isin(["test"])

        dataset_index = dataset_index[dataset_index].index
        self.metadata = metadata.loc[dataset_index, :].copy().reset_index(drop=True)
        self.labels_map = labels_map
        self.transform = transform
        self.is_hsv = is_hsv and random.random() < 0.33
        
        self.to_tensor = tfm.ToTensor()
        self.from_tensor = tfm.ToPILImage()

    def __len__(self):
        return len(self.metadata)
    
    def colorize(self, image):
        # Tăng cường độ nhiễu màu
        hue = random.choice(np.linspace(-0.1, 0.1))
        saturation = random.choice(np.linspace(-1, 1))
        hsv = rgb2hsv(image)
        hsv[:, :, 1] = saturation
        hsv[:, :, 0] = hue
        return hsv2rgb(hsv)            

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Đọc đường dẫn và mở ảnh
        file_path = self.metadata.loc[idx, "image"]
        image = imread(file_path)[:, :, [0, 1, 2]]
        
        # Lấy tên nhãn (VD: 'BAS') và tra từ điển ra số (VD: 0)
        label_name = self.metadata.loc[idx, "label"]
        label = self.labels_map[label_name]
        
        # Biến đổi màu sắc nếu trúng tỷ lệ 33%
        if self.is_hsv:
            image = self.colorize(image).clip(0., 1.)
        
        # Chuyển thành PIL Image
        img = self.to_tensor(copy.deepcopy(image))
        image = self.from_tensor(img)
        
        # 2. ĐÃ XÓA PHẦN CẮT 345x345. 
        # Hàm transform này (từ file Train truyền vào) sẽ tự động bóp ảnh về chuẩn 224x224 cho Kansformer
        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor(label).long()
        
        return image, label