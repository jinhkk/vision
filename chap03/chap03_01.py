from PIL import Image
import time
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2
import numpy as np



def cv_image_read(image_path):
    print(image_path)
    return cv2.imread(image_path)

def show_image(cv_image):
    rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB) # BGR 2 RGB
    plt.figure()
    plt.imshow(rgb) # 이미지 보여주기
    plt.show() # show

show_image(cv_image_read('./data/cat_2.jpg'))

# torchvision 사용 데이터셋 클래스 생성

class TorchvisionDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):  # 생성자
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)  # 파일의 양?

    def __getitem__(self, idx):  # 특정 인덱스의 파일, 라벨 가져오기
        label = self.labels[idx]
        file_path = self.file_paths[idx]
        # 이미지 읽기
        image = Image.open(file_path)
        # 이미지 변경 수행
        if self.transform:
            image = self.transform(image)
        return image, label

# Torchvision 이미지 변형 ( 사이즈 변경, 자르기, 수평 뒤집기 )
torchvision_transform = transforms.Compose([
    transforms.Resize((220, 220)),
    transforms.RandomCrop(120),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

# TorchvisionDataset 클래스 객체 생성
torchvision_dataset = TorchvisionDataset(
    file_paths=['./data/cat_2.jpg'],
    labels=[1],
    transform=torchvision_transform,
)

# 랜덤으로 2번 변형 수행
for i in range(2):
    sample, _ = torchvision_dataset[0]
    plt.figure()
    plt.imshow(transforms.ToPILImage()(sample))
    plt.show()

# Albumentations 패키지를 이용한 데이터 증강
import albumentations
import albumentations.pytorch

class AlbumentationsDataset(Dataset):
    """__init__ and __len__ functions are the same as in TorchvisionDataset"""
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        label = self.labels[idx]
        file_path = self.file_paths[idx]
        # 이미지 읽기
        image = cv2.imread(file_path)
        # BGR opencv 이미지를 RGB 이미지로 변경
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 이미지 변경 수행
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        return image, label

# albumentations 이미지 변형 (사이즈 변경, 자르기, 90도 회전, 수평 뒤집기, 가우시안 노이즈)
albumentations_transform = albumentations.Compose([
    albumentations.Resize(220, 220),
    albumentations.RandomCrop(120, 120),
    albumentations.RandomRotate90(p=1),
    albumentations.HorizontalFlip(),
    albumentations.GaussNoise(p=1),
    albumentations.pytorch.transforms.ToTensorV2()
])
#AlbumentationsDataset 클래스 객체 생성
albumentations_dataset = AlbumentationsDataset(
    file_paths=['./data/cat_2.jpg'],
    labels=[1],
    transform=albumentations_transform,
)


# 랜덤으로 2번 변형 수행
for i in range(2):
  sample, _ = albumentations_dataset[0]

  plt.figure()
  plt.imshow(transforms.ToPILImage()(sample))
  plt.show()