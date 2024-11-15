import os, numpy as np, PIL, albumentations as A
from os.path import join, dirname, basename
from glob import glob
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms, datasets
from albumentations.pytorch import ToTensorV2

class DefectDatasetV3(datasets.ImageFolder):
    def __init__(self, data_dir:str, image_size:int, transform_mode:str, normal_classes: list, abnormal_classes: list=[], in_channels: int=1):
        # Define transforms for preprocessing
        if 'test' in transform_mode:
            target_data_dir = os.path.join(data_dir, 'test')
            transforms = A.Compose([
                A.Resize(image_size, image_size),
                A.Normalize(mean=0.5, std=0.5),
                ToTensorV2()
            ])

            # Set dataset: Set to dataset if the Class folder is included in normal or Abnormal case
            super().__init__(target_data_dir, transform=transforms)
            test_samples = [item for item in self.imgs if basename(dirname(item[0])) in normal_classes + abnormal_classes]

            train_data_dir = os.path.join(data_dir, 'train')
            # Load additional `abnormal_classes` samples from `train` folder
            train_folder = datasets.ImageFolder(train_data_dir)
            train_samples = [item for item in train_folder.imgs if basename(dirname(item[0])) in abnormal_classes]
            
            # Combine samples
            self.samples = test_samples + train_samples
            self.classes = [cls for cls in self.classes if cls in normal_classes + abnormal_classes]    # Overl Class Information
            if in_channels == 1:
                self.samples_image = [np.array(self.loader(item[0]).convert("L")) for item in self.imgs+train_folder.imgs if basename(dirname(item[0])) in normal_classes + abnormal_classes]
            else:
                self.samples_image = [np.array(self.loader(item[0])) for item in self.imgs+train_folder.imgs if basename(dirname(item[0])) in normal_classes + abnormal_classes]
        else:
            raise KeyError(f"Unknown transform mode: {transform_mode}")

        # Find Normal / Abnormal Class IDX
        self.normal_idx = []
        self.abnormal_idx = []
        self.class_to_idx = {k:v for v, k in enumerate(self.classes)}
        for k, v in self.class_to_idx.items():
            if k in abnormal_classes:
                self.abnormal_idx.append(self.classes.index(k))
            else:
                self.normal_idx.append(self.classes.index(k))

    def __getitem__(self, index):
        path, target = self.samples[index]

        class_name = path.split('/')[-2]
        target = self.class_to_idx[class_name]
        
        sample = self.samples_image[index] #np.array(self.loader(path).convert("L"))
        sample = self.transform(image=sample)
        name = path
        return sample['image'], target, name    
