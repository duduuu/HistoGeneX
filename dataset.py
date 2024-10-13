import os
import cv2
import config as CFG

from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

class MAIDataset(Dataset):
    def __init__(self, img_path_list, label_list, augment=False):
        self.img_path_list = img_path_list
        self.label_list = label_list
        
        if augment == False:
            self.transforms = base_transform
        else:
            self.transforms = aug_transform
        
    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        
        image = cv2.imread(os.path.join('data', img_path))
        
        if self.transforms is not None:
            image = self.transforms(image=image)['image']
        
        if self.label_list is not None:
            label = self.label_list[index]
            return image, label
        else:
            return image
        
    def __len__(self):
        return len(self.img_path_list)
    
    
base_transform = A.Compose([
                            A.Resize(CFG.img_size, CFG.img_size),
                            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                            ToTensorV2()
                            ])

aug_transform = A.Compose([
                            A.Transpose(p=0.5),
                            A.VerticalFlip(p=0.5),
                            A.HorizontalFlip(p=0.5),
                            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.1, p=0.5),
                            A.OneOf(
                                [
                                    A.MotionBlur(blur_limit=5),
                                    A.MedianBlur(blur_limit=5),
                                    A.GaussianBlur(blur_limit=5),
                                    A.GaussNoise(var_limit=(5.0, 30.0)),
                                ],
                                p=0.75,
                            ),
                            A.CLAHE(clip_limit=4.0, p=0.5),
                            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
                            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.5),
                            A.Resize(CFG.img_size, CFG.img_size),
                            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                            ToTensorV2()
                            ])