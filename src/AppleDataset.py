import os
import numpy as np
from torch.utils.data import Dataset
import random
import cv2
import pandas as pd
from tqdm import tqdm
import torch

class AppleDataset(Dataset):
    def __init__(self, images_dir, labels_csv, crop_size=(3, 224, 224), stride=224, label_col = ['TSS_brix', 'Firmness', 'Titerabile acid percentage']):
        self.images_dir = images_dir
        self.c, self.h, self.w  = crop_size
        self.stride = stride

        self.labels_df = pd.read_csv(labels_csv)
        # label_col = ['TSS_brix', 'Firmness', 'Titerabile acid percentage']   # change this if your column name differs
        mean_value = self.labels_df[label_col].mean()
        self.labels_df[label_col] = self.labels_df[label_col] - mean_value

        # List all images
        self.images = os.listdir(images_dir)

        self.img_data = []
        self.img_labels = []

        for image in tqdm(self.images, desc="Loading dataset"):
            img_path = os.path.join(images_dir, image)
            img = cv2.imread(img_path)

            if image.startswith(".") or img is None:
                print(f"⚠️ Warning: Could not read {image}, skipping.")
                continue

            self.H, self.W, self.C = img.shape


            # Normalize and convert to CHW
            img = np.float32(img) / 255.0
            img = np.transpose(img, (2, 0, 1))

            # Match label
            label_row = self.labels_df[self.labels_df['filenames'] == image]
            label = label_row[label_col].values.squeeze()


            for y in range(0, self.H - self.h + 1, self.stride):
                for x in range(0, self.W - self.w + 1, self.stride):
                    cropped_image = img[:, y:y + self.h, x:x + self.w]
                    self.img_data.append(cropped_image)
                    self.img_labels.append(label)


    def __len__(self):
        return len(self.img_data)

    def augment(self, img):
        """Apply random rotation and flips."""
        # Random rotation (0, 90, 180, 270)
        rotTimes = random.randint(0, 3)
        if rotTimes > 0:
            img = np.rot90(img.copy(), k=rotTimes, axes=(1, 2))

        # Random vertical flip
        if random.random() < 0.5:
            img = np.flip(img, axis=1).copy()  # flip along H

        # Random horizontal flip
        if random.random() < 0.5:
            img = np.flip(img, axis=2).copy()  # flip along W

        # ✅ ensure contiguous memory (fixes negative stride issue)
        img = np.ascontiguousarray(img)
        return img

    def __getitem__(self, idx):
        img = self.img_data[idx]
        label = self.img_labels[idx]

        img = self.augment(img)

        # Convert to torch tensors safely
        img_tensor = torch.from_numpy(np.array(img, copy=True))  # avoids negative stride issue
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return img_tensor, label_tensor



class ValidDataset(Dataset):
    def __init__(self, images_dir, labels_csv, crop_size=(3, 224, 224), stride=224, label_col = ['TSS_brix', 'Firmness', 'Titerabile acid percentage']):
        self.images_dir = images_dir
        self.c, self.h, self.w  = crop_size
        self.stride = stride

        self.labels_df = pd.read_csv(labels_csv)
        # label_col = ['TSS_brix', 'Firmness', 'Titerabile acid percentage']   # change this if your column name differs
        mean_value = self.labels_df[label_col].mean()
        self.labels_df[label_col] = self.labels_df[label_col] - mean_value

        # List all images
        self.images = os.listdir(images_dir)

        self.img_data = []
        self.img_labels = []

        for image in tqdm(self.images, desc="Loading dataset"):
            img_path = os.path.join(images_dir, image)
            img = cv2.imread(img_path)

            if image.startswith(".") or img is None:
                print(f"⚠️ Warning: Could not read {image}, skipping.")
                continue

            self.H, self.W, self.C = img.shape


            # Normalize and convert to CHW
            img = np.float32(img) / 255.0
            img = np.transpose(img, (2, 0, 1))

            # Match label
            label_row = self.labels_df[self.labels_df['filenames'] == image]
            label = label_row[label_col].values.squeeze()


            # take center crop
            if self.H >= self.h and self.W >= self.w:
                top = (self.H - self.h) // 2
                left = (self.W - self.w) // 2
                crop = img[:, top:top+self.h, left:left+self.w]
                self.img_data.append(crop)
                self.img_labels.append(label)


    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, idx):
        img = self.img_data[idx]
        label = self.img_labels[idx]


        # Convert to torch tensors safely
        img_tensor = torch.from_numpy(np.array(img, copy=True))  # avoids negative stride issue
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return img_tensor, label_tensor
