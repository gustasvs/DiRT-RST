import os
import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle
from functions.settings import *
import cv2

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, WeightedRandomSampler

from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import Compose, ToTensor, Normalize

def prepare_image(image):
    """
        Transforms input image to tensor and normalizes it
        Final shape: (C x H x W) in the range [0.0, 1.0]
    """
    # are first rescaled to [0.0, 1.0] and then normalized using mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].
    def _normalize(img):
        if GRAYSCALE:
            return Normalize(mean=[0.485], std=[0.229])(img)
        else:
            return Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

    torch_transform = Compose([
        # converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
        ToTensor(),
        _normalize
    ])
    return torch_transform(image)

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.prepared_data = []
        
        self.prepare_data()

    def __len__(self):
        return len(self.prepared_data)

    def __getitem__(self, idx):
        return self.prepared_data[idx]

    def prepare_data(self):
        for idx in range(len(self.data)):
            self.prepared_data.append(self.prepare_item(idx))

    def prepare_item(self, idx):
        input = self.data[idx][0]

        target = self.data[idx][1]
        target = torch.tensor(target, dtype=torch.float32, device=DEVICE)

        if TEMPORAL_FRAME_WINDOW == 1:
            input = prepare_image(input).to(DEVICE)
        else:
            input = torch.stack([prepare_image(frame).to(DEVICE) for frame in input], dim=0)

        # cv2.imshow(f'targets: {target}', cv2.resize(input.cpu().detach().numpy().transpose(1, 2, 0), (700, 700)))
        # cv2.waitKey(0)
        # exit(0)

        # TEMPORAL_FRAME_WINDOW > 1 - output shape is (num_frames, C, H, W)
        # TEMPORAL_FRAME_WINDOW = 1 - output shape is (C, H, W)
        return input, target
    

def get_dataloader():
    all_data = []
    current_file_index = 0

    while True:
        try:
            file = np.load(f'{DATA_PATH}_{current_file_index}.npy', allow_pickle=True)
            print(f'loaded {DATA_PATH}_{current_file_index}.npy')
            all_data += list(file)
            current_file_index += 1
        except:
            break

    assert len(all_data) > 0, "No data collected, please run `create_data.py` to collect data"

    df = pd.DataFrame(all_data)
    print(Counter(df[1].apply(str)))
    print(f'total data  --> {len(all_data)}')

    class_counts = Counter(df[1].apply(str))
    sorted_labels = sorted(class_counts.keys(), reverse=True)
    pos_weight = []
    for label in sorted_labels:
        print(f'{label}: {class_counts[label]}')
        count = class_counts[label]
        neg_count = len(all_data) - count
        pos_weight.append(neg_count / count)
    
    print(f'pos_weight: {pos_weight}')
    
    pos_weight = torch.tensor(pos_weight, dtype=torch.float32, device=DEVICE)

    custom_dataset = CustomDataset(all_data)

    # class_counts = Counter(df[1].apply(str))
    # num_samples = len(custom_dataset)
    # class_weights = [num_samples/class_counts[i] for i in range(len(class_counts))]
    # weights = [class_weights[i] for i in range(len(custom_dataset))]
    
    dataloader = DataLoader(custom_dataset, 
                            batch_size=BATCH_SIZE, 
                            # sampler=WeightedRandomSampler(weights=weights, num_samples=len(custom_dataset), replacement=True),
                            shuffle=True)
    
    return dataloader, pos_weight

if __name__ == "__main__":
    example_dataloader = get_dataloader(file_name, 32)
    # log first sample in first batch
    for i, (inputs, targets) in enumerate(example_dataloader):

        print("inputs shape: ", inputs.shape)

        cv2.imshow(f'targets: {targets[0]}', inputs[0].numpy())
        print(targets[0])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        break
    





