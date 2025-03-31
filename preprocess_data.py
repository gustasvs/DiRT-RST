import os
import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle
from functions.settings import *
import cv2
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, WeightedRandomSampler

from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import Compose, ToTensor, Normalize

def prepare_image(image, mean=0, std=0, normalize=True):
    """
        Transforms input image to tensor and normalizes it
        Final shape: (C x H x W) in the range [0.0, 1.0]
    """
    # are first rescaled to [0.0, 1.0] and then normalized using mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].
    def _normalize(img):

        if not normalize:
            return img

        if GRAYSCALE:
            return Normalize(mean=[mean], std=[std])(img)
        else:
            return Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

    # height = image.shape[0]; width = image.shape[1]
    # crop_amount = 0; crop_amount_width = 0
    # while True:
    #     if crop_amount > 0:
    #         cropped = image[crop_amount:height - crop_amount, :]
    #     else:
    #         cropped = image.copy()
    #     if crop_amount_width > 0:
    #         cropped = cropped[:, crop_amount_width:width - crop_amount_width]
    #     else:
    #         cropped = cropped.copy()
    #     window_title = f"Cropping: {crop_amount} | {crop_amount_width}"
    #     resized_cropped = cv2.resize(cropped, (cropped.shape[1] * 2, cropped.shape[0] * 2), interpolation=cv2.INTER_LINEAR)
    #     cv2.imshow(window_title, resized_cropped)
    #     key = cv2.waitKey(0) & 0xFF
    #     cv2.destroyAllWindows()
    #     if key == ord('n'):
    #         crop_amount += 10
    #         if crop_amount >= height:
    #             print("Reached image height, cannot crop further.")
    #             break
    #     if key == ord('b'):
    #         crop_amount -= 10
    #         if crop_amount < 0:
    #             crop_amount = 0
    #     if key == ord('v'):
    #         crop_amount_width += 10
    #         if crop_amount_width >= width:
    #             print("Reached image width, cannot crop further.")
    #             break
    #     if key == ord('c'):
    #         crop_amount_width -= 10
    #         if crop_amount_width < 0:
    #             crop_amount_width = 0

    #     elif key == ord('q'):
    #         break

    crop_top = 10
    crop_bottom = 65
    crop_left = 65
    crop_right = 65

    image = image[crop_top:image.shape[0] - crop_bottom, crop_left:image.shape[1] - crop_right]

    # image = cv2.resize(image, (MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT))
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

        self.mean = None
        self.std = None

        self.calculate_normalization_params()
        self.prepare_data()

    def __len__(self):
        offset = (TEMPORAL_FRAME_WINDOW * TEMPORAL_FRAME_GAP) - 1
        return len(self.prepared_data) - offset

    def __getitem__(self, idx):
        offset = (TEMPORAL_FRAME_WINDOW * TEMPORAL_FRAME_GAP) - 1

        real_idx = idx + offset

        if TEMPORAL_FRAME_WINDOW > 1:
            input_frames = []
            for i in range(0, TEMPORAL_FRAME_WINDOW * TEMPORAL_FRAME_GAP, TEMPORAL_FRAME_GAP):
                input_frames.append(self.prepared_data[real_idx - offset + i][0])

            input_tensor = torch.stack(input_frames, dim=0)
            target = self.prepared_data[real_idx][1]

            return input_tensor, target

        return self.prepared_data[idx]

    def calculate_normalization_params(self):
        means = []
        stds = []

        for idx in tqdm(range(len(self.data)), desc="Calculating normalization params"):
            input = self.data[idx][0]
            input = prepare_image(input, normalize=False)

            input = input.cpu().detach().numpy()
            input = input.transpose(1, 2, 0)

            means.append(np.mean(input))
            stds.append(np.std(input))

        dataset_mean = np.mean(means, axis=0)
        dataset_std = np.mean(stds, axis=0)

        print(f"MEAN: {dataset_mean} | STD: {dataset_std}")

        self.mean = dataset_mean
        self.std = dataset_std

    def prepare_data(self):

        for idx in tqdm(range(len(self.data)), desc="Preparing data"):
            self.prepared_data.append(self.prepare_item(idx))

    def prepare_item(self, idx):
        input = self.data[idx][0]

        # from matplotlib import pyplot as plt
        # fig = plt.figure(figsize=(20, 10))
        # for i, frame in enumerate(input):
        #     ax = fig.add_subplot(1, min(len(input), 3), i % 3 + 1)
        #     ax.imshow(prepare_image(frame).cpu().detach().numpy().transpose(1, 2, 0), cmap='gray')
        #     ax.set_title(f"{i}")
        #     ax.axis('off')
        #     ax.set_autoscale_on(True)
        #     if (i + 1) % 3 == 0 and i != len(input) - 1:
        #         fig = plt.figure(figsize=(20, 10))
        # ax = fig.add_subplot(1, min(len(input), 3), (len(input) % 3) + 1)
        # ax.imshow(prepare_image(self.data[idx][0]).cpu().detach().numpy().transpose(1, 2, 0), cmap='gray')
        # ax.set_title(f"ACTUAL")
        # ax.axis('off')
        # plt.show()

        target = self.data[idx][1]
        target = torch.tensor(target, dtype=torch.float32)

        input = prepare_image(input, self.mean, self.std, normalize=True)
        # if idx % 100 == 0:
        #     import matplotlib.pyplot as plt
        #     scaled_input = cv2.resize(input.cpu().detach().numpy().transpose(1, 2, 0), 
        #                               (input.shape[2] * 8, input.shape[1] * 8), 
        #                               interpolation=cv2.INTER_LINEAR)
        #     plt.imshow(scaled_input, cmap='gray')
        #     plt.axis('off')
        #     plt.show()
        
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

    # used for testing purposes
    if MAX_DATA_SAMPLES:
        all_data = all_data[len(all_data) - MAX_DATA_SAMPLES:]

    train_data = all_data[:int(len(all_data) * (1 - VALIDATION_SPLIT))]
    val_data = all_data[int(len(all_data) * (1 - VALIDATION_SPLIT)):]

    df = pd.DataFrame(train_data)
    print(f'TRAIN SAMPLES: {len(train_data)} | VAL SAMPLES: {len(val_data)}')

    # labels_array = np.array(df[1].tolist())  # shape: (num_samples, num_classes)
    # total_samples = labels_array.shape[0]
    # class_counts = labels_array.sum(axis=0)
    # pos_weight = (total_samples - class_counts) / class_counts
    # print('pos_weight:', pos_weight)
    class_counts = Counter(df[1].apply(str))
    sorted_labels = sorted(class_counts.keys(), reverse=True)
    pos_weight = []
    for label in sorted_labels:
        print(f'{label}: {class_counts[label]}')
        count = class_counts[label]
        neg_count = len(train_data) - count
        pos_weight.append(neg_count / count)
    print(f'pos_weight: {pos_weight}')
    
    pos_weight = torch.tensor(pos_weight, dtype=torch.float32, device=DEVICE)

    train_dataset = CustomDataset(train_data)
    val_dataset = CustomDataset(val_data)

    # class_counts = Counter(df[1].apply(str))
    # num_samples = len(custom_dataset)
    # class_weights = [num_samples/class_counts[i] for i in range(len(class_counts))]
    # weights = [class_weights[i] for i in range(len(custom_dataset))]
    
    train_dataloader = DataLoader(train_dataset, 
                            batch_size=BATCH_SIZE,
                            pin_memory=True,
                            # persistent_workers=True,
                            # num_workers=1,
                            # prefetch_factor=64,
                            shuffle=True
        )
    val_dataloader = DataLoader(val_dataset, 
                            batch_size=BATCH_SIZE,
                            pin_memory=True,
                            shuffle=True
        )
    
    return train_dataloader, val_dataloader, pos_weight

if __name__ == "__main__":
    example_dataloader = get_dataloader()
    # log first sample in first batch
    for i, batch in enumerate(example_dataloader):
        inputs, targets, *_ = batch

        img = inputs[0].cpu().detach().numpy()
        img = img.squeeze(0)
        cv2.imshow(f'targets: {targets[0]}', img)
        print(targets[0])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        break
    





