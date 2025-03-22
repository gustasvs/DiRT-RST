import os
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from model.model import Model
from functions.settings import *

from preprocess_data import get_dataloader

model = Model()
model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

training_loader, pos_weight = get_dataloader() 

loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# check directory for existing model
if not os.path.exists(EXISTING_MODEL_PATH.replace(EXISTING_MODEL_PATH.split('/')[-1], '')):
    os.makedirs(EXISTING_MODEL_PATH.replace(EXISTING_MODEL_PATH.split('/')[-1], ''))
    print(f'Directory {EXISTING_MODEL_PATH.replace(EXISTING_MODEL_PATH.split("/")[-1], "")} created')
else:
    print(f'Directory {EXISTING_MODEL_PATH.replace(EXISTING_MODEL_PATH.split("/")[-1], "")} already exists')

if os.path.exists(EXISTING_MODEL_PATH):
    model.load_state_dict(torch.load(EXISTING_MODEL_PATH))
    print('Model loaded')
else:
    print('No model found')

model.eval()

# sanity check (log first sample)
# fig = plt.figure(figsize=(20, 10))
# for i, (input, target) in enumerate(training_loader):
#     output = model(input)
#     if (i > 4):
#         break
#     ax = fig.add_subplot(2, 3, i + 1)
#     ax.imshow(input[0].cpu().detach().numpy().transpose(1, 2, 0))
#     out = output[0].cpu().detach().numpy()
#     outstr = "["
#     for i in range(len(out)):
#         outstr += f"{out[i]:.2f}"
#         if i < len(out) - 1:
#             outstr += ", "
#     outstr += "]"
#     ax.set_title(f"{target[0]}\n {outstr}")
#     ax.axis('off')
#     ax.set_autoscale_on(True)
# plt.show()
# exit(0)

losses = []

import cv2

def train_one_epoch(epoch_index, losses):
    running_loss = 1e-5

    pbar = tqdm(training_loader, desc=f"Epoch {epoch_index + 1}/{EPOCHS}")
    for i, data in enumerate(pbar):
        inputs, labels = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()
        # Make predictions for this batch
        outputs = model(inputs)

        # sanity check first sample
        # print(f" outputs: {torch.sigmoid(outputs[0]).cpu().detach().numpy()}")
        # print(f"labels: {labels[0]}")
        # single_output = model(inputs[0].unsqueeze(0))
        # single_loss = loss_fn(single_output, labels[0].unsqueeze(0))
        # print(f"Single sample loss: {single_loss.item()}")
        # cv2.imshow(f'targets: {labels[0]}', cv2.resize(inputs[0].cpu().detach().numpy().transpose(1, 2, 0), (700, 700)))
        # cv2.waitKey(0)

        loss = loss_fn(outputs, labels)
        loss.backward()
        # Adjust learning weights
        optimizer.step()
        # Gather data and report

        running_loss += loss.item()

        pbar.set_postfix(epoch_loss = running_loss / (i + 1))
        losses.append(min(loss.item(), 2)) # no need to plot outliers

if __name__ == "__main__":

    for epoch in range(EPOCHS):
        train_one_epoch(epoch, losses)
        torch.save(model.state_dict(), EXISTING_MODEL_PATH)

    plt.plot(losses)
    plt.show()