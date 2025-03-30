import os
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from model.model import Model, TransformerModel, VideoModel, r21d, VideoModel3sD
from functions.settings import *

from preprocess_data import get_dataloader

# model = Model()
# model = TransformerModel(d_model=512, nhead=4, num_encoder_layers=4)
model = VideoModel3sD()
model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

train_dataloader, val_dataloader, pos_weight = get_dataloader() 

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
# if TEMPORAL_FRAME_WINDOW == 1: fig = plt.figure(figsize=(20, 10))
# for i, (input, target) in enumerate(val_dataloader):
#     if TEMPORAL_FRAME_WINDOW > 1:
#         fig = plt.figure(figsize=(20, 10), tight_layout=True); manager = plt.get_current_fig_manager(); manager.window.wm_geometry("+0+0")
#     output = model(input.to(DEVICE)); out = output[0].cpu().detach().numpy(); out = torch.softmax(torch.tensor(out), dim=0); outstr = "["
#     for i in range(len(out)):
#         outstr += f"{out[i]:.2f}"
#         if i < len(out) - 1: outstr += ", "
#     outstr += "]"; target = target[0]; targetstr = "["
#     for i in range(len(target)):
#         targetstr += f"{target[i]:.2f}"
#         if i < len(target) - 1: targetstr += ", "
#     targetstr += "]"; keymapstr = "[  W  ,  A  ,  D  ,  S  , WA  , WD  , NK  ]"
#     if (i > 30): break
#     if TEMPORAL_FRAME_WINDOW == 1:
#         ax = fig.add_subplot(2, 3, i + 1)
#         ax.imshow(input[0].cpu().detach().numpy().transpose(1, 2, 0), cmap='gray')
#     else:
#         ax = fig.add_subplot(1, 1, 1); num_cycles = 3
#         for cycle in range(num_cycles):
#             for j in range(TEMPORAL_FRAME_WINDOW):
#                 ax.imshow(input[0][j].cpu().detach().numpy().transpose(1, 2, 0), cmap='gray'); ax.autoscale(False); ax.axis('off')
#                 ax.set_title(f"{keymapstr}\n{targetstr}\n {outstr}"); plt.pause(0.1)  # Delay to simulate 5fps video playback
#                 if j < TEMPORAL_FRAME_WINDOW - 1: ax.cla()  # Clear the axis for the next frame
#     if TEMPORAL_FRAME_WINDOW > 1: plt.show(); plt.close(fig)
# if TEMPORAL_FRAME_WINDOW == 1: plt.show()
# os._exit(0)

losses = []

import cv2

def train_one_epoch(epoch_index, losses):
    running_loss = 0.0

    optimizer.zero_grad()
    model.train()  # set model to training mode

    pbar = tqdm(train_dataloader, desc=f"Epoch {epoch_index + 1}/{EPOCHS}")
    for mini_batch_idx, data in enumerate(pbar):
        inputs, labels = data
        
        inputs = inputs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        outputs = model(inputs)

        # sanity check first sample
        # print(f" outputs: {torch.sigmoid(outputs[0]).cpu().detach().numpy()}")
        # print(f"labels: {labels[0]}")
        # single_output = model(inputs[0].unsqueeze(0))
        # single_loss = loss_fn(single_output, labels[0].unsqueeze(0))
        # print(f"Single sample loss: {single_loss.item()}")
        # cv2.imshow(f'targets: {labels[0]}', cv2.resize(inputs[0].cpu().detach().numpy().transpose(1, 2, 0), (700, 700)))
        # cv2.waitKey(0)

        # print(f"outputs: {outputs}")
        # print(f"labels: {labels}")

        loss = loss_fn(outputs, labels)
        loss.backward()

        if (mini_batch_idx + 1) % BATCHES_TO_AGGREGATE == 0 or mini_batch_idx == len(train_dataloader) - 1:
            optimizer.step()
            optimizer.zero_grad()
            losses.append(min(loss.item(), 2)) # no need to plot outliers

        running_loss += loss.item()
        pbar.set_postfix(epoch_loss = running_loss / (mini_batch_idx + 1)) #, gpu_mem=round(torch.cuda.memory_reserved() / 1e9, 2))

def validate_one_epoch(epoch_index):
    model.eval()  # set model to evaluate mode
    running_loss = 0.0

    pbar = tqdm(val_dataloader, desc=f"Validation epoch {epoch_index + 1}/{EPOCHS}")

    with torch.no_grad():
        for mini_batch_idx, data in enumerate(pbar):
            inputs, labels = data
            
            inputs = inputs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            running_loss += loss.item()
            pbar.set_postfix(val_loss=running_loss / (mini_batch_idx + 1))

if __name__ == "__main__":

    for epoch in range(EPOCHS):
        train_one_epoch(epoch, losses)
        validate_one_epoch(epoch)
        torch.save(model.state_dict(), EXISTING_MODEL_PATH)

    plt.plot(losses)
    plt.show()