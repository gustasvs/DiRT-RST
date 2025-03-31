import numpy as np
import cv2
import time
import os
import random
import colorama
from functions.grabscreen import grab_screen
from functions.getkeys import key_check
from functions.settings import *
from functions.directkeys import PressKey, ReleaseKey, W, A, S, D
from vjoy import vJoy, ultimate_release

from preprocess_data import prepare_image
from model.model import VideoModel, VideoModel3sD

TEMPORAL_FRAME_GAP = int(TEMPORAL_FRAME_GAP / 2)

colorama.init()
vj = vJoy()


model = VideoModel3sD()
model.to(DEVICE)


if os.path.exists(EXISTING_MODEL_PATH):
    model.load_state_dict(torch.load(EXISTING_MODEL_PATH))
    print('Model loaded')
else:
    print('No model found')

model.eval()

# trigger vjoy so it is ready to be used
vj.open()
joystickPosition = vj.generateJoystickPosition()
vj.update(joystickPosition)
time.sleep(0.001)
vj.close()

# main script
def main():
    XYRANGE = 16393
    ZRANGE = 32786

    paused = False

    past_frames = []

    while True:
        
        last_time = time.time()

        keys = key_check()
        if 'T' in keys:
            if paused:
                paused = False
                time.sleep(1)
            else:
                paused = True
                vj.open()
                joystickPosition = vj.generateJoystickPosition(wAxisZRot = 0)
                vj.update(joystickPosition)
                time.sleep(1)
                vj.close()

                ReleaseKey(W)
                ReleaseKey(A)
                ReleaseKey(S)
                ReleaseKey(D)

        if paused:
            continue

        # grab screen. we are using 40 pixels from the top of the screen to avoid the window title bar
        screen = grab_screen(region=(0, 40, SOURCE_WIDTH, SOURCE_HEIGHT))
        # resize the screen to the model input size BEFORE saving to save space
        screen = cv2.resize(screen, (MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT))
        if GRAYSCALE:
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)

        past_frames.append(prepare_image(screen))
        if len(past_frames) > TEMPORAL_FRAME_GAP * TEMPORAL_FRAME_WINDOW:
            past_frames.pop(0)

        if len(past_frames) < TEMPORAL_FRAME_GAP * TEMPORAL_FRAME_WINDOW:
            prediction = [0 for i in range(TARGET_CLASS_COUNT)]
        else:
            past_frames_to_model = []
            for i in range(0, TEMPORAL_FRAME_GAP * TEMPORAL_FRAME_WINDOW, TEMPORAL_FRAME_GAP):
                past_frames_to_model.append(past_frames[i])
            
            input_tensor = torch.stack(past_frames_to_model, dim=0)

            # print(input_tensor.shape, TARGET_CLASS_COUNT, TEMPORAL_FRAME_WINDOW, TEMPORAL_FRAME_GAP)
            prediction = model(torch.unsqueeze(input_tensor, 0).to(DEVICE))
            prediction = torch.sigmoid(prediction[0]).cpu().detach().numpy()

            # fig, axes = plt.subplots(1, len(past_frames_to_model), figsize=(15, 5))
            # for i, (frame, ax) in enumerate(zip(past_frames_to_model, axes)):
            #     ax.imshow(frame.squeeze(), cmap='gray' if GRAYSCALE else None)
            #     ax.axis('off')
            #     ax.set_title(f"Frame {i+1}")
            # plt.tight_layout()
            # plt.show()

            # manager = plt.get_current_fig_manager()
            # manager.window.wm_geometry("+900+0")
            # plt.imshow(past_frames[-1].squeeze(), cmap='gray' if GRAYSCALE else None)
            # plt.axis('off')
            # plt.pause(0.001)
        
        # prediction_labels = ['w', 'a', 's', 'd', 'wa', 'wd', 'nk']
        # prediction = dict(zip(prediction_labels, prediction))
        # printable_str = ''
        # for key, value in prediction.items():
        #     printable_str += str(round(value, 1)).ljust(4) + ' | '
        # printable_str += '\n'
        # for key, value in prediction.items():
        #     printable_str += str(key).ljust(4) + ' | '
        # print(printable_str, end = '\r') 
        # # time.sleep(0.25)
        # continue

        # w a s d wa wd nk
        w  = prediction[0]
        a  = prediction[1]
        s  = prediction[2]
        d  = prediction[3]
        wa = prediction[4]
        wd = prediction[5] 
        nk = prediction[6]

        gas = w + wa + wd - nk - (s * 3)
        turn = d + wd - a - wa
        # turn = d - a
        gas = round(gas, 2)
        turn = round(turn, 2)

        # vj.open()
        # joystickPosition = vj.generateJoystickPosition(wAxisX = XYRANGE + int(round(XYRANGE * turn)))

        # vj.update(joystickPosition)
        # vj.close()

        # if (turn > 0.5):
        #     PressKey(D)
        #     ReleaseKey(A)
        # elif (turn < -0.5):
        #     PressKey(A)
        #     ReleaseKey(D)

        # if (gas > 0):
        #     PressKey(W)
        #     ReleaseKey(S)
        # else: 
        #     PressKey(S)
        #     ReleaseKey(W)


        
        sgas = str(round(gas / 30 * 100)).ljust(4)
        
        sture = ['□' for i in range(10)]
        sture.append('|')
        for i in range(11): sture.append('□')
        turn_int = min(10, max(-10, int(round(turn * 5))))
        if turn_int < 0:
            for i in range(turn_int, 0): sture[10 + i] = colorama.Fore.LIGHTBLUE_EX +  '■' + colorama.Fore.RESET
        if turn_int > 0:
            for i in range(turn_int): sture[11 + i] = colorama.Fore.LIGHTBLUE_EX +  '■' + colorama.Fore.RESET
        # # '□''■'
        ssture = ''
        for i in sture: ssture += str(i)
        fps = int(1 / (time.time() - last_time))
        last_time = time.time()
        print(colorama.Fore.GREEN if gas > 0 else colorama.Fore.LIGHTCYAN_EX, sgas, colorama.Fore.RESET, ssture, round(w + a + s + d + wa + wd + nk), fps, end = '\r')

main()
