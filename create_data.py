import numpy as np
import cv2
import time
import os
from tqdm import tqdm
from functions.grabscreen import grab_screen
from functions.getkeys import key_check
from functions.settings import *


def get_next_file_index():
    file_index = 0
    if not os.path.exists(DATA_PATH.replace(DATA_PATH.split('/')[-1], '')):
        os.makedirs(DATA_PATH.replace(DATA_PATH.split('/')[-1], ''))
    while os.path.isfile(f'{DATA_PATH}_{file_index}.npy'):
        file_index += 1

    if file_index > 0:
        print(f"Found existing files. Starting from {file_index}")
    
    return file_index


def save_data(training_data, file_index):
    np.save(f'{DATA_PATH}_{file_index}', training_data)


def keys_to_output(keys):
    #[W, A, S, D, WA, WD, NK]
    w  = [1,0,0,0,0,0,0]
    a  = [0,1,0,0,0,0,0]
    s  = [0,0,1,0,0,0,0]
    d  = [0,0,0,1,0,0,0]
    wa = [0,0,0,0,1,0,0]
    wd = [0,0,0,0,0,1,0]
    nk = [0,0,0,0,0,0,1]
    output = [0,0,0,0,0,0,0]

    if 'W' in keys and 'A' in keys and not 'D' in keys:
        output = wa
    elif 'W' in keys and 'D' in keys and not 'A' in keys:
        output = wd
    elif 'W' in keys:
        output = w
    elif 'S' in keys:
        output = s
    elif 'A' in keys and not 'D' in keys:
        output = a
    elif 'D' in keys and not 'A' in keys:
        output = d
    else:
        output = nk

    return output

def countdown(seconds=3):
    for i in list(range(seconds))[::-1]:
        print(i + 1, end=' ')
        time.sleep(1)
    print()

def undo_frames(training_data, file_index, frame_count, pbar):
    frames_to_remove = frame_count
    for _ in range(frame_count):
        if (len(training_data) == 0):
            break
        training_data.pop()
        frames_to_remove -= 1
        pbar.update(-1)
    
    if (frames_to_remove and file_index > 0):
        print("Also removing frames from the previous file")
        # have to remove also from the previous file
        training_data = []
        previous_file_path = f'{DATA_PATH}_{file_index - 1}.npy'
        if os.path.isfile(previous_file_path):
            previous_file_data = list(np.load(previous_file_path, allow_pickle=True))
            for i in range(len(previous_file_data) - frames_to_remove):
                previous_file_data.pop()
            
            training_data = previous_file_data
            os.remove(previous_file_path)
            file_index -= 1
            print(f'Frames removed from {previous_file_path}')
        else:
            print(f'File {previous_file_path} not found')

    return training_data, file_index



def main():
    file_index = get_next_file_index()

    training_data = []
    paused = True
    print("Press 'Y' to start and 'T' to pause")

    frames_to_average_fps = 30
    average_fps = []    
    last_time = time.time() 
    pbar = tqdm(total=SAMPLES_IN_ONE_FILE)

    while True:
        keys = key_check()

        if not paused:
            # grab screen. we are using 40 pixels from the top of the screen to avoid the window title bar
            screen = grab_screen(region=(0, 40, SOURCE_WIDTH, SOURCE_HEIGHT))
            # resize the screen to the model input size BEFORE saving to save space
            screen = cv2.resize(screen, (MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT))
            if GRAYSCALE:
                screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)

            output = keys_to_output(keys)

            training_data.append([screen, output])

            # some code to keep track of the progress
            pbar.update(1)
            average_fps.append(1 // (time.time()-last_time))
            if len(average_fps) > frames_to_average_fps:
                average_fps.pop(0)
            pbar.set_description(f"Creating file {file_index}, FPS = {sum(average_fps) / len(average_fps):.1f}".ljust(20))
            last_time = time.time()

            # save data in chunks
            if len(training_data) % SAMPLES_IN_ONE_FILE == 0:
                save_data(training_data, file_index)
                file_index += 1
                training_data = []
                pbar.reset()

        # handle pausing
        if 'T' in keys and not paused:
            paused = True
            print(' PAUSED', end=' | ')
        if 'Y' in keys and paused:
            paused = False
            print('RESUMED', end=' ')
            countdown()
        if 'R' in keys and not paused:
            training_data, file_index = undo_frames(training_data, file_index, 300, pbar)
            paused = True

if __name__ == "__main__":
    main()
