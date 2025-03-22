import os
from functions.settings import *

def reset_current_data():

    # missinput chech
    check = input('Are you sure you want to reset the current data? (y/n): ')
    if check.lower() == 'y':
        file_index = 0
        while os.path.isfile(f'{DATA_PATH}_{file_index}.npy'):
            try:
                os.remove(f'{DATA_PATH}_{file_index}.npy')
                file_index += 1
                print(f'{DATA_PATH}_{file_index}.npy removed')
            except:
                print(f'{DATA_PATH}_{file_index}.npy could not be removed')
    else:
        print('Reset canceled')

    check = input('Remove the model? (y/n): ')
    if check.lower() != 'y':
        print('Reset canceled')
        return
    
    try:
        os.remove(EXISTING_MODEL_PATH)
        print(f'{EXISTING_MODEL_PATH} removed')
    except:
        print(f'{EXISTING_MODEL_PATH} could not be removed')

if __name__ == '__main__':
    reset_current_data()