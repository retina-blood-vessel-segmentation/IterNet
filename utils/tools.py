import cv2
import imageio
import json
import math
import mlflow
import numpy as np
import pickle
import os

from pathlib import Path
from sklearn.metrics import recall_score, roc_auc_score, accuracy_score, roc_curve
from sklearn.metrics import confusion_matrix, precision_score, jaccard_score


def crop_to_shape(data, shape):
    """
    Crops the array to the given image shape by removing the border (expects a tensor of shape [batches, nx, ny, channels].

    :param data: the array to crop
    :param shape: the target shape
    """
    #

    offset0 = (data.shape[1] - shape[1])//2
    offset1 = (data.shape[2] - shape[2])//2
    if offset0==0:
        if data.shape[1] % 2 == 1 or shape[1] % 2 == 1:
            return data[:, offset0:data.shape[1], offset1:(-offset1)]
        elif data.shape[2] % 2 == 1 or shape[2] % 2 == 1:
            return data[:, offset0:data.shape[1], offset1:(-offset1 - 1)]
        else:
            return data[:, offset0:data.shape[1], offset1:(-offset1)]
    elif offset1==0:
        if data.shape[1] % 2 == 1 or shape[1] % 2 == 1:
            return data[:, offset0:(-offset0 - 1), offset1:data.shape[2]]
        elif data.shape[2] % 2 == 1 or shape[2] % 2 == 1:
            return data[:, offset0:-offset0, offset1:data.shape[2]]
        else:
            return data[:, offset0:-offset0, offset1:data.shape[2]]
    else:
        if data.shape[1] % 2 == 1 or shape[1] % 2 == 1:
            return data[:, offset0:(-offset0 - 1), offset1:(-offset1)]
        elif data.shape[2] % 2 == 1 or shape[2] % 2 == 1:
            return data[:, offset0:-offset0, offset1:(-offset1-1)]
        else:
            return data[:, offset0:-offset0, offset1:(-offset1)]


def load_files(images_path, label_path, desired_size, label_name_fnc, mode,startat,bucket):
    """

    :param path:
    :param desired_size:
    :return:
    """

    images_path = Path(images_path).resolve()
    label_path = Path(label_path).resolve()

    images = list()
    labels = list()
    fnames = list()
    i = 0
    for p in sorted(images_path.glob('**/*')):
        if i < startat:
            i = i + 1
            continue
        if i >= startat + bucket:
            break
        i = i + 1
        fnames.append(p.stem)
        im = imageio.imread(str(p))
        old_size = im.shape[:2]
        delta_w = desired_size - old_size[1]
        delta_h = desired_size - old_size[0]

        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        images.append(cv2.resize(new_im, (desired_size, desired_size)))
        label = None
        if (label_path / label_name_fnc(p)).suffix == ".gif":
            label = imageio.imread(label_path / label_name_fnc(p))
        else:
            label = imageio.imread(label_path / label_name_fnc(p), pilmode='L')
        if mode.lower() in ['train', 'validate']:
            new_label = cv2.copyMakeBorder(label, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0])
            _, temp = cv2.threshold(new_label, 127, 255, cv2.THRESH_BINARY)
        else:
            _, temp = cv2.threshold(label, 127, 255, cv2.THRESH_BINARY)
        labels.append(temp)

    x_data = np.array(images).astype('float32') / 255.
    x_data = np.reshape(x_data, (len(x_data), desired_size, desired_size, 3))
    y_data = np.array(labels).astype('float32') / 255.
    if mode.lower() in ['train', 'validate']:
        y_data = np.reshape(y_data, (len(y_data), desired_size, desired_size, 1))

    return x_data, y_data, i-startat,fnames


def load_mask_files(mask_path, test_path, mask_name_fnc,startat,bucket):
    """

    :param mask_path:
    :param test_path:
    :param mask_name_fnc:
    :return:
    """

    test_path = Path(test_path).resolve()
    mask_path = Path(mask_path).resolve()

    all_masks_data = list()
    i = 0
    for p in sorted(test_path.glob("**/*")):
        if i < startat:
            i = i + 1
            continue
        if i >= startat + bucket:
            break
        i = i + 1
        mask_data = imageio.imread(mask_path / mask_name_fnc(p))
        all_masks_data.append(np.array(mask_data).astype('float32') / 255.)

    return all_masks_data


def get_mask_pattern_for_dataset(dataset):
    if dataset == 'DRIVE':
        return get_mask_name_drive
    elif dataset == 'CHASE':
        return get_mask_name_chase
    elif dataset == 'DRIVE-eval':
        return get_mask_name_drive_eval
    elif dataset == 'CHASE-eval':
        return get_mask_name_chase_eval
    elif dataset == 'DROPS':
        return get_mask_name_drops
    elif dataset == 'STARE':
        return get_mask_name_stare
    else:
        return get_mask_name_universal


def get_label_pattern_for_dataset(dataset):
    if dataset == 'DRIVE':
        return get_label_name_drive
    elif dataset == 'CHASE':
        return get_label_name_chase
    elif dataset == 'DRIVE-eval':
        return get_label_name_drive_eval
    elif dataset == 'CHASE-eval':
        return get_label_name_chase_eval
    elif dataset == 'DROPS':
        return get_label_name_drops
    elif dataset == 'STARE':
        return get_label_name_stare
    elif dataset == 'STARE-eval':
        return get_label_name_stare_eval
    else:
        return get_label_name_universal


def get_desired_size(dataset):
    """
    Returns desired image size for a given dataset.

    :param dataset: A case sensitive dataset name.
    :return: Desired image size.
    :rtype: int
    """
    if dataset == 'DRIVE':
        return 592
    elif dataset == 'CHASE':
        return 1008
    elif dataset == 'DROPS':
        return 1008
    elif dataset == 'STARE':
        return 1008
    if dataset == 'DRIVE-eval':
        return 592
    elif dataset == 'CHASE-eval':
        return 1008
    elif dataset == 'DROPS-eval':
        return 1008
    elif dataset == 'STARE-eval':
        return 1008
    else:
        return 1008


def get_label_name_drops(image_path):
    return Path(image_path).stem + '.png'


def get_mask_name_drops(image_path):
    return Path(image_path).stem + '.png'


def get_label_name_drive(image_path):
    return Path(image_path).stem.split('_')[0] + '_manual1.png'


def get_mask_name_drive(image_path):
    return Path(image_path).stem + '_mask.gif'


def get_label_name_chase(image_path):
    return Path(image_path).stem + '_1stHO.png'


def get_mask_name_chase(image_path):
    return Path(image_path).stem + '.png'


def get_label_name_stare(image_path):
    return Path(image_path).stem + '.ah.ppm'


def get_mask_name_stare(image_path):
    return Path(image_path).stem + '.png'

def get_mask_name_universal(image_path):
    return Path(image_path).name

def get_label_name_universal(image_path):
    return Path(image_path).name

def get_mask_name_drive_eval(image_path):
    p = Path(image_path)
    return p.stem + ".png" 
def get_label_name_drive_eval(image_path):
    p = Path(image_path)
    return p.stem + ".png" 

def get_label_name_stare_eval(image_path):
    p = Path(image_path)
    return p.stem + ".ppm" 


def get_mask_name_chase_eval(image_path):
    p = Path(image_path)
    return p.stem + ".png"

def get_label_name_chase_eval(image_path):
    p = Path(image_path)
    return p.stem + ".png"
