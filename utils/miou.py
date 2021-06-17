import numpy as np
import cv2
import os
import random

import json
from collections import OrderedDict
import argparse
from PIL import Image as PILImage
from utils.transforms import transform_parsing

LABELS = ['Background', 'Hat', 'Hair', 'Sunglasses', 'Upper-clothes', 'Skirt', 'Pants', 'Dress', 'Belt', \
          'Left-shoe', 'Right-shoe', 'Face', 'Left-leg', 'Right-leg', 'Left-arm', 'Right-arm', 'Bag', 'Scarf']

part_dict = {
        'dress':7,
        'skirt':5,
        'scarf':17,
        'sunglass':3
        }

def get_atr_palette():
    palette = [0,0,0,
            128,0,0,
            255,0,0,
            0,85,0,
            170,0,51,
            255,85,0,
            0,0,85,
            0,119,221,
            85,85,0,
            0,85,85,
            85,51,0,
            52,86,128,
            0,128,0,
            0,0,255,
            51,170,221,
            0,255,255,
            85,255,170,
            170,255,85]
    return palette

def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """

    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette

def get_confusion_matrix(gt_label, pred_label, num_classes):
    """
    Calcute the confusion matrix by given label and pred
    :param gt_label: the ground truth label
    :param pred_label: the pred label
    :param num_classes: the nunber of class
    :return: the confusion matrix
    """
    index = (gt_label * num_classes + pred_label).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_classes, num_classes))

    for i_label in range(num_classes):
        for i_pred_label in range(num_classes):
            cur_index = i_label * num_classes + i_pred_label
            if cur_index < len(label_count):
                confusion_matrix[i_label, i_pred_label] = label_count[cur_index]

    return confusion_matrix


def compute_mean_ioU(preds, scales, centers, num_classes, datadir, input_size=[384, 384], dataset='val'):
    list_path = os.path.join(datadir, dataset + '_id.txt')
    val_id = [i_id.strip() for i_id in open(list_path)]

    confusion_matrix = np.zeros((num_classes, num_classes))

    for i, im_name in enumerate(val_id):
        gt_path = os.path.join(datadir, dataset + '_segmentations', im_name + '.png')

        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        h, w = gt.shape
        pred_out = preds[i]
        s = scales[i]
        c = centers[i]
        pred = transform_parsing(pred_out, c, s, 0, w, h, input_size)

        gt = np.asarray(gt, dtype=np.int32)
        pred = np.asarray(pred, dtype=np.int32)
        pred_save = np.array(pred, dtype=np.uint8)
        ignore_index = gt != 255

        gt = gt[ignore_index] # shape(240000,)
        pred = pred[ignore_index]

        confusion_matrix += get_confusion_matrix(gt, pred, num_classes) #size(18,18)

    pos = confusion_matrix.sum(1) #sum row    # shape(18,)
    res = confusion_matrix.sum(0) #sum column # shape(18,)
    tp = np.diag(confusion_matrix)            # shape(18,) 
    fp = res - tp # shape(18,)
    fn = pos - tp # shape(18,)

    pixel_accuracy = (tp.sum() / pos.sum()) * 100
    mean_accuracy = ((tp / np.maximum(1.0, pos)).mean()) * 100 #recall
    recall = mean_accuracy
    precision = ((tp / np.maximum(1.0, res)).mean()) * 100 #precision
    f1 = (2*recall*precision)/(recall+precision)

    precision_class = (tp / np.maximum(1.0, res))*100 # shape(18,)
    recall_class = (tp / np.maximum(1.0, pos))*100    # shape(18,)
    f1_class = (2*recall_class*precision_class)/(recall_class+precision_class)  # shape(18,)

    IoU_array = (tp / np.maximum(1.0, pos + res - tp)) # IoU=tp/(tp+fp+fn)
    IoU_array = IoU_array * 100
    mean_IoU = IoU_array.mean()


    name_value = []
    f1_value = []
    for i, (label, iou) in enumerate(zip(LABELS, IoU_array)):
        name_value.append((label, iou))
    for j,(label, f1) in enumerate(zip(LABELS, f1_class)):
        f1_value.append((label, f1))
    print(f1_value)
    print('\n')

    name_value.append(('Pixel accuracy', pixel_accuracy))
    name_value.append(('Mean accuracy', mean_accuracy))
    name_value.append(('Mean IU', mean_IoU))
    name_value = OrderedDict(name_value)
    return name_value


