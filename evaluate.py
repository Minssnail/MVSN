import argparse
import datetime
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils import data
import torchvision.transforms as transforms
from copy import deepcopy
from PIL import Image as PILImage

from networks.model import MVSN_Model
from dataset.pose_edge_datasets import ATRDataSet
from utils import transforms as tr
from utils.miou import compute_mean_ioU, get_atr_palette #!
from utils.encoding import DataParallelModel

import torchsnooper
import pysnooper

DATA_DIRECTORY = './dataset/ATR/'
DATA_LIST_PATH = './dataset/ATR/val_id.txt' 
VAL_POSE_ANNO_FILE='./dataset/ATR/TrainVal_pose_annotations/ATR_SP_VAL_annotations.json' 
IGNORE_LABEL = 255
NUM_CLASSES = 18 
SNAPSHOT_DIR = './snapshots/'
INPUT_SIZE = '384,384'
RESTORE_FROM = './snapshots/ATR_best.pth'#! add default=RESTORE_FROM

start = datetime.datetime.now()
#  print("Start at: ", start)

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="MVSN Network")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--pose-anno-file", type=str,
                        help="Path to the annotation file of keypoint")
    parser.add_argument("--data-name", type=str, default='atr',
                        help='Dataset to be evaluated.')
    parser.add_argument("--save-dir", type=str, default='./output/',
                        help='Directory to save parsing results')
    parser.add_argument("--dataset", type=str, default='val',
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-points", type=int, help='Num of class of keypoint')
    parser.add_argument("--restore-from", type=str,default=RESTORE_FROM,
                        help="Where restore model parameters from.")  #! add default=RESTORE_FROM
    parser.add_argument("--gpu", type=str, default='0',
                        help="choose gpu device.")  # !default='0,1'
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")

    return parser.parse_args()


#@pysnooper.snoop('./evallog.txt')
#@torchsnooper.snoop('./evallog.txt')
def valid(model, valloader, input_size, num_samples, gpus):
    model.eval()
    args = get_arguments()
    palette = get_atr_palette()

    parsing_preds = np.zeros((num_samples, input_size[0], input_size[1]), dtype=np.uint8)
    gt = np.zeros((num_samples, input_size[0], input_size[1]), dtype=np.uint8)

    parsing_ = torch.zeros([num_samples, input_size[0], input_size[1]], dtype=torch.int32)

    scales = np.zeros((num_samples, 2), dtype=np.float32)
    centers = np.zeros((num_samples, 2), dtype=np.int32)

    idx = 0
    gt_idx = 0
    interp = torch.nn.Upsample(size=(input_size[0], input_size[1]), mode='bilinear', align_corners=True)
    # interp = F.interpolate(input, size=(input_size[0], input_size[1]), mode='bilinear', align_corners=True)
    # interp = torch.nn.interpolate(size=(input_size[0], input_size[1]), mode='bilinear', align_corners=True) 
    
    with torch.no_grad():
        for index, batch in enumerate(valloader):
            if args.data_name == 'atr':
                image, label, pose, edge, vector, meta = batch #!
            num_images = image.size(0)
            if index % 100 == 0:
                print('%d  processd' % (index * num_images))

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            scales[idx:idx + num_images, :] = s[:, :]
            centers[idx:idx + num_images, :] = c[:, :]

            outputs = model(image.cuda())
            gt[gt_idx:gt_idx + num_images, :, :] = label

            if gpus > 1:
                i = 0
                for output in outputs:
                    if not isinstance(output, list):
                        parsing = output
                    else:
                        parsing = output[0]
                    if not isinstance(parsing, list):
                        parsing = parsing
                    else:
                        parsing = parsing[1]

                    nums = len(parsing)
                    parsing = interp(parsing)
                    parsing = parsing.permute(0, 2, 3, 1)
                    parsing_[idx:idx + nums, :, :] = parsing.max(3)[1]

                    idx += nums
                    i += nums

            else:
                parsing = outputs[0][1]
                parsing = interp(parsing).data.cpu().numpy()
                #parsing = F.interpolate(parsing, size=(input_size[0], input_size[1]), mode='bilinear', align_corners=True).data.cpu().numpy()
                parsing = parsing.transpose(0, 2, 3, 1)  # NCHW NHWC
                parsing_preds[idx:idx + num_images, :, :] = np.asarray(np.argmax(parsing, axis=3), dtype=np.uint8)

                seg_pred = np.asarray(np.argmax(parsing, axis=3), dtype=np.uint8)
                num = seg_pred.shape[0]

                for i in range(num):
                    output_im = PILImage.fromarray(seg_pred[i])
                    output_im.putpalette(palette)
                    output_im.save(args.save_dir + meta['name'][i] + '.png')

                idx += num_images

    if gpus > 1:
        parsing_preds = parsing_[:num_samples, :, :].numpy()
    else:
        parsing_preds = parsing_preds[:num_samples, :, :]
    return parsing_preds, scales, centers



def main():
    torch.multiprocessing.set_start_method("spawn", force=True)
    """Create the model and start the evaluation process."""
    args = get_arguments()

    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    gpus = [int(i) for i in args.gpu.split(',')]
    h, w = map(int, args.input_size.split(','))
    
    input_size = (h, w)
    #print("h:",h)
    print("input_size:",input_size)
    deeplab = MVSN_Model(args.num_classes, args.num_points)
    if len(gpus) > 1:
        model = DataParallelModel(deeplab)
    else:
        model = deeplab

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    if args.data_name == 'atr':
        atr_dataset = ATRDataSet(args.data_dir, VAL_POSE_ANNO_FILE, args.dataset, crop_size=input_size, transform=transform)
        num_samples = len(atr_dataset)
        valloader = data.DataLoader(atr_dataset, batch_size=args.batch_size * len(gpus),
                                    shuffle=False, num_workers=4, pin_memory=True)

    restore_from = args.restore_from
    state_dict = model.state_dict().copy()
    state_dict_old = torch.load(restore_from) # ! , map_location='cuda:0'

    for key in state_dict.keys():
        if key not in state_dict_old.keys():
            print(key)
    for key, nkey in zip(state_dict_old.keys(), state_dict.keys()):
        if key != nkey:
            state_dict[key[7:]] = deepcopy(state_dict_old[key])
        else:
            state_dict[key] = deepcopy(state_dict_old[key])

    model.load_state_dict(state_dict)#!, map_location='cuda:0'
    model.eval()
    model.cuda()

    parsing_preds, scales, centers = valid(model, valloader, input_size, num_samples, len(gpus))

    mIoU = compute_mean_ioU(parsing_preds, scales, centers, args.num_classes, args.data_dir, input_size, args.dataset)
    print(mIoU)

    end = datetime.datetime.now()
    print(end - start, 'seconds')
    print(end)


if __name__ == '__main__':
    main()
