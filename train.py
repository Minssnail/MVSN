import argparse
import datetime, time
import timeit
import os, sys
import os.path as osp
import numpy as np
import shutil
from copy import deepcopy
from collections import OrderedDict

import torch
import torch.multiprocessing as mp
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn
from torch.utils import data

from networks.model import MVSN_Model
from dataset.pose_edge_datasets import ATRDataSet

from utils.criterion import CriterionPoseEdge
from utils.encoding import DataParallelModel, DataParallelCriterion
from utils.miou import compute_mean_ioU

import torchsnooper
import pysnooper

start = datetime.datetime.now()

BATCH_SIZE = 4
DATA_DIRECTORY = './dataset/ATR/'
VAL_ANNO_FILE = './dataset/ATR/TrainVal_pose_annotations/ATR_SP_VAL_annotations.json'
POSE_ANNO_FILE = './dataset/ATR/TrainVal_pose_annotations/ATR_SP_TRAIN_annotations.json'
DATA_LIST_PATH = './dataset/ATR/train_id.txt'

IGNORE_LABEL = 255
INPUT_SIZE = '384,384'
LEARNING_RATE_SGD = 1e-3 # SGD
LEARNING_RATE_ADAM = 1e-4 # Adam
MOMENTUM = 0.9
NUM_CLASSES = 18 #!
POWER = 0.9
RANDOM_SEED = 1234

RESTORE_FROM = './dataset/resnet101-imagenet.pth'


SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 10000
SNAPSHOT_DIR = './snapshots/'
WEIGHT_DECAY = 0.0005

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def to_cudaTensor(array):
    """
    Convert nparray to cudaTensor

    :array
    list: contains four part gt label

    """
    tensor_list = []
    for arr in array:
        arr_ = torch.from_numpy(arr).long().cuda()
        tensor_list.append(arr_)
    return tensor_list


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="MVSN Network")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the dataset.")
    parser.add_argument("--pose-anno-file", type=str,default=POSE_ANNO_FILE,
                        help="Keypoint annotation file.")#! must have default value
    parser.add_argument("--dataset", type=str, default='train', choices=['train', 'val', 'trainval', 'test'],
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--dataset-list", type=str, default='_id.txt', choices=['_id.txt', '_part_id.txt'],
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE_SGD,#ADAM,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-points", type=int, default=14, 
                        help="Number of keypoints to predict (NOT including background).")
    parser.add_argument("--start-iters", type=int, default=0,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--gpu", type=str, default='0',
                        help="choose gpu device.")
    parser.add_argument("--start-epoch", type=int, default=0,
                        help="choose the number of recurrence.")
    parser.add_argument("--epochs", type=int, default=40, 
                        help="choose the number of recurrence.")
    return parser.parse_args()


args = get_arguments()

# lr_policy: https://blog.csdn.net/qq_39463274/article/details/104998110
def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

# adjust_learning_rate
def adjust_learning_rate(optimizer, i_iter, total_iters):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = lr_poly(args.learning_rate, i_iter, total_iters, args.power)
    optimizer.param_groups[0]['lr'] = lr
    return lr

def set_bn_eval(m): #https://blog.csdn.net/jacke121/article/details/103810682
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def set_bn_momentum(m): #https://blog.csdn.net/SHU15121856/article/details/88852328
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1 or classname.find('InPlaceABN') != -1:
        m.momentum = 0.0003

#@torchsnooper.snoop()
#@pysnooper.snoop()
#@torchsnooper.snoop('./trainlog.txt')
def main():
    torch.cuda.empty_cache()
    """start multiprocessing method"""
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass

    """Create the model and start the training."""
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    gpus = [int(i) for i in args.gpu.split(',')] # gpus = [0,1]
    # gpus = [args.gpu.split(',')] # !wrong gpus = [['0','1']]
    len_gpu=len(gpus)
    print(len_gpu)
  
    if not args.gpu == 'None':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu # ! if only one gpu be used, try to rewrite args.gpu into '0,1'

    h, w = map(int, args.input_size.split(','))

    input_size = [h, w]
    cudnn.enabled = True
    # cudnn related setting
    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True #False
    torch.backends.cudnn.enabled = True
    torch.cuda.empty_cache()

    deeplab = MVSN_Model(num_classes=args.num_classes)
    saved_state_dict = torch.load(args.restore_from)
    new_params = deeplab.state_dict().copy()

    i=0
    print("Now is loading pre-trained res101 model!")
    for i in saved_state_dict:
        i_parts = i.split('.')
        if not i_parts[0] == 'fc':
            new_params['.'.join(i_parts[0:])] = saved_state_dict[i]

    deeplab.load_state_dict(new_params)
    criterion = CriterionPoseEdge()
    criterion = DataParallelCriterion(criterion)
    criterion.cuda()

    snapshot_fname = osp.join(args.snapshot_dir, 'ATR_epoch_')
    snapshot_best_fname = osp.join(args.snapshot_dir, 'ATR_best.pth')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    dataset_ATR = ATRDataSet(args.data_dir, args.pose_anno_file, args.dataset, crop_size=input_size, dataset_list=args.dataset_list, transform=transform)
    trainloader = data.DataLoader(dataset_ATR, batch_size=args.batch_size * len(gpus), shuffle=True, num_workers=1, pin_memory=True)
    ATR_dataset = ATRDataSet(args.data_dir, VAL_ANNO_FILE, 'val', crop_size=input_size, dataset_list=args.dataset_list, transform=transform)
    num_samples = len(ATR_dataset)
    valloader = data.DataLoader(ATR_dataset, batch_size=args.batch_size * len(gpus), shuffle=False, num_workers=0, pin_memory=True)

    optimizer = optim.SGD(deeplab.parameters(),lr=args.learning_rate,momentum=args.momentum,weight_decay=args.weight_decay)

    #optimizer = optim.Adam(deeplab.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay)

    model = DataParallelModel(deeplab)
    model.cuda()

    optimizer.zero_grad()

    total_iters = args.epochs * len(trainloader)
    print("total iters:", total_iters)

    total_iter_per_batch = len(trainloader)
    

    best_iou = 0
    i_iter = 0
    temp = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        model.train()
        for i_iter, batch in enumerate(trainloader):
            iter_lr = i_iter + epoch * len(trainloader)
            lr = adjust_learning_rate(optimizer, iter_lr, total_iters)
            print('\nEpoch: %d | i_iter: %d | LR: %.8f' % (epoch + 1, i_iter,lr)) 
            images, labels, pose, edge, vector, _ = batch #!
            labels = labels.long().cuda(non_blocking=True)
            edge = edge.long().cuda(non_blocking=True)
            pose = pose.float().cuda(non_blocking=True)
            classL = vector.float().cuda(non_blocking=True) #!

            preds = model(images)
            loss = criterion(preds, [labels, edge, pose, classL]) #!
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i_iter % 500 == 0:
                tim = time.time()
                print('iter:{}/{},loss:{:.3f},lr:{:.3e},time:{:.1f}'.format(i_iter, total_iter_per_batch, loss.data.cpu().numpy(), lr, tim - temp))
                temp = tim

        h=time.time()
        if epoch % 5 == 0:
            print("----->Epoch:", epoch) 
            parsing_preds, scales, centers = valid(model, valloader, input_size, num_samples, len(gpus), criterion, args) # validate for each 5 epoches
            if args.dataset_list == '_id.txt':
                mIoU = compute_mean_ioU(parsing_preds, scales, centers, args.num_classes, args.data_dir, input_size)
            miou = mIoU['Mean IU']
            is_best_iou = miou > best_iou
            best_iou = max(miou, best_iou)
            torch.save(model.state_dict(), snapshot_fname + '.pth')
            if is_best_iou:
                print("Best iou epoch: ", epoch)
                shutil.copyfile(snapshot_fname + '.pth', snapshot_best_fname)

    end = datetime.datetime.now()
    print(end - start, 'seconds')
    print(end)


#@torchsnooper.snoop('./validlog.txt')
def valid(model, valloader, input_size, num_samples, gpus, criterion, args):
    model.eval()
    parsing_preds = np.zeros((num_samples, input_size[0], input_size[1]),
                           dtype=np.uint8)

    parsing_ = torch.zeros([num_samples, input_size[0], input_size[1]], dtype=torch.int32)
    scales = np.zeros((num_samples, 2), dtype=np.float32)
    centers = np.zeros((num_samples, 2), dtype=np.int32)

    idx = 0
    interp = torch.nn.Upsample(size=(input_size[0], input_size[1]), mode='bilinear', align_corners=True) 
    #interp = torch.nn.interpolate(size=(input_size[0], input_size[1]), mode='bilinear', align_corners=True) #!interpolate instead Upsample
    with torch.no_grad():
        for index, batch in enumerate(valloader):
            image, label, pose, edge, vector, meta = batch #!  must have enough return value (otherwise: too many values to unpack (expected 5))
            outputs = model(image.cuda())

            num_images = image.size(0)
            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            scales[idx:idx + num_images, :] = s[:, :]
            centers[idx:idx + num_images, :] = c[:, :]

            if gpus > 1:
                for output in outputs:
                    if not isinstance(output, list):
                        parsing = output
                    else:
                        parsing = output[0]
                    if not isinstance(parsing, list):
                        parsing = parsing
                    else:
                        parsing = parsing[2] 
                    nums = len(parsing)

                    parsing = interp(parsing)
                    parsing = parsing.permute(0, 2, 3, 1)  
                    parsing_[idx:idx + nums, :, :] = parsing.max(3)[1]
                    idx += nums
            else:
                parsing = outputs
                parsing = interp(parsing).data.cpu().numpy()
                parsing = parsing.transpose(0, 2, 3, 1) 
                parsing_preds[idx:idx + num_images, :, :] = np.asarray(np.argmax(parsing, axis=3), dtype=np.uint8)

                idx += num_images

    parsing_preds = parsing_[:num_samples, :, :].numpy()

    return parsing_preds, scales, centers

if __name__ == '__main__':
    main()
