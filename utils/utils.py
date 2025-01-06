#!/usr/bin/env python
# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import torch
from medpy import metric
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import SimpleITK as sitk
from scipy.ndimage import zoom
import os
import random
import cv2
import fcntl

def set_seed(seed = 123):
    """Set all seeds to make results reproducible (deterministic mode)."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.backends.cudnn.enabled = True
    cv2.setRNGSeed(seed)

set_seed()

class Normalize():
    def __call__(self, sample):
        function = transforms.Normalize((.5, .5, .5), (0.5, 0.5, 0.5))
        return function(sample[0]), sample[1]


class ToTensor():
    def __call__(self, sample):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        function = transforms.ToTensor()
        return function(sample[0]), function(sample[1])


class RandomRotation():
    def __init__(self):
        pass

    def __call__(self, sample):
        img, label = sample
        random_angle = np.random.randint(0, 360)
        return img.rotate(random_angle, Image.NEAREST), label.rotate(random_angle, Image.NEAREST)


class RandomFlip():
    def __init__(self):
        pass

    def __call__(self, sample):
        img, label = sample
        temp = np.random.random()
        if temp > 0 and temp < 0.25:
            return img.transpose(Image.FLIP_LEFT_RIGHT), label.transpose(Image.FLIP_LEFT_RIGHT)
        elif temp >= 0.25 and temp < 0.5:
            return img.transpose(Image.FLIP_TOP_BOTTOM), label.transpose(Image.FLIP_TOP_BOTTOM)
        elif temp >= 0.5 and temp < 0.75:
            return img.transpose(Image.ROTATE_90), label.transpose(Image.ROTATE_90)
        else:
            return img, label


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes

        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(),
                                                                                                  target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


'''
def calculate_metric_percase(output, target):
    smooth = 1e-5  

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    if output.sum() > 0 and target.sum() > 0:
        hd = metric.binary.hd(output, target)
    else:
        hd = 0
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
           (output.sum() + target.sum() + smooth), hd
'''


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1, 0
    else:
        return 0, 0


def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1,device_id=0):
    if len(image.shape) ==4:
        image, label = image.squeeze(0).unsqueeze(1).cuda(device_id).float(),label.squeeze(0).cpu().detach().numpy()
    else:
        pass
    prediction = np.zeros_like((label))
    net.eval()
    with torch.no_grad():
        out = torch.argmax(torch.softmax(net(image), dim=1), dim=1)
        pred = out.cpu().detach().numpy()
    x, y = label.shape[-2:]
    if x != patch_size[0] or y != patch_size[1]:
        for slice_i in range(prediction.shape[0]):
            prediction[slice_i] = zoom(pred[slice_i], (x / patch_size[0], y / patch_size[1]), order=0)
    else :
        prediction = pred


    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
    image= image.squeeze(1).cpu().detach().numpy()
    if test_save_path is not None:
        print('saving test results...' +test_save_path)
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    return metric_list


def cleanup_old_models(save_path, current_avg_dcs, with_gabor):
    """
    Delete older model files in the save directory that have the same `with_gabor`
    setting and a lower average Dice score (avg_dcs) than the current model.
    """
    try:
        # List all files in the directory
        for filename in os.listdir(save_path):
            if filename.endswith(".pth"):
                # Extract avg_dcs and with_gabor value from the filename
                avg_dcs_start_idx = filename.index("avg_dcs=") + len("avd_dcs=")
                avg_dcs_end_idx = filename.index("_", avg_dcs_start_idx)
                file_avg_dcs = float(filename[avg_dcs_start_idx:avg_dcs_end_idx])

                gabor_start_idx = filename.index("_withgabor_") + len("_withgabor_")
                gabor_end_idx = filename.index(".pth", gabor_start_idx)
                file_with_gabor = (filename[gabor_start_idx:gabor_end_idx]) == "True"
                # Check if the file matches the current `with_gabor` setting and has a lower avg_dcs
                if file_with_gabor == with_gabor and file_avg_dcs < current_avg_dcs:
                    # Delete the file
                    os.remove(os.path.join(save_path, filename))
    except Exception as e:
        print("Failed to clean up old models:", e)



def add_results_to_experimets_file(parameters, result):
    # open the experiments.txt file and append the results
    with open("experiments.txt", "a") as myfile:
        fcntl.flock(myfile, fcntl.LOCK_EX)
        try:

            myfile.write(str(parameters) +"     " +f'{result}' + '\n')
            myfile.write("-" * 60 + "\n")
        finally:
            fcntl.flock(myfile, fcntl.LOCK_UN)