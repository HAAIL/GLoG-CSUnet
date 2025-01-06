#!/usr/bin/env python
# -*- coding:utf-8 -*-
import sys
import logging
import uuid

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
import torchvision
# import matplotlib.pyplot as plt
from utils.utils import DiceLoss, cleanup_old_models, add_results_to_experimets_file
from torch.utils.data import DataLoader
from dataset.dataset_ACDC import ACDCdataset, RandomGenerator
from tqdm import tqdm
import os
from torchvision import transforms
from utils.test_ACDC import inference
from medpy.metric import dc, hd95
import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.CSUnet.vision_transformer import CS_Unet
from networks.CSUnet.config import get_config
from networks.GLoGNet import GLoGNet
import wandb
from wandb_utils import wandb_init


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        default='ACDC', help='experiment_name')
    parser.add_argument("--list_dir", default="./lists/lists_ACDC")
    parser.add_argument("--root_path", default="./data/ACDC/")
    parser.add_argument("--volume_path", default="./data/ACDC/test")
    parser.add_argument("--z_spacing", default=10)
    parser.add_argument("--num_classes", default=4)
    parser.add_argument('--output_dir', type=str,
                        default='./checkpoint/ACDC/csunet/', help='output dir')
    parser.add_argument('--test_save_dir', default=None, help='saving prediction as nii!')
    parser.add_argument('--max_epochs', type=int, default=300, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int, default=24, help='batch_size per gpu')
    parser.add_argument("--n_skip", type=int, default=1)
    parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
    parser.add_argument('--base_lr', type=float, default=0.01, help='segmentation network learning rate')
    parser.add_argument('--base_weight', type=float, default=0.0002, help='segmentation network learning rate')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='warmup_epochs')
    parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument(
        '--cfg',
        default="./pretrained_ckpt/swin_tiny_patch4_window7_224_lite.yaml",
        type=str,
        metavar="FILE",
        help='path to config file'
    )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    parser.add_argument("--checkpoint", help="checkpoint file path",default="")
    parser.add_argument("--with_gabor", action='store_true')
    parser.add_argument("--n_gabor_filter", default=5, type=int)
    parser.add_argument("--gabor_filter_kernel_size", default=3, type=int)
    parser.add_argument("--with_LoG", action='store_true')
    parser.add_argument("--n_LoG_filter", default=5, type=int)
    parser.add_argument("--LoG_filter_kernel_size", default=3, type=int)

    parser.add_argument("--project", default="GLoGNet")
    parser.add_argument("--group", default="GLoG")
    parser.add_argument("--name", default="GLoGNet")
    parser.set_defaults(with_gabor=True, resume=False, with_LoG=True)
    args = parser.parse_args()
    return args





def main(args, device_id=0):
    config = get_config(args)

    # create log dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    logging.basicConfig(filename=args.output_dir + "/log.txt", level=logging.INFO,
                        format='%(asctime)s   %(levelname)s   %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_name = args.dataset

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    train_dataset = ACDCdataset(args.root_path, args.list_dir, split="train", transform=
    transforms.Compose(
        [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    Train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    db_val = ACDCdataset(base_dir=args.root_path, list_dir=args.list_dir, split="valid")
    valloader = DataLoader(db_val, batch_size=1, shuffle=False)
    db_test = ACDCdataset(base_dir=args.volume_path, list_dir=args.list_dir, split="test")
    testloader = DataLoader(db_test, batch_size=1, shuffle=False)

    base_model = CS_Unet(config, img_size=args.img_size, num_classes=args.num_classes).cuda(device_id)

    model = GLoGNet(base_model, with_gabor=args.with_gabor, n_gabor_filters=args.n_gabor_filter,
                    gabor_filter_kernel_size=args.gabor_filter_kernel_size, with_LoG=args.with_LoG,
                    n_LoG_filters=args.n_LoG_filter,
                    LoG_filter_kernel_size=args.LoG_filter_kernel_size)



    if args.resume:
        model.load_state_dict(torch.load(args.checkpoint))

    model = model.cuda(device_id)
    inference(args, model, testloader, None, device_id=device_id)
    model.train()


    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(args.num_classes)

    iter_num = 0
    Test_Accuracy = []

    avg_dcs = -np.inf
    Best_dcs = 0.88

    max_iterations = args.max_epochs * len(Train_loader)

    base_lr = args.base_lr
    base_weight = args.base_weight
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=base_weight)


    for epoch in range(args.max_epochs):
        model.train()
        train_loss = 0
        with tqdm(desc='Epoch %d - train' % (epoch),
                  unit='it', total=len(Train_loader)) as pbar:

            for i_batch, sampled_batch in enumerate(Train_loader):
                image_batch, label_batch = sampled_batch["image"], sampled_batch["label"]
                image_batch, label_batch = image_batch.type(torch.FloatTensor), label_batch.type(torch.FloatTensor)
                image_batch, label_batch = image_batch.cuda(device_id), label_batch.cuda(device_id)

                outputs = model(image_batch)

                loss_ce = ce_loss(outputs, label_batch[:].long())
                loss_dice = dice_loss(outputs, label_batch[:], softmax=True)
                loss = loss_dice * 0.5 + loss_ce * 0.5

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_



                iter_num = iter_num + 1

                train_loss += loss.item()
                pbar.set_postfix(loss=train_loss / (i_batch + 1), lr=lr_)
                pbar.update()


        # ---------- Validation ----------
        if (train_loss / (i_batch + 1)) < 0.03:
            dc_sum = 0
            model.eval()
            for i, val_sampled_batch in enumerate(valloader):
                val_image_batch, val_label_batch = val_sampled_batch["image"], val_sampled_batch["label"]
                val_image_batch, val_label_batch = val_image_batch.type(torch.FloatTensor), val_label_batch.type(
                    torch.FloatTensor)
                val_image_batch, val_label_batch = val_image_batch.cuda(device_id).unsqueeze(
                    1), val_label_batch.cuda(device_id).unsqueeze(1)

                val_outputs = model(val_image_batch)
                val_outputs = torch.argmax(torch.softmax(val_outputs, dim=1), dim=1).squeeze(0)
                dc_sum += dc(val_outputs.cpu().data.numpy(), val_label_batch[:].cpu().data.numpy())
                avg_dcs = dc_sum / len(valloader)
            logging.info("Validation ===>avg_dsc: %f" % avg_dcs)

            if avg_dcs > Best_dcs:

                Best_dcs = avg_dcs

                # ---------- Test ----------
                avg_dcs, avg_hd = inference(args, model, testloader, args.test_save_dir,
                                            device_id=device_id)  # args.test_save_dir
                Test_Accuracy.append(avg_dcs)
                cleanup_old_models(args.output_dir, avg_dcs, args.with_gabor)
                save_mode_path = os.path.join(args.output_dir,
                                              'epoch={}_avg_dcs={}_withgabor_{}_n_gabor{}_withLoG{}_n_log{}.pth'.format(
                                                  epoch, avg_dcs,
                                                  args.with_gabor, args.n_gabor_filter, args.with_LoG,
                                                  args.n_LoG_filter))
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))



        if epoch >= args.max_epochs - 1:
            # ---------- Test ----------
            avg_dcs, avg_hd = inference(args, model, testloader, None, device_id=device_id)
            cleanup_old_models(args.output_dir, avg_dcs, args.with_gabor)
            save_mode_path = os.path.join(args.output_dir,
                                          'epoch={}_avg_dcs={}_lr={}_withgabor_{}_n_gabor{}_withLoG{}_n_log{}'.format(
                                              epoch, avg_dcs, lr_,
                                              args.with_gabor, args.n_gabor_filter, args.with_LoG, args.n_LoG_filter))
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

            Test_Accuracy.append(avg_dcs)
            wandb.log({"Best_dcs": Best_dcs})
            print("max:", max(Test_Accuracy))
            print("last:", (Test_Accuracy[-1]))

            #
            vars(args).update({"dataset": "ACDC",
                               "base_model": "CSUnet", })

            add_results_to_experimets_file(vars(args), {"dsc": Test_Accuracy[-1], "max_dsc": max(Test_Accuracy)})

            break

if __name__ == "__main__":
        args = get_args()
        args.name="GLoGNet"+str(uuid.uuid4())
        wandb_init(vars(args))
        main(args,device_id=1)
        wandb.finish()
