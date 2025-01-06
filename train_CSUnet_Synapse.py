
import sys
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
from utils.utils import DiceLoss, cleanup_old_models, add_results_to_experimets_file
from torch.utils.data import DataLoader
from dataset.dataset_Synapse import Synapsedataset, RandomGenerator
from tqdm import tqdm
from torchvision import transforms
from utils.test_Synapse import inference
import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.CSUnet.vision_transformer import CS_Unet as ViT_seg
from networks.CSUnet.config import get_config
from networks.GLoGNet import GLoGNet
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument("--list_dir", default="./lists/lists_Synapse")
parser.add_argument("--root_path", default="./data/Synapse/")
parser.add_argument("--volume_path", default="./data/Synapse/test_vol_h5")
parser.add_argument("--z_spacing", default=1)
parser.add_argument("--num_classes", default=9)

parser.add_argument('--output_dir', type=str,
                    default='./checkpoint/Synapse/csunet_conv/', help='output dir')
parser.add_argument('--test_save_dir', default=None, help='saving prediction as nii!')
parser.add_argument('--max_epochs', type=int, default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24, help='batch_size per gpu')
parser.add_argument("--n_skip", type=int, default=4)
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=1e-3, help='segmentation network learning rate')
parser.add_argument('--base_weight', type=float, default=2e-4, help='segmentation network learning rate')
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
parser.add_argument('--checkpoint', type=str, help='checkpoint file',
                    default="")
parser.add_argument("--with_gabor", action='store_true')
parser.add_argument("--n_gabor_filter", default=2, type=int)
parser.add_argument("--gabor_filter_kernel_size", default=3, type=int)
parser.add_argument("--with_LoG", action='store_true')
parser.add_argument("--n_LoG_filter", default=5, type=int)
parser.add_argument("--LoG_filter_kernel_size", default=3, type=int)
parser.set_defaults(with_gabor=True, resume=False, with_LoG=True)

args = parser.parse_args()
config = get_config(args)
device_id = 1


def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)


if __name__ == "__main__":

    logging.basicConfig(filename=args.output_dir + "/log.txt", level=logging.INFO,
                        format='%(asctime)s   %(levelname)s   %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    writer = SummaryWriter(args.output_dir + '/log')

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

    db_train = Synapsedataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                              transform=transforms.Compose(
                                  [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    trainloader = DataLoader(db_train, batch_size=args.batch_size, shuffle=True,
                             worker_init_fn=worker_init_fn)
    print("The length of train set is: {}".format(len(db_train)))

    db_test = Synapsedataset(base_dir=args.volume_path, split="test", list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    print("The test iterations per epoch is: {}".format(len(testloader)))

    # -------- model -------------------------------
    base_model = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes).cuda(device_id)
    # base_model.load_from(config)
    model = GLoGNet(base_model, with_gabor=args.with_gabor, n_gabor_filters=args.n_gabor_filter,
                    gabor_filter_kernel_size=args.gabor_filter_kernel_size, with_LoG=args.with_LoG, n_LoG_filters=args.n_LoG_filter,
                    LoG_filter_kernel_size=args.LoG_filter_kernel_size)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    if args.resume:
        state_dict=torch.load(args.checkpoint)
        model.load_state_dict(state_dict, strict=False)


    model = model.cuda(device_id)
    # -------- inference -------------------------------

    # avg_dcs, avg_hd = inference(args, model, testloader, args.test_save_dir, device_id=device_id)
    # -------- training -------------------------------

    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(args.num_classes)
    save_interval = args.n_skip  # int(max_epoch/6)

    iter_num = 0
    Loss = []
    Test_Accuracy = []

    best_performance = 0.75

    max_iterations = args.max_epochs * len(trainloader)

    base_lr = args.base_lr
    base_weight = args.base_weight
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=base_weight)

    for epoch in range(args.max_epochs):
        model.train()
        train_loss = 0
        with tqdm(desc='Epoch %d - train' % (epoch),
                  unit='it', total=len(trainloader)) as pbar:
            for i_batch, sampled_batch in enumerate(trainloader):
                image_batch, label_batch = sampled_batch["image"], sampled_batch["label"]
                image_batch, label_batch = image_batch.type(torch.FloatTensor), label_batch.type(torch.FloatTensor)
                image_batch, label_batch = image_batch.cuda(device_id), label_batch.cuda(device_id)

                outputs = model(image_batch)

                loss_ce = ce_loss(outputs, label_batch[:].long())
                loss_dice = dice_loss(outputs, label_batch[:], softmax=True)
                loss = loss_dice * 0.6 + loss_ce * 0.4

                optimizer.zero_grad()

                loss.backward()
                optimizer.step()

                lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

                iter_num = iter_num + 1

                train_loss += loss.item()
                writer.add_scalar('info/lr', lr_, iter_num)
                writer.add_scalar('info/total_loss', loss, iter_num)
                writer.add_scalar('info/loss_ce', loss_ce, iter_num)
                # logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))
                pbar.set_postfix(loss=train_loss / (i_batch + 1), lr=lr_)
                pbar.update()

                if iter_num % 20 == 0:
                    image = image_batch[1, 0:1, :, :]
                    image = (image - image.min()) / (image.max() - image.min())
                    writer.add_image('train/Image', image, iter_num)
                    outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                    writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                    labs = label_batch[1, ...].unsqueeze(0) * 50
                    writer.add_image('train/GroundTruth', labs, iter_num)

        # ---------- Validation ----------
        if (train_loss / (i_batch + 1))<0.031:
            avg_dcs, avg_hd = inference(args, model, testloader, args.test_save_dir, device_id=device_id)

            if avg_dcs > best_performance:
                best_performance = avg_dcs
                cleanup_old_models(args.output_dir, best_performance, args.with_gabor)
                save_mode_path = os.path.join(args.output_dir,
                                              'best_epoch{}_avg_dcs={}_withgabor_{}_n_gabor{}_withLoG{}_n_log{}.pth'.format(epoch, best_performance,
                                                                                                args.with_gabor, args.n_gabor_filter, args.with_LoG, args.n_LoG_filter))
                torch.save(model.state_dict(), save_mode_path)
                print("save model to {}".format(save_mode_path))

        if epoch >= args.max_epochs - 1:
            avg_dcs, avg_hd = inference(args, model, testloader, args.test_save_dir, device_id=device_id)
            cleanup_old_models(args.output_dir, avg_dcs, args.with_gabor)
            save_mode_path = os.path.join(args.output_dir,
                                          'last_epoch{}_avg_dcs={}_withgabor_{}_n_gabor{}_withLoG{}_n_log{}.pth'.format(
                                              epoch, avg_dcs,
                                              args.with_gabor, args.n_gabor_filter, args.with_LoG, args.n_LoG_filter))
            torch.save(model.state_dict(), save_mode_path)
            print("save model to {}".format(save_mode_path))




            print("best_performance:",best_performance)
            print("last_performance:",avg_dcs)
            vars(args).update({"dataset": "Synapse",
                               "base_model": "CSUnet", })

            add_results_to_experimets_file(vars(args), {"dsc": avg_dcs, "max_dsc": best_performance})

    writer.close()
    logging.info('End of training')
