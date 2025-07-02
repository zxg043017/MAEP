# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.utils.data.distributed
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from MM_trainer_TAO import run_training
from utils.mm_TAO500_data_utils_test import get_loader

from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from model.SwinUNETR_ResNet import SwinUNETR
from model.mm_Unet_Resnet import UNet3D as Foundatiom_model

from monai.transforms import AsDiscrete
from monai.utils.enums import MetricReduction
from monai.losses import ContrastiveLoss
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.checkpoint")

parser = argparse.ArgumentParser(description="Swin UNETR segmentation pipeline")
parser.add_argument("--checkpoint", default=None, help="start training from saved checkpoint")
parser.add_argument("--logdir", default="FM_TAO", type=str, help="directory to save the tensorboard logs")
parser.add_argument(
    "--pretrained_dir", default="./pretrained_models/", type=str, help="pretrained checkpoint directory")
parser.add_argument("--data_dir", default="./dataset/TAO/", type=str, help="dataset directory")
parser.add_argument("--json_list", default="AP_TAO.json", type=str, help="dataset json file")
parser.add_argument(
    "--pretrained_model_name",
    default="swin_unetr.epoch.b4_5000ep_f48_lr2e-4_pretrained.pt",
    type=str,
    help="pretrained model name",
)
parser.add_argument("--save_checkpoint", default=1, type=int, help="save checkpoint during training")
parser.add_argument("--max_epochs", default=500, type=int, help="max number of training epochs")
parser.add_argument("--batch_size", default=4, type=int, help="number of batch size")
parser.add_argument("--sw_batch_size", default=4, type=int, help="number of sliding window batch size")
parser.add_argument("--optim_lr", default=0.0003, type=float, help="optimization learning rate")
parser.add_argument("--optim_name", default="adam", type=str, help="optimization algorithm")
parser.add_argument("--reg_weight", default=1e-5, type=float, help="regularization weight")
parser.add_argument("--momentum", default=0.99, type=float, help="momentum")
parser.add_argument("--noamp", action="store_true", help="do NOT use amp for training")
parser.add_argument("--val_every", default=1, type=int, help="validation frequency")
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--world_size", default=1, type=int, help="number of nodes for distributed training")
parser.add_argument("--rank", default=0, type=int, help="node rank for distributed training")
parser.add_argument("--dist-url", default="tcp://127.0.0.1:23456", type=str, help="distributed url")
parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
parser.add_argument("--norm_name", default="instance", type=str, help="normalization name")
parser.add_argument("--workers", default=0, type=int, help="number of workers")
parser.add_argument("--feature_size", default=48, type=int, help="feature size")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=10, type=int, help="number of output channels")
parser.add_argument("--use_normal_dataset", default=1, type=int, help="use monai Dataset class")
parser.add_argument('--train_modality', default='MRI', type=str, choices=['CT', 'MRI', 'unlabeled'], help='CT or MRI' or 'unlabeled')
parser.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=250.0, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=2.0, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=64, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=64, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=32, type=int, help="roi size in z direction")
parser.add_argument("--val_roi_x", default=96, type=int, help="roi size in x direction")
parser.add_argument("--val_roi_y", default=96, type=int, help="roi size in y direction")
parser.add_argument("--val_roi_z", default=32, type=int, help="roi size in z direction")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")
parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
parser.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")
parser.add_argument("--lrschedule", default="warmup_cosine", type=str, help="type of learning rate scheduler")
parser.add_argument("--warmup_epochs", default=50, type=int, help="number of warmup epochs")
parser.add_argument("--resume_ckpt", action="store_true", help="resume training from pretrained checkpoint")
parser.add_argument("--smooth_dr", default=1e-6, type=float, help="constant added to dice denominator to avoid nan")
parser.add_argument("--smooth_nr", default=0.0, type=float, help="constant added to dice numerator to avoid zero")
# parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")
parser.add_argument("--use_checkpoint", default=1, type=int, help="use gradient checkpointing to save memory")
# parser.add_argument("--use_ssl_pretrained", action="store_true", help="use self-supervised pretrained weights")
parser.add_argument("--use_ssl_pretrained", default=1, type=int, help="use self-supervised pretrained weights")
parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
parser.add_argument("--squared_dice", action="store_true", help="use squared Dice")
parser.add_argument('--start_fusion_epoch', default=550, type=int)
parser.add_argument('--backbone', default='Foundation_model', choices=['Foundation_model', 'SwinUNETR', 'VIT3D'], help='backbone [Foundation_model or SwinUNETR or VIT3D]')
parser.add_argument('--loss_opt', default='CSC', type=str, choices=['CSC', 'CAC', 'ALL', 'NONE'], help='select for loss')
def CAC_loss(pred1, pred2, similarity='cosine'):
    """
    Compute CAC loss
    """
    smooth = 1e-6
    dim_len = len(pred1.size())
    if dim_len == 5:
       dim=(2,3,4)
    elif dim_len == 4:
       dim=(2,3)
    intersect = torch.sum(pred1 * pred2,dim=dim)
    y_sum = torch.sum(pred1 * pred1,dim=dim)
    z_sum = torch.sum(pred2 * pred2,dim=dim)
    dice_sim = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    return dice_sim.mean()
def CSC_loss(pred1,pred2):
    channel_losses = 0.0
    lens = pred1.shape[0]
    for c in range(pred1.shape[0]):
        pred1_output_channel = pred1[c, :, :, :]  # select the c_th channel of predi
        pred2_output_channel = pred2[c, :, :, :]  # select the c_th channel of pred2
        pred1_2d_flat = pred1_output_channel.reshape(-1, pred1_output_channel.shape[0])  # resize shape
        pred2_2d_flat = pred2_output_channel.reshape(-1, pred2_output_channel.shape[0])  # resize shape

        # compute the ContrastiveLoss from each channel
        cl_loss = ContrastiveLoss(batch_size=2, temperature=0.5)
        cl_value = cl_loss(pred1_2d_flat, pred2_2d_flat)
        channel_losses = channel_losses + cl_value
    mean_loss = channel_losses/ lens
    return mean_loss

def main():
    args = parser.parse_args()
    args.amp = not args.noamp
    args.logdir = "./runs/" + args.logdir
    if args.distributed:
        args.ngpus_per_node = torch.cuda.device_count()
        print("Found total gpus", args.ngpus_per_node)
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args,))
    else:
        main_worker(gpu=0, args=args)


def main_worker(gpu, args):
    if args.distributed:
        torch.multiprocessing.set_start_method("fork", force=True)
    np.set_printoptions(formatter={"float": "{: 0.3f}".format}, suppress=True)
    args.gpu = gpu
    if args.distributed:
        args.rank = args.rank * args.ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
        )
    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True
    args.test_mode = False

    ### load dataset
    t1_loader = get_loader(args,train_modality='T1')
    t1c_loader = get_loader(args,train_modality='T1c')

    print(args.rank, " gpu", args.gpu)
    if args.rank == 0:
        print("Batch size is:", args.batch_size, "epochs", args.max_epochs)
    inf_size = [args.roi_x, args.roi_y, args.roi_z]

    pretrained_dir = args.pretrained_dir
    if args.backbone == "Foundation_model":
        model = Foundatiom_model(n_class=args.out_channels)
    elif args.backbone == "SwinUNETR":
        model = SwinUNETR(
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            feature_size=args.feature_size,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=args.dropout_path_rate,
            use_checkpoint=args.use_checkpoint,
        )

    if args.resume_ckpt:
        model_dict = torch.load(os.path.join(pretrained_dir, args.pretrained_model_name))["state_dict"]
        model.load_state_dict(model_dict)
        print("Use pretrained weights")

    if args.use_ssl_pretrained:
        try:
            if args.backbone == "Foundation_model":
                model_dict = torch.load("./pretrained_models/unet.pth")
                state_dict = model_dict["net"]
                if "module." in list(state_dict.keys())[0]:
                    print("Tag 'module.' found in state dict - fixing!")
                    for key in list(state_dict.keys()):
                        state_dict[key.replace("module.backbone.", "")] = state_dict.pop(key)

                model.load_state_dict(state_dict, strict=False)
                print("Using pretrained foundation model backbone weights !")

            elif args.backbone == "SwinUNETR":
                model_dict = torch.load("./pretrained_models/swin_unetr_pretrained.pt")
                state_dict = model_dict["state_dict"]
                tmp_model_state_dict = model.state_dict()
                # fix potential differences in state dict keys from pre-training to
                # fine-tuning
                if "module." in list(state_dict.keys())[0]:
                    print("Tag 'module.' found in state dict - fixing!")
                    for key in list(state_dict.keys()):
                        state_dict[key.replace("module.", "")] = state_dict.pop(key)
                if "swinViT" in list(state_dict.keys())[0]:
                    print("Tag 'swin_vit' found in state dict - fixing!")
                    for key in list(state_dict.keys()):
                        state_dict[key.replace("swin_vit", "swinViT")] = state_dict.pop(key)
                        if "out.conv.conv" not in key:
                            # print("key",key)
                            tmp_model_state_dict[key] = state_dict.pop(key)
                            # state_dict[key] = state_dict[key]
                # We now load model weights, setting param `strict` to False, i.e.:
                # this load the encoder weights (Swin-ViT, SSL pre-trained), but leaves
                # the decoder weights untouched (CNN UNet decoder).
                model.load_state_dict(tmp_model_state_dict, strict=False)
                print("Using pretrained self-supervised Swin UNETR backbone weights !")
        except ValueError:
            raise ValueError("Self-supervised pre-trained weights not available for" + str(args.model_name))
    if args.squared_dice:
        dice_loss = DiceCELoss(
            to_onehot_y=True, softmax=True, squared_pred=True, smooth_nr=args.smooth_nr, smooth_dr=args.smooth_dr
        )
    else:
        dice_loss = DiceCELoss(to_onehot_y=True, softmax=True)

    weights = torch.tensor([1.0, 2.0, 0.5])  # 第二类权重更高
    # ce_loss = CrossEntropyLoss(weight=weights, abel_smoothing=0.1)
    # ce_loss = losses.CrossEntropyLoss()
    ce_loss = torch.nn.CrossEntropyLoss().cuda()
    dice_ce_loss = DiceCELoss(to_onehot_y=True, softmax=True, lambda_dice=1.0, lambda_ce=1.0)
    post_label = AsDiscrete(to_onehot=args.out_channels)
    post_pred = AsDiscrete(argmax=True, to_onehot=args.out_channels)
    dice_acc = DiceMetric(include_background=False, reduction=MetricReduction.MEAN, get_not_nans=True)
    model_inferer = partial(
        sliding_window_inference,
        roi_size=inf_size,
        sw_batch_size=args.sw_batch_size,
        predictor=model,
        overlap=args.infer_overlap,
    )
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters count", pytorch_total_params)
    best_acc = 0
    start_epoch = 0
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint["state_dict"].items():
            new_state_dict[k.replace("backbone.", "")] = v
        model.load_state_dict(new_state_dict, strict=False)
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"]
        if "best_acc" in checkpoint:
            best_acc = checkpoint["best_acc"]
        print("=> loaded checkpoint '{}' (epoch {}) (bestacc {})".format(args.checkpoint, start_epoch, best_acc))
    unlabeled_model = model
    model.cuda(args.gpu)
    unlabeled_model.cuda(args.gpu)
    torch.cuda.empty_cache()
    if args.distributed:
        torch.cuda.set_device(args.gpu)
        if args.norm_name == "batch":
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu)
    if args.optim_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.optim_lr, betas=(0.9, 0.999), weight_decay=args.reg_weight)
    elif args.optim_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.optim_lr, momentum=args.momentum, nesterov=True, weight_decay=args.reg_weight
        )
    else:
        raise ValueError("Unsupported Optimization Procedure: " + str(args.optim_name))
    if args.lrschedule == "warmup_cosine":
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.max_epochs
        )
    elif args.lrschedule == "cosine_anneal":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
        if args.checkpoint is not None:
            scheduler.step(epoch=start_epoch)
    else:
        scheduler = None
    accuracy = run_training(
        model=model,
        t1_train_loader=t1_loader[0],
        t1_val_loader=t1_loader[1],
        t1c_train_loader=t1c_loader[0],
        t1c_val_loader=t1c_loader[1],
        optimizer=optimizer,
        loss_func=dice_ce_loss,
        acc_func=dice_acc,
        args=args,
        model_inferer=model_inferer,
        scheduler=scheduler,
        start_epoch=start_epoch,
        post_label=post_label,
        post_pred=post_pred,
    )
    return accuracy


if __name__ == "__main__":
    main()
