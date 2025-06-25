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
from trainer_TAO_CLS import run_training
from utils.MM_CLS_TAO_data_utils_v1 import get_loader
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from model.CLS.mm_classification_SwinUnter import MM_SwinUnter_Classification
from model.CLS.mm_classification_Foundation_model_plus import Foundation_Model_Classification
from monai.losses import FocalLoss

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser(description="Swin UNETR segmentation pipeline")
parser.add_argument("--checkpoint", default=None, help="start training from saved checkpoint")
parser.add_argument("--logdir", default="VLM_cls_4", type=str, help="directory to save the tensorboard logs")
parser.add_argument(
    "--pretrained_dir", default="./pretrained_models/", type=str, help="pretrained checkpoint directory"
)
parser.add_argument("--data_dir", default="./dataset/mm_tao_cls/", type=str, help="dataset directory")
parser.add_argument("--json_list", default="mm_tao_cls_4_label_test.json", type=str, help="dataset json file")
parser.add_argument("--save_checkpoint", default=1, type=int, help="save checkpoint during training")
parser.add_argument("--max_epochs", default=500, type=int, help="max number of training epochs")
parser.add_argument("--batch_size", default=1, type=int, help="number of batch size")
parser.add_argument("--sw_batch_size", default=4, type=int, help="number of sliding window batch size")
parser.add_argument("--optim_lr", default=1e-4, type=float, help="optimization learning rate")
parser.add_argument("--optim_name", default="adamw", type=str, help="optimization algorithm")
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
parser.add_argument("--out_channels", default=4, type=int, help="number of output channels")
parser.add_argument("--use_normal_dataset", default=1, type=int, help="use monai Dataset class")
parser.add_argument('--train_modality', default='MRI', type=str, choices=['CT', 'MRI', 'unlabeled'], help='CT or MRI' or 'unlabeled')
parser.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=250.0, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=2.0, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=128, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=128, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=128, type=int, help="roi size in z direction")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")
parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
parser.add_argument("--lrschedule", default="warmup_cosine", type=str, help="type of learning rate scheduler")
parser.add_argument("--warmup_epochs", default=50, type=int, help="number of warmup epochs")
parser.add_argument("--resume_ckpt", action="store_true", help="resume training from pretrained checkpoint")
parser.add_argument("--smooth_dr", default=1e-6, type=float, help="constant added to dice denominator to avoid nan")
parser.add_argument("--smooth_nr", default=0.0, type=float, help="constant added to dice numerator to avoid zero")
parser.add_argument("--use_checkpoint", default=1, type=int, help="use gradient checkpointing to save memory")
parser.add_argument("--use_ssl_pretrained", default=0, type=int, help="use self-supervised pretrained weights")
parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
parser.add_argument("--squared_dice", default=1, type=int, help="use squared Dice")
parser.add_argument('--backbone', default='SwinUNETR', choices=['Foundation_model','Semi_SM_model', 'SwinUNETR', 'VIT3D'], help='backbone [Foundation_model or SwinUNETR or VIT3D]')
parser.add_argument('--pretrain_dir', default=f"./run/Foundation_model.pth", type=str)
parser.add_argument('--CLIP_text_pretrain_dir', default=f"./Text-emmbedding-gen/TAO_clip_txt_encoding.pth", type=str)
parser.add_argument('--Bert_text_pretrain_dir', default=f"./Text-emmbedding-gen/TAO_bert_txt_encoding.pth", type=str)
parser.add_argument('--pretrain', default=1, type=int)
parser.add_argument('--text_encoding_type', default='word_embedding', choices=['word_embedding','rand_embedding', 'None'],
                    help='the type of encoding: rand_embedding or word_embedding')
parser.add_argument('--text_prompt_name', default='Bert_embedding', choices=['CLIP_embedding','Bert_embedding', 'None'], help='text embedding type')
parser.add_argument('--use_text_prompt', default=1, type=int)
parser.add_argument('--text_prompt_loss', default=1, type=int)
parser.add_argument('--fusion_module', default='Cross_Attention', choices=['DoubleAttention', 'Attention_Fusion', 'Cross_Attention', 'SingleAttention', 'Concat_Fusion'])
parser.add_argument('--res_depth', default=101, type=int, choices=[18, 34, 50, 101, 152])
parser.add_argument('--hybrid_loss', default='focal_loss', choices=['ce_loss', 'focal_loss', 'hybrid_loss'] )

def main():
    args = parser.parse_args()
    args.amp = not args.noamp

    sw_batch_size = "sw_batch_size:" + str(args.sw_batch_size)
    use_text_prompt = "use_text_prompt:" + str(args.use_text_prompt)
    text_prompt_loss = "text_prompt_loss:" + str(args.text_prompt_loss)
    res_depth = str(args.res_depth)
    args.logdir = "./out_liver/" + args.logdir + "_" + args.text_prompt_name + "_" + args.fusion_module + "_" + use_text_prompt + "_" + text_prompt_loss+ "_" + res_depth+"_" + args.hybrid_loss
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
    t1_loader = get_loader(args,train_modality='t1')
    t1c_loader = get_loader(args,train_modality='t1c')

    print(args.rank, " gpu", args.gpu)
    if args.rank == 0:
        print("Batch size is:", args.batch_size, "epochs", args.max_epochs)
    if args.backbone =="Foundation_model":
        model = Foundation_Model_Classification(n_class=args.out_channels,
                                                text_prompt=args.use_text_prompt,
                                                fusion_module=args.fusion_module,
                                                text_prompt_name=args.text_prompt_name,
                                                text_encoding=args.text_encoding_type,
                                                res_depth=args.res_depth)
    elif args.backbone=="SwinUNETR":
        model = MM_SwinUnter_Classification(n_class=args.out_channels,
                                                text_prompt=args.use_text_prompt,
                                                fusion_module=args.fusion_module,
                                                text_prompt_name=args.text_prompt_name,
                                                text_encoding=args.text_encoding_type,
                                                res_depth=args.res_depth)

    if args.pretrain:
        model.load_params(torch.load(args.pretrain_dir, map_location='cpu')['net'])
    if args.text_prompt_name=="CLIP_embedding":
        args.use_text_prompt=True
        word_embedding = torch.load(args.CLIP_text_pretrain_dir)
        model.VLM_branch.organ_embedding.data = word_embedding.float()
        print('load CLIP word embedding')
    elif args.text_prompt_name=="Bert_embedding":
        args.use_text_prompt=True
        word_embedding = torch.load(args.Bert_text_pretrain_dir)
        model.VLM_branch.organ_embedding.data = word_embedding.float()
        print('load CLIP word embedding')
    if args.resume_ckpt:
        resume_ckpt_path = os.path.join(args.logdir,"model_final.pt")
        model_dict = torch.load(os.path.join(resume_ckpt_path, args.pretrained_model_name))["state_dict"]
        model.load_state_dict(model_dict)
        print("Use pretrained weights")

    loss_function = torch.nn.BCEWithLogitsLoss()
    focal_loss_function = FocalLoss(
        to_onehot_y=False,  # 是否自动 one-hot 标签
        # sigmoid=True,
        gamma=2.0,  # 聚焦因子
        alpha=0.25  # 类别权重（可选）
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

    model.cuda(args.gpu)
    torch.cuda.empty_cache()

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        if args.norm_name == "batch":
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu, find_unused_parameters=True)
    if args.optim_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
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
        loss_func=loss_function,
        focal_loss_func=focal_loss_function,
        args=args,
        scheduler=scheduler,
        start_epoch=start_epoch,
    )
    return accuracy


if __name__ == "__main__":
    main()
