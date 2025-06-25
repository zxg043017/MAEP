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

import nibabel as nib
import numpy as np
import torch
from utils.mm_TAO500_data_utils import get_loader
from utils.utils import dice, resample_3d
from utils.MM_TAO_data_utils import get_loader
from model.mm_Unet_Resnet import UNet3D as Foundatiom_model

from monai.inferers import sliding_window_inference
# from monai.networks.nets import SwinUNETR
# from AMOS.model.bak.SwinUNETR import SwinUNETR
from model.SwinUNETR_ResNet import SwinUNETR
from monai.data import decollate_batch
from monai.transforms import AsDiscrete
from monai.metrics import DiceMetric, SurfaceDistanceMetric
from monai.utils.enums import MetricReduction

parser = argparse.ArgumentParser(description="Swin UNETR segmentation pipeline")
parser.add_argument(
    "--pretrained_dir", default="./runs/Best_63_dice_ce_unet/", type=str, help="pretrained checkpoint directory"
)
parser.add_argument("--data_dir", default="./dataset/TAO/", type=str, help="dataset directory")
parser.add_argument("--exp_name", default="test_TOM_best", type=str, help="experiment name")
parser.add_argument("--json_list", default="AP_TAO_test.json", type=str, help="dataset json file")
parser.add_argument(
    "--pretrained_model_name",
    default="swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt",
    type=str,
    help="pretrained model name",
)
parser.add_argument("--feature_size", default=48, type=int, help="feature size")
parser.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=10, type=int, help="number of output channels")
parser.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=250.0, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=2.0, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=128, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=32, type=int, help="roi size in z direction")
parser.add_argument("--val_roi_x", default=96, type=int, help="roi size in x direction")
parser.add_argument("--val_roi_y", default=96, type=int, help="roi size in y direction")
parser.add_argument("--val_roi_z", default=32, type=int, help="roi size in z direction")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--workers", default=1, type=int, help="number of workers")
parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")
parser.add_argument('--backbone', default='Foundation_model', choices=['Foundation_model', 'SwinUNETR', 'VIT3D'], help='backbone [Foundation_model or SwinUNETR or VIT3D]')
parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")

def rotate_tensor_3d(tensor, angle):
    """旋转 3D Tensor（Depth, Height, Width），保持通道顺序"""
    if angle == 90:
        return tensor.rot90(1, [2, 3])  # 沿 (H, W) 旋转 90°
    elif angle == 180:
        return tensor.rot90(2, [2, 3])  # 旋转 180°
    elif angle == 270:
        return tensor.rot90(3, [2, 3])  # 旋转 270°
    else:
        return tensor  # 原始方向

def inverse_rotate_tensor_3d(tensor, angle):
    """反向旋转，将预测结果恢复到原始方向"""
    if angle == 90:
        return tensor.rot90(3, [2, 3])  # 旋转 -90°
    elif angle == 180:
        return tensor.rot90(2, [2, 3])  # 旋转 -180°
    elif angle == 270:
        return tensor.rot90(1, [2, 3])  # 旋转 -270°
    else:
        return tensor  # 原始方向

def flip_tensor_3d(tensor, axis):
    """翻转 3D Tensor，axis=2（水平），axis=3（垂直），axis=1（深度）"""
    return torch.flip(tensor, dims=[axis])


def segment_with_augmentation(model, input_tensor,device,args):
    ####多种增强方式输入
    """
        使用 8 种数据增强方式进行分割，并融合结果
        """
    #  计算不同增强方式的预测
    outputs = []
    # 原始输入
    output = sliding_window_inference(
        input_tensor, (args.roi_x, args.roi_y, args.roi_z), 4, model, overlap=args.infer_overlap, mode="gaussian"
    )
    output = torch.softmax(output, 1).cpu().numpy()
    output = np.argmax(output, axis=1).astype(np.uint8)[0]
    outputs.append(output)
    # 旋转
    for angle in [45, 90, 180, 270]:
        rotated_input = rotate_tensor_3d(input_tensor, angle)
        output = sliding_window_inference(
            rotated_input, (args.roi_x, args.roi_y, args.roi_z), 4, model, overlap=args.infer_overlap, mode="gaussian"
        )
        output = inverse_rotate_tensor_3d(output, angle)
        output = torch.softmax(output, 1).cpu().numpy()
        output = np.argmax(output, axis=1).astype(np.uint8)[0]
        outputs.append(output)

    # 翻转
    for axis in [2, 3]:  # 水平、垂直、深度翻转
        flipped_input = flip_tensor_3d(input_tensor, axis)
        output = sliding_window_inference(
            flipped_input, (args.roi_x, args.roi_y, args.roi_z), 4, model, overlap=args.infer_overlap, mode="gaussian"
        )
        output = flip_tensor_3d(output, axis)  # 反向翻转恢复
        output = torch.softmax(output, 1).cpu().numpy()
        output = np.argmax(output, axis=1).astype(np.uint8)[0]
        outputs.append(output)
        # 采用融合策略（最大置信度融合）
    outputs = [torch.from_numpy(out).to(device) if isinstance(out, np.ndarray) else out.to(device) for out in
               outputs]
    fused_output = torch.maximum(outputs[0], outputs[1])
    for i in range(2, len(outputs)):
        fused_output = torch.maximum(fused_output, outputs[i])

    fused_output = fused_output.cpu().numpy()

    return fused_output


def main():
    args = parser.parse_args()
    args.test_mode = True
    output_directory = "./outputs/" + args.exp_name
    # output_directory = "/media/zxg/Nixi_plus/基督得胜硬盘备份/TAO unlabeled T1 T1c/Reg Crop TAO UL/t1c_seg_output"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    # val_loader = get_loader(args)
    t1_loader = get_loader(args,train_modality='T1')
    t1c_loader = get_loader(args,train_modality='T1')
    pretrained_dir = args.pretrained_dir
    # model_name = args.pretrained_model_name
    model_name = "model.pt"
    post_label = AsDiscrete(to_onehot=args.out_channels)
    post_pred = AsDiscrete(argmax=True, to_onehot=args.out_channels)
    dice_acc = DiceMetric(include_background=False, reduction=MetricReduction.MEAN, get_not_nans=True)
    assd_metric = SurfaceDistanceMetric(
        include_background=False,  # 通常设置为 False，只评估前景
        symmetric=True,
        reduction="mean"  # 对每个 batch 的类别结果做 mean
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_pth = os.path.join(pretrained_dir, model_name)
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
    model_dict = torch.load(pretrained_pth)["state_dict"]
    # model.load_state_dict(model_dict, strict=False)
    model.load_state_dict(model_dict)
    model.eval()
    model.to(device)

    with torch.no_grad():
        cnt = 0
        dice_list_case0 = []
        dice_list_case_0 = []
        dice_list_case_1 = []
        dice_list_case1 = []
        dice_list_case2 = []
        dice_list_case3 = []
        for i, (batch_t1, batch_t1c) in enumerate(zip(t1_loader, t1c_loader)):

            t1_data, t1_target = batch_t1c["image_m1"].cuda(), batch_t1c["label_m1"].cuda()
            # t1c_data, t1c_target = batch_t1c["image_t1c"].cuda(), batch_t1c["image_t1c"].cuda()
            img_name_ = batch_t1c["image_m1_meta_dict"]["filename_or_obj"][0].split("/")[-2]
            img_name = img_name_+"_"+batch_t1c["image_m1_meta_dict"]["filename_or_obj"][0].split("/")[-1]
            original_affine = batch_t1c["image_m1_meta_dict"]["affine"][0].numpy()
            ori_t1_data = t1_data[0, 0].detach().cpu().numpy()
            # ori_t1c_data = t1c_data[0, 0].detach().cpu().numpy()
            gt_t1_data = t1_target[0, 0].detach().cpu().numpy()
            _, _, h, w, d = t1_target.shape
            target_shape = (h, w, d)
            print("Inference on case {}".format(img_name))
            val_outputs_t1 = sliding_window_inference(
                t1_data, (args.roi_x, args.roi_y, args.roi_z), 4, model, overlap=args.infer_overlap, mode="gaussian"
            )
            val_t1_labels_list = decollate_batch(t1_target)
            # val_t1c_labels_list = decollate_batch(t1c_target)
            val_t1_labels_convert = [post_label(val_t1_label_tensor) for val_t1_label_tensor in val_t1_labels_list]
            # val_t1c_labels_convert = [post_label(val_t1c_label_tensor) for val_t1c_label_tensor in val_t1c_labels_list]
            val_t1_outputs_list = decollate_batch(val_outputs_t1)
            # val_t1c_outputs_list = decollate_batch(t1c_logits)
            val_t1_output_convert = [post_pred(val_t1_pred_tensor) for val_t1_pred_tensor in val_t1_outputs_list]
            # val_t1c_output_convert = [post_pred(val_t1c_pred_tensor) for val_t1c_pred_tensor in val_t1c_outputs_list]
            dice_acc.reset()
            assd_metric.reset()
            dice_acc(y_pred=val_t1_output_convert, y=val_t1_labels_convert)
            assd_metric(y_pred=val_t1_output_convert, y=val_t1_labels_convert)
            acc, not_nans = dice_acc.aggregate()
            assd = assd_metric.aggregate()
            acc = acc.cuda(0)
            assd = assd.cuda(0)
            print("Mean Organ dice_acc & assd: {}{}".format(acc,assd))

            ## for t1 data
            val_outputs_t1 = torch.softmax(val_outputs_t1, 1).cpu().numpy()
            val_outputs_t1 = np.argmax(val_outputs_t1, axis=1).astype(np.uint8)[0]

            # for t1c data
            # val_outputs_t1c = torch.softmax((val_outputs_t1c), 1).cpu().numpy()
            # val_outputs_t1c = np.argmax(val_outputs_t1c, axis=1).astype(np.uint8)[0]

            # 确保 mask1 和 mask2 都是 Tensor
            if isinstance(val_outputs_t1, np.ndarray):
                val_outputs_t1_ = torch.from_numpy(val_outputs_t1).to(device)

            # if isinstance(val_outputs_t1c, np.ndarray):
            #     val_outputs_t1c_ = torch.from_numpy(val_outputs_t1c).to(device)

            # 进行融合
            # fused_mask = torch.maximum(val_outputs_t1_, val_outputs_t1c_)
            # fused_mask = fused_mask.cpu().numpy()

            # 进行图像增强
            t1_seg_out = segment_with_augmentation(model, t1_data, device, args)
            # t1c_seg_out = segment_with_augmentation(model, t1c_data, device, args)
            # 确保 mask1 和 mask2 都是 Tensor

            if isinstance(t1_seg_out, np.ndarray):
                t1c_seg_out_ = torch.from_numpy(t1_seg_out).to(device)
            val_outputs_t1_ = torch.from_numpy(val_outputs_t1).to(device)
            t1_seg_out_ = torch.from_numpy(t1_seg_out).to(device)
            #
            val_labels = t1_target.cpu().numpy()[0, 0, :, :, :]
            val_labels_ = torch.from_numpy(val_labels).to(device)

            final_fused_mask = torch.maximum(t1_seg_out_, val_labels_)
            final_fused_mask = final_fused_mask.cpu().numpy()

            dice_list_sub0 = []
            for i in range(1, 10):
                organ_Dice0 = dice(val_outputs_t1 == i, val_labels == i)
                dice_list_sub0.append(organ_Dice0)
            mean_dice0 = np.mean(dice_list_sub0)
            print("Mean Organ Dice: {}".format(mean_dice0))

            if mean_dice0>0.4:
                file_name = img_name.split("_")[0]
                output_directory_ = os.path.join(output_directory,file_name)
                if not os.path.exists(output_directory_):
                    os.makedirs(output_directory_)
                # dice_list_case0.append(mean_dice0)

                # nib.save(
                #     nib.Nifti1Image(val_outputs_t1.astype(np.uint8), original_affine), os.path.join(output_directory_, img_name.replace("",""))
                # )
                # nib.save(
                #     nib.Nifti1Image(t1_seg_out.astype(np.uint8), original_affine),
                #     os.path.join(output_directory_, "aug_" + img_name.replace("", ""))
                # )
                # nib.save(
                #     nib.Nifti1Image(final_fused_mask.astype(np.uint8), original_affine),
                #     os.path.join(output_directory_, "fuse_" + img_name.replace("", ""))
                # )
                # nib.save(
                #     nib.Nifti1Image(val_outputs_t1c.astype(np.uint8), original_affine), os.path.join(output_directory_,img_name)
                # )
                # nib.save(
                #     nib.Nifti1Image(fused_mask.astype(np.uint8), original_affine), os.path.join(output_directory_, "fuse_"+img_name.replace("t1c_sequence",""))
                # )
            cnt = cnt + 1
        print("Overall Mean Dice: {}".format(np.mean(dice_list_case1)))


if __name__ == "__main__":
    main()
