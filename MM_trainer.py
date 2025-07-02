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

import os
import shutil
import time

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from utils.utils import AverageMeter, distributed_all_gather

from monai.data import decollate_batch

def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length

def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))
def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))


def load(model, model_dict):
    if "state_dict" in model_dict.keys():
        state_dict = model_dict["state_dict"]
    elif "network_weights" in model_dict.keys():
        state_dict = model_dict["network_weights"]
    elif "net" in model_dict.keys():
        state_dict = model_dict["net"]
    else:
        state_dict = model_dict

    if "module." in list(state_dict.keys())[0]:
        # print("Tag 'module.' found in state dict - fixing!")
        for key in list(state_dict.keys()):
            state_dict[key.replace("module.", "")] = state_dict.pop(key)

    if "backbone." in list(state_dict.keys())[0]:
        # print("Tag 'backbone.' found in state dict - fixing!")
        for key in list(state_dict.keys()):
            state_dict[key.replace("backbone.", "")] = state_dict.pop(key)

    if "swin_vit" in list(state_dict.keys())[0]:
        # print("Tag 'swin_vit' found in state dict - fixing!")
        for key in list(state_dict.keys()):
            state_dict[key.replace("swin_vit", "swinViT")] = state_dict.pop(key)

    current_model_dict = model.state_dict()
    new_state_dict = {
        k: state_dict[k] if (k in state_dict.keys()) and (state_dict[k].size() == current_model_dict[k].size()) else current_model_dict[k]
        for k in current_model_dict.keys()}

    model.load_state_dict(new_state_dict, strict=True)

    return model

def train_epoch(model, t1_loader, t1c_loader, optimizer, scaler, epoch, loss_func, args):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    save_log_dir = args.logdir
    for idx, (batch_t1, batch_t1c) in enumerate(zip(t1_loader,t1c_loader)):
        if isinstance(batch_t1, list):
            t1_data, t1_target = batch_t1
        else:
            t1_data, t1_target = batch_t1["image_m1"], batch_t1["label_m1"]
            t1c_data, t1c_target = batch_t1c["image_m2"], batch_t1c["label_m2"]
        t1_data, t1_target = t1_data.cuda(args.rank), t1_target.cuda(args.rank)
        t1c_data, t1c_target = t1c_data.cuda(args.rank), t1c_target.cuda(args.rank)
        for param in model.parameters():
            param.grad = None
        with autocast(enabled=args.amp):
            t1_logit = model(t1_data)
            t1c_logit = model(t1c_data)
            t1_loss = loss_func(t1_logit, t1_target)
            t1c_loss = loss_func(t1c_logit, t1c_target)
            loss = (t1_loss + t1c_loss)/2
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if args.distributed:
            loss_list = distributed_all_gather([loss], out_numpy=True, is_valid=idx < t1_loader.sampler.valid_length)
            run_loss.update(
                np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0), n=args.batch_size * args.world_size
            )
        else:
            run_loss.update(loss.item(), n=args.batch_size)
        if args.rank == 0:
            print(
                "Epoch {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(t1_loader)),
                "loss: {:.4f}".format(run_loss.avg),
                "t1_loss: {:.4f}".format(t1_loss),
                "t1c_loss: {:.4f}".format(t1c_loss),
                "time {:.2f}s".format(time.time() - start_time),
            )
            with open(os.path.join(save_log_dir, 'log.txt'), 'a') as f:
                print(
                    "Epoch {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(t1_loader)),
                    "loss: {:.4f}".format(run_loss.avg),
                    "t1_loss: {:.4f}".format(t1_loss),
                    "t1c_loss: {:.4f}".format(t1c_loss),
                    "time {:.2f}s".format(time.time() - start_time),
                )
        start_time = time.time()
    for param in model.parameters():
        param.grad = None
    return run_loss.avg


def val_epoch(model, t1_loader, epoch, acc_func, args, model_inferer=None, post_label=None, post_pred=None):
    model.eval()
    run_acc = AverageMeter()
    start_time = time.time()
    with torch.no_grad():
        for idx, (batch_t1) in enumerate(zip(t1_loader)):
            if isinstance(batch_t1, list) and isinstance(batch_t1, list):
                t1_data, t1_target = batch_t1
            else:
                t1_data, t1_target = batch_t1[0]["image_m1"], batch_t1[0]["label_m1"]
            t1_data, t1_target = t1_data.cuda(args.rank), t1_target.cuda(
                args.rank)
            with autocast(enabled=args.amp):
                if model_inferer is not None:
                    t1_logit = model_inferer(t1_data)
                    # _, _, t1_logit = torch.chunk(t1_logits, 3, dim=1)  # 按通道拆分
                else:
                    t1_logit= model(t1_data)
                    # _, _, t1_logit = torch.chunk(t1_logits, 3, dim=1)  # 按通道拆分
            if not t1_logit.is_cuda:
                t1_target = t1_target.cpu()
            val_t1_labels_list = decollate_batch(t1_target)
            # val_t1c_labels_list = decollate_batch(t1c_target)
            val_t1_labels_convert = [post_label(val_t1_label_tensor) for val_t1_label_tensor in val_t1_labels_list]
            # val_t1c_labels_convert = [post_label(val_t1c_label_tensor) for val_t1c_label_tensor in val_t1c_labels_list]
            val_t1_outputs_list = decollate_batch(t1_logit)
            # val_t1c_outputs_list = decollate_batch(t1c_logits)
            val_t1_output_convert = [post_pred(val_t1_pred_tensor) for val_t1_pred_tensor in val_t1_outputs_list]
            # val_t1c_output_convert = [post_pred(val_t1c_pred_tensor) for val_t1c_pred_tensor in val_t1c_outputs_list]
            acc_func.reset()
            acc_func(y_pred=val_t1_output_convert, y=val_t1_labels_convert)
            # acc_func(y_pred=val_t1c_output_convert, y=val_t1c_labels_convert)
            acc, not_nans = acc_func.aggregate()
            acc = acc.cuda(args.rank)

            if args.distributed:
                acc_list, not_nans_list = distributed_all_gather(
                    [acc, not_nans], out_numpy=True, is_valid=idx < t1_loader.sampler.valid_length
                )
                for al, nl in zip(acc_list, not_nans_list):
                    run_acc.update(al, n=nl)

            else:
                run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())

            if args.rank == 0:
                avg_acc = np.mean(run_acc.avg)
                print(
                    "Val {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(t1_loader)),
                    "acc",
                    avg_acc,
                    "time {:.2f}s".format(time.time() - start_time),
                )
            start_time = time.time()
    return run_acc.avg

def save_checkpoint(model, epoch, args, filename="model.pt", best_acc=0, optimizer=None, scheduler=None):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    filename = os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)


def run_training(
    model,
    t1_train_loader,
    t1_val_loader,
    t1c_train_loader,
    t1c_val_loader,
    optimizer,
    loss_func,
    acc_func,
    args,
    model_inferer=None,
    scheduler=None,
    start_epoch=0,
    post_label=None,
    post_pred=None,
):
    writer = None
    if args.logdir is not None and args.rank == 0:
        writer = SummaryWriter(log_dir=args.logdir)
        if args.rank == 0:
            print("Writing Tensorboard logs to ", args.logdir)
    scaler = None
    if args.amp:
        scaler = GradScaler()
    val_acc_max = 0.0
    save_log_dir = args.logdir
    for epoch in range(start_epoch, args.max_epochs):
        if args.distributed:
            t1_train_loader.sampler.set_epoch(epoch)
            # t1c_train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()
        print(args.rank, time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()
        train_loss = train_epoch(
            model, t1_train_loader, t1c_train_loader, optimizer, scaler=scaler, epoch=epoch, loss_func=loss_func, args=args
        )
        if args.rank == 0:
            print(
                "Final training  {}/{}".format(epoch, args.max_epochs - 1),
                "loss: {:.4f}".format(train_loss),
                "time {:.2f}s".format(time.time() - epoch_time),
            )
            with open(os.path.join(save_log_dir, 'log.txt'), 'a') as f:
                print(
                    "Final training  {}/{}".format(epoch, args.max_epochs - 1),
                    "loss: {:.4f}".format(train_loss),
                    "time {:.2f}s".format(time.time() - epoch_time),file=f
                )
        if args.rank == 0 and writer is not None:
            writer.add_scalar("train_loss", train_loss, epoch)
        b_new_best = False
        if (epoch + 1) % args.val_every == 0:
            if args.distributed:
                torch.distributed.barrier()
            epoch_time = time.time()
            val_avg_acc = val_epoch(
                model,
                t1_val_loader,
                epoch=epoch,
                acc_func=acc_func,
                model_inferer=model_inferer,
                args=args,
                post_label=post_label,
                post_pred=post_pred,
            )

            val_avg_acc = np.mean(val_avg_acc)

            if args.rank == 0:
                print(
                    "Final validation  {}/{}".format(epoch, args.max_epochs - 1),
                    "acc",
                    val_avg_acc,
                    "time {:.2f}s".format(time.time() - epoch_time),
                )
                with open(os.path.join(save_log_dir, 'log.txt'), 'a') as f:
                    print(
                        "Final validation  {}/{}".format(epoch, args.max_epochs - 1),
                        "acc",
                        val_avg_acc,
                        "time {:.2f}s".format(time.time() - epoch_time),file=f
                    )
                if writer is not None:
                    writer.add_scalar("val_acc", val_avg_acc, epoch)
                if val_avg_acc > val_acc_max:
                    print("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, val_avg_acc))
                    with open(os.path.join(save_log_dir, 'log.txt'), 'a') as f:
                        print("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, val_avg_acc),file=f)
                    val_acc_max = val_avg_acc
                    b_new_best = True
                    if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                        save_checkpoint(
                            model, epoch, args, best_acc=val_acc_max, optimizer=optimizer, scheduler=scheduler
                        )
            if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                save_checkpoint(model, epoch, args, best_acc=val_acc_max, filename="model_final.pt")
                if b_new_best:
                    print("Copying to model.pt new best model!!!!")
                    with open(os.path.join(save_log_dir, 'log.txt'), 'a') as f:
                        print("Copying to model.pt new best model!!!!",file=f)
                    shutil.copyfile(os.path.join(args.logdir, "model_final.pt"), os.path.join(args.logdir, "model.pt"))
        if scheduler is not None:
            scheduler.step()
    print("Training Finished !, Best Accuracy: ", val_acc_max)
    with open(os.path.join(save_log_dir, 'log.txt'), 'a') as f:
        print("Training Finished !, Best Accuracy: ", val_acc_max,file=f)

    return val_acc_max
