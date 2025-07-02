import os
import shutil
import time
import ast

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from utils.utils import AverageMeter, distributed_all_gather
from utils.utils import dice, resample_3d, ORGAN_NAME
from sklearn.metrics import confusion_matrix, roc_auc_score, ConfusionMatrixDisplay
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn

def cls_score(pred, label,prob):
    """
    pred:  [[1, 0, 0, 0], [0, 0, 0, 1]]
    label: [[0, 0, 1, 1], [1, 1, 1, 1]]
    true: [[ 1.000  1.000  1.000  0.000  1.000  1.000  0.000], [ 1.000  0.000  0.000  1.000  0.000  0.000  0.000]]
    pred: [[ 1.000  0.000  1.000  0.000  0.000  0.000  1.000], [ 1.000  0.000  1.000  0.000  0.000  0.000  1.000]]
    """

    # case level
    case_pred = np.array([np.any(item) for item in pred], dtype=int)
    case_true = np.array([np.any(item) for item in label], dtype=int)
    # 强制指定类别
    labels = [0, 1]
    tn, fp, fn, tp = confusion_matrix(case_true, case_pred,labels=labels).ravel()

    case_acc = np.sum(case_pred == case_true) / (len(case_true) + 1e-9)
    case_sen = tp / (tp + fn + 1e-9)
    case_prec = tp / (tp + fp + 1e-9)
    case_f1 = (2 * case_prec * case_sen) / (case_prec + case_sen + 1e-9)
    # AUC for case-level
    # case_pred_proba = np.array([np.max(item) for item in prob])  # 使用最大概率作为预测值
    # case_auc = roc_auc_score(label, prob, multi_class='ovo')


    # organ level
    organ_pred = np.array(pred).ravel()
    organ_true = np.array(label).ravel()
    tn, fp, fn, tp = confusion_matrix(organ_true, organ_pred).ravel()

    organ_acc = np.sum(organ_pred == organ_true) / (len(organ_true) + 1e-9)
    organ_sen = tp / (tp + fn + 1e-9)
    organ_prec = tp / (tp + fp + 1e-9)
    organ_f1 = (2 * organ_prec * organ_sen) / (organ_prec + organ_sen + 1e-9)

    # AUC for organ-level
    organ_pred_proba = np.array(prob).ravel()  # 使用原始概率
    organ_auc = roc_auc_score(organ_true, organ_pred_proba, multi_class='ovo')

    score_table = {
        "case_acc": case_acc,
        "case_sensitive": case_sen,
        "case_precision": case_prec,
        "case_f1": case_f1,
        "organ_acc": organ_acc,
        "organ_sensitive": organ_sen,
        "organ_precision": organ_prec,
        "organ_f1": organ_f1,
        "organ_auc": organ_auc
    }
    score = 0.3 * organ_auc + 0.2 * organ_f1 + 0.2 * organ_prec + 0.3 * organ_acc

    return score, score_table

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


def get_TAO_embedding(labels):
    TAO_embedding = torch.load("./Text-emmbedding-gen/TAO_clip_txt_encoding.pth")
    # labels shape: B, 4
    B = labels.shape[0]
    batch_tao_features = []
    for b in range(B):
        embedding_TAO_activity = TAO_embedding[0][(labels[b, 0].int()), ...]
        embedding_Protosis = TAO_embedding[0][((labels[b, 1] + 2).int()), ...]
        embedding_Periorbital_fat_edema = TAO_embedding[0][((labels[b, 2] + 4).int()), ...]
        embedding_Eyelid_edema = TAO_embedding[0][((labels[b, 2] + 4).int()), ...]
        tao_feature = torch.cat((embedding_TAO_activity, embedding_Protosis, embedding_Periorbital_fat_edema, embedding_Eyelid_edema), axis=0)
        batch_tao_features.append(tao_feature)

    batch_tao_features = torch.stack(batch_tao_features)

    return batch_tao_features

def label_to_text_prompt(labels):
    path = "./Text-emmbedding-gen/TAO_clip_txt_encoding.pth"
    text_embedding = torch.load(path)
    prompt_template = [[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
 [1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0], [0, 1, 0, 1],
 [0, 0, 1, 1], [1, 1, 1, 0], [1, 1, 0, 1], [1, 0, 1, 1], [0, 1, 1, 1],
 [1, 1, 1, 1]]
    text_prompts = []
    for array in labels:
        index = prompt_template.index(array.tolist())
        text_prompts.append(text_embedding[0][index-1].cpu())

    text_prompts = np.stack(text_prompts, axis=0)
    text_prompts = torch.from_numpy(text_prompts)

    return text_prompts

def train_epoch(model, t1_loader, t1c_loader, optimizer, scaler, epoch, loss_func, focal_loss_func, args):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    save_log_dir = args.logdir
    for idx, (batch_ct, batch_mri) in enumerate(zip(t1_loader, t1c_loader)):
        if isinstance(batch_ct, list) and isinstance(batch_ct, list):
            t1_data, t1_target = batch_ct
            t1c_data, t1c_target = batch_mri
        else:
            t1_data, t1_target, label = batch_ct["img_t1"], batch_ct["mask_t1"],batch_ct["label"]
            unique_label = list({item.split('/')[-1] for item in label})
            parsed_labels = [ast.literal_eval(label) for label in unique_label]
            parsed_labels = [ast.literal_eval(item.split('/')[-1]) for item in label]

            labels = np.array(parsed_labels, dtype=float)
            labels = torch.FloatTensor(labels)
            t1c_data, t1c_target = batch_mri["img_t1c"], batch_mri["mask_t1c"]
        t1_data, t1_target, t1c_data, t1c_target, labels = t1_data.cuda(args.rank), t1_target.cuda(args.rank), t1c_data.cuda(args.rank), t1c_target.cuda(args.rank),labels.cuda(args.rank)

        with autocast(enabled=args.amp):
            text_prompts = label_to_text_prompt(labels).cuda()
            TAO_embedding = get_TAO_embedding(labels).cuda()
            outputs, vl_align_feature = model(t1_data, t1c_data)
            ce_loss = loss_func(outputs, labels)
            consistency_weight = cosine_rampdown(epoch, args.max_epochs)

            if args.hybrid_loss == "hybrid_loss":
                focal_loss = focal_loss_func(outputs, labels)
                sup_loss = 0.4 * ce_loss + 0.6 * focal_loss
            elif args.hybrid_loss == "focal_loss":
                focal_loss = focal_loss_func(outputs, labels)
                sup_loss = focal_loss
            else:
                sup_loss = ce_loss

            if args.text_prompt_loss:
                kl_input = F.log_softmax(vl_align_feature, dim=1)
                kl_target = F.log_softmax(text_prompts, dim=1)
                kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
                KL_prompt_loss = kl_loss(kl_input, kl_target)
            else:
                KL_prompt_loss = 0

            loss = consistency_weight * sup_loss + (1 - consistency_weight) * KL_prompt_loss

        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        else:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
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
                "ce_loss: {:.4f}".format(ce_loss),
                "KL_prompt_loss: {:.4f}".format(KL_prompt_loss),
                "time {:.2f}s".format(time.time() - start_time),
            )
            with open(os.path.join(save_log_dir, 'log.txt'), 'a') as f:
                print(
                    "Epoch {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(t1_loader)),
                    "loss: {:.4f}".format(run_loss.avg),
                    "ce_loss: {:.4f}".format(ce_loss),
                    "KL_prompt_loss: {:.4f}".format(KL_prompt_loss),
                    "time {:.2f}s".format(time.time() - start_time),
                )
        start_time = time.time()
    return run_loss.avg


def val_epoch(model, t1_loader, t1c_loader, epoch, args):
    model.eval()
    y_true = []
    y_pred = []
    y_pred_prob = []
    eval_time = time.time()
    save_log_dir = args.logdir
    start_time = time.time()
    with torch.no_grad():
        for idx, (batch_ct, batch_mri) in enumerate(zip(t1_loader,t1c_loader)):
            if isinstance(batch_ct, list) and isinstance(batch_mri, list):
                t1_data, t1_target = batch_ct
                t1c_data, t1c_target = batch_mri
            else:
                t1_data, t1_target, label = batch_ct["img_t1"], batch_ct["mask_t1"],batch_ct["label"]
                unique_label = list({item.split('/')[-1] for item in label})
                parsed_labels = [ast.literal_eval(label) for label in unique_label]
                labels = np.array(parsed_labels, dtype=float)
                labels = torch.FloatTensor(labels)
                t1c_data, t1c_target = batch_mri["img_t1c"], batch_mri["mask_t1c"]
            t1_data, t1_target, t1c_data, t1c_target, labels = t1_data.cuda(args.rank), t1_target.cuda(
                args.rank), t1c_data.cuda(args.rank), t1c_target.cuda(args.rank), labels.cuda(args.rank)
            _, _, h, w, d = t1_target.shape
            with autocast(enabled=args.amp):
                val_preds, _ = model(t1_data, t1c_data)

            img_name = batch_ct["img_t1_meta_dict"]["filename_or_obj"][0].split("/")[-1]
            y_pred_prob.extend(val_preds.cpu().numpy())
            val_preds[val_preds >= 0] = 1
            val_preds[val_preds < 0] = 0

            val_labels = labels.cpu().numpy()
            val_preds = val_preds.detach().cpu().numpy()
            y_true.extend(val_labels)
            y_pred.extend(val_preds)
            print("names", img_name, "val labels:", val_labels, "preds:", val_preds,
                  'time {:.2f}s'.format(time.time() - start_time))
            with open(os.path.join(save_log_dir, 'log.txt'), 'a') as f:
                print("names", img_name, "val labels:", val_labels, "preds:", val_preds,
                      'time {:.2f}s'.format(time.time() - start_time), file=f)
            start_time = time.time()
    score, score_table = cls_score(y_pred, y_true,y_pred_prob)
    print('Final validation  {}/{}'.format(epoch, args.max_epochs - 1),"organ_acc", 'Score', score,score_table['organ_acc'], "organ_auc", score_table['organ_auc'],"Organ F1",
          score_table['organ_f1'], 'score_table', score_table,'time {:.2f}s'.format(time.time() - eval_time))
    with open(os.path.join(save_log_dir, 'log.txt'), 'a') as f:
        print('Final validation  {}/{}'.format(epoch, args.max_epochs - 1),"organ_acc", 'Score', score,score_table['organ_acc'], "organ_auc", score_table['organ_auc'],"Organ F1",
          score_table['organ_f1'], 'score_table', score_table,'time {:.2f}s'.format(time.time() - eval_time), file=f)
    return score

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
    focal_loss_func,
    args,
    scheduler=None,
    start_epoch=0,
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
            t1c_train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()
        print(args.rank, time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()
        train_loss = train_epoch(
            model, t1_train_loader,t1c_train_loader, optimizer, scaler=scaler, epoch=epoch, loss_func=loss_func, focal_loss_func=focal_loss_func, args=args
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
        if (epoch + 1) % args.val_every == 0 or (epoch + 1)== args.max_epochs:
            if args.distributed:
                torch.distributed.barrier()
            epoch_time = time.time()
            val_avg_acc = val_epoch(
                model,
                t1_val_loader,
                t1c_val_loader,
                epoch=epoch,
                args=args,
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
