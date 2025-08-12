import os
import sys
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from tensorboardX import SummaryWriter

from dataloaders.dataloader import TextDataLoader
from models.losses import *
from models.image_models import *
from models.model import *
from utils.utils import post_process_depth, flip_lr, compute_errors

os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg


parser = argparse.ArgumentParser(description='TRIDE PyTorch implementation.', fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args

# ======= Core =======
parser.add_argument('--mode',                                       type=str, default='train', help='train or test')
parser.add_argument('--model_name',                                 type=str, default='TRIDE')
parser.add_argument('--main_path',                                  type=str, required=True)

# ======= Data paths =======
parser.add_argument('--train_text_feature_general_path',            type=str, required=True)
parser.add_argument('--train_text_feature_left_path',               type=str, required=True)
parser.add_argument('--train_text_feature_mid_left_path',           type=str, required=True)
parser.add_argument('--train_text_feature_mid_right_path',          type=str, required=True)
parser.add_argument('--train_text_feature_right_path',              type=str, required=True)
parser.add_argument('--train_radar_path',                           type=str, required=True)
parser.add_argument('--train_weather_condition_path',               type=str, required=True)
parser.add_argument('--train_image_path',                           type=str, required=True)
parser.add_argument('--train_ground_truth_path',                    type=str, required=True)
parser.add_argument('--train_lidar_path',                           type=str, required=True)

parser.add_argument('--validation_image_path',                      type=str, required=True)
parser.add_argument('--validation_text_feature_general_path',       type=str, required=True)
parser.add_argument('--validation_text_feature_left_path',          type=str, required=True)
parser.add_argument('--validation_text_feature_mid_left_path',      type=str, required=True)
parser.add_argument('--validation_text_feature_mid_right_path',     type=str, required=True)
parser.add_argument('--validation_text_feature_right_path',         type=str, required=True)
parser.add_argument('--validation_weather_condition_path',          type=str, required=True)
parser.add_argument('--validation_ground_truth_path',               type=str, required=True)
parser.add_argument('--validation_radar_path',                      type=str, required=True)

# ======= Model =======
parser.add_argument('--k',                                          type=int, default=4)
parser.add_argument('--encoder_radar',                              type=str, default='resnet18')
parser.add_argument('--radar_input_channels',                       type=int, default=4)
parser.add_argument('--radar_gcn_channel_in',                       type=int, default=6)
parser.add_argument('--encoder',                                    type=str, default='resnet34_bts')
parser.add_argument('--n_filters_decoder',                          type=int, nargs='+', default=[256, 256, 128, 64, 32])
parser.add_argument('--input_height',                               type=int, default=352)
parser.add_argument('--input_width',                                type=int, default=704)
parser.add_argument('--max_depth',                                  type=float, default=100)
parser.add_argument('--fuse',                                       type=str, default='wafb', help='fusion of radar & image')
parser.add_argument('--text_fuse',                                  type=str, default='cross_attention')
parser.add_argument('--text_hidden_dim',                            type=int, default=128)
parser.add_argument('--use_img_feat',                               action='store_true')
parser.add_argument('--point_hidden_dim',                           type=int, default=128)

# ======= Logging / Checkpoints =======
parser.add_argument('--log_directory',                              type=str, default='')
parser.add_argument('--checkpoint_path',                            type=str, default='')
parser.add_argument('--log_freq',                                   type=int, default=100)
parser.add_argument('--save_freq',                                  type=int, default=500)

# ======= Train =======
parser.add_argument('--weight_decay',                               type=float, default=1e-2)
parser.add_argument('--retrain',                                    action='store_true')
parser.add_argument('--adam_eps',                                   type=float, default=1e-6)
parser.add_argument('--reg_loss',                                   type=str, default='l1', choices=['l1','l2','silog'])
parser.add_argument('--w_smoothness',                               type=float, default=0.00)
parser.add_argument('--variance_focus',                             type=float, default=0.85)
parser.add_argument('--batch_size',                                 type=int, default=4)
parser.add_argument('--num_epochs',                                 type=int, default=100)
parser.add_argument('--learning_rate',                              type=float, default=1e-4)
parser.add_argument('--end_learning_rate',                          type=float, default=-1)
parser.add_argument('--num_threads',                                type=int, default=1)

# ======= Eval =======
parser.add_argument('--do_online_eval',                             action='store_true')
parser.add_argument('--min_depth_eval',                             type=float, default=1e-3)
parser.add_argument('--max_depth_eval',                             type=float, default=80)
parser.add_argument('--eval_freq',                                  type=int, default=500)
parser.add_argument('--eval_summary_directory',                     type=str, default='')

# ======= Distributed =======
parser.add_argument('--distributed',                                action='store_true', help='use DistributedDataParallel')
parser.add_argument('--dist_url',                                   default='env://', type=str, help='url to set up distributed training')
parser.add_argument('--local_rank',                                 default=-1, type=int, help='local rank passed by torchrun')
parser.add_argument('--seed',                                       default=42, type=int)

eval_metrics = ['silog', 'abs_rel', 'log10', 'rms', 'sq_rel', 'log_rms', 'mae', 'd1', 'd2', 'd3']


# ------------------ Distributed helpers ------------------
def init_distributed_mode(args):
    args.distributed = args.distributed or ("RANK" in os.environ and "WORLD_SIZE" in os.environ)
    if not args.distributed:
        args.rank = 0
        args.world_size = 1
        args.local_rank = -1 if args.local_rank is None else args.local_rank
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.local_rank == -1:
        # torchrun sets LOCAL_RANK et al.
        args.local_rank = int(os.environ["LOCAL_RANK"])
    args.rank = int(os.environ["RANK"])
    args.world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl', init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
    dist.barrier()
    return torch.device('cuda', args.local_rank)


def is_main_process(args):
    return (not args.distributed) or args.rank == 0


def cleanup_distributed(args):
    if args.distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)





# ------------------ Eval ------------------
def online_eval(model, dataloader_eval, device_index, args, post_process=False):
    model.eval()
    correct = total = 0
    correct_rain = total_rain = 0
    correct_night = total_night = 0

    eval_measures = torch.zeros(11, device=device_index if device_index is not None else 'cpu')

    for _, eval_sample_batched in enumerate(tqdm(dataloader_eval, disable=not is_main_process(args))):
        with torch.no_grad():
            image                       = eval_sample_batched['image'].cuda(device_index, non_blocking=True)
            text_mask                   = eval_sample_batched['text_mask'].cuda(device_index, non_blocking=True)
            text_feature_general        = eval_sample_batched['text_feature_general'].cuda(device_index, non_blocking=True)
            text_feature_left           = eval_sample_batched['text_feature_left'].cuda(device_index, non_blocking=True)
            text_feature_mid_left       = eval_sample_batched['text_feature_mid_left'].cuda(device_index, non_blocking=True)
            text_feature_mid_right      = eval_sample_batched['text_feature_mid_right'].cuda(device_index, non_blocking=True)
            text_feature_right          = eval_sample_batched['text_feature_right'].cuda(device_index, non_blocking=True)
            text_length                 = eval_sample_batched['text_length'].cuda(device_index, non_blocking=True)

            radar_channels              = eval_sample_batched['radar_channels'].cuda(device_index, non_blocking=True)
            radar_points                = eval_sample_batched['radar_points'].cuda(device_index, non_blocking=True)

            label                       = eval_sample_batched['label'].cuda(device_index, non_blocking=True)
            gt_depth                    = eval_sample_batched['depth']

            pred_depth, class_pred      = model(
                image, radar_channels, radar_points, text_feature_general, text_feature_left,
                text_feature_mid_left, text_feature_mid_right, text_feature_right, text_mask, text_length
            )

            if post_process:
                image_flipped = flip_lr(image)
                radar_flipped = flip_lr(radar_channels) if radar_channels is not None else None
                text_mask_flipped = flip_lr(text_mask)

                pred_depth_flipped, _ = model(
                    image_flipped, radar_flipped, radar_points, text_feature_general, text_feature_left,
                    text_feature_mid_left, text_feature_mid_right, text_feature_right, text_mask_flipped, text_length
                )
                pred_depth = post_process_depth(pred_depth, pred_depth_flipped)

            # compute classification acc
            _, predicted = class_pred.max(1)

            if label.dim() == 2 and label.size(1) > 1:
                _, targets = label.max(1)
            else:
                targets = label.view(-1).long()

            total += label.size(0)
            correct += predicted.eq(targets).sum().item()

            total_rain += (targets == 2).sum().item()
            correct_rain += ((predicted == 2) & (targets == 2)).sum().item()

            total_night += (targets == 1).sum().item()
            correct_night += ((predicted == 1) & (targets == 1)).sum().item()

            pred_depth = pred_depth.detach().cpu().numpy().squeeze()
            gt_depth = gt_depth.cpu().numpy().squeeze()

        pred_depth[pred_depth < args.min_depth_eval] = args.min_depth_eval
        pred_depth[pred_depth > args.max_depth_eval] = args.max_depth_eval
        pred_depth[np.isinf(pred_depth)] = args.max_depth_eval
        pred_depth[np.isnan(pred_depth)] = args.min_depth_eval

        valid_mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)
        measures = compute_errors(gt_depth[valid_mask], pred_depth[valid_mask])

        eval_measures[:-1] += torch.tensor(measures, device=eval_measures.device)
        eval_measures[-1] += 1

    # gather from all ranks if distributed
    if args.distributed:
        dist.all_reduce(eval_measures, op=dist.ReduceOp.SUM)

    cnt = eval_measures[-1].item()
    if cnt > 0:
        eval_measures /= cnt

    if is_main_process(args) and total != 0:
        print('--------------Classification--------------')
        acc = 100.0 * correct / total
        acc_rain = 100.0 * (correct_rain / max(total_rain, 1))
        acc_night = 100.0 * (correct_night / max(total_night, 1))
        print('{:>7}, {:>13}, {:>14}'.format('Accuracy', 'Accuracy Rain', 'Accuracy Night'))
        print('{:.3f}%, {:.3f}%, {:.3f}%'.format(acc, acc_rain, acc_night))

        print('-------------Depth Estimation-------------')
        print('Computing errors for {} eval samples'.format(int(cnt)))
        print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format(
            'silog', 'abs_rel', 'log10', 'rms', 'sq_rel', 'log_rms', 'mae', 'd1', 'd2', 'd3'))
        for i in range(9):
            print('{:7.3f}, '.format(eval_measures[i].item()), end='')
        print('{:7.3f}'.format(eval_measures[9].item()))

    model.train()
    return eval_measures  # (10 + counter)


def main_worker(args):
    # ---------- Setup distributed/device ----------
    device = init_distributed_mode(args)
    set_seed(args.seed)
    cudnn.benchmark = True

    # ---------- Datasets & DataLoaders ----------
    td_train = TextDataLoader(args, 'train')
    td_val   = TextDataLoader(args, 'test')

    train_loader = td_train.data
    val_loader_rank0 = td_val.data if is_main_process(args) else None

    # ---------- Model ----------
    model = TRIDE(args)
    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print("Total number of parameters: {}".format(num_params))
    model.decoder.apply(weights_init_xavier)
    model.to(device)

    if args.distributed:
        # model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=False)
        model = DDP(
        model,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        find_unused_parameters=True,
        broadcast_buffers=False   
        # static_graph=False       
    )
    elif torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    base_model = model.module if hasattr(model, 'module') else model

    # ---------- Optimizer ----------
    optimizer = torch.optim.AdamW(
        [
            {'params': base_model.image_encoder.parameters(), 'weight_decay': args.weight_decay},
            {'params': base_model.radar_encoder.parameters(), 'weight_decay': args.weight_decay},
            {'params': base_model.text_encoder.parameters(), 'weight_decay': args.weight_decay},
            {'params': base_model.decoder.parameters(), 'weight_decay': 0.0},
        ],
        lr=args.learning_rate,
        eps=args.adam_eps
    )

    # ---------- Checkpoint load ----------
    global_step = 0
    best_eval_measures_lower_better = torch.zeros(7).cpu() + 1e3
    best_eval_measures_higher_better = torch.zeros(3).cpu()
    best_eval_steps = np.zeros(10, dtype=np.int32)

    model_just_loaded = False
    if args.checkpoint_path:
        if os.path.isfile(args.checkpoint_path):
            if is_main_process(args):
                print(f"Loading checkpoint '{args.checkpoint_path}'")
            map_location = {'cuda:%d' % 0: 'cuda:%d' % args.local_rank} if args.distributed else None
            checkpoint = torch.load(args.checkpoint_path, map_location=map_location)

            # load state dict robustly (DDP/DP/single)
            state_dict = checkpoint['model']
            try:
                model.load_state_dict(state_dict, strict=True)
            except RuntimeError:
                # strip possible "module." prefixes
                from collections import OrderedDict
                new_state = OrderedDict()
                for k, v in state_dict.items():
                    new_k = k.replace('module.', '') if k.startswith('module.') else k
                    new_state[new_k] = v
                base_model.load_state_dict(new_state, strict=True)

            if 'global_step' in checkpoint:
                global_step = checkpoint['global_step']
            try:
                best_eval_measures_higher_better = checkpoint['best_eval_measures_higher_better'].cpu()
                best_eval_measures_lower_better = checkpoint['best_eval_measures_lower_better'].cpu()
                best_eval_steps = checkpoint['best_eval_steps']
            except KeyError:
                if is_main_process(args):
                    print("Could not load values for online evaluation")

            if is_main_process(args):
                print(f"Loaded checkpoint '{args.checkpoint_path}' (global_step {global_step})")
        else:
            if is_main_process(args):
                print(f"No checkpoint found at '{args.checkpoint_path}'")
        model_just_loaded = True

    if args.retrain:
        global_step = 0

    # ---------- Logging ----------
    if is_main_process(args):
        writer = SummaryWriter(os.path.join(args.log_directory, args.model_name, 'summaries'), flush_secs=30)
        eval_summary_writer = None
        if args.do_online_eval:
            eval_summary_path = os.path.join(args.eval_summary_directory or os.path.join(args.log_directory, 'eval'),
                                             args.model_name)
            eval_summary_writer = SummaryWriter(eval_summary_path, flush_secs=30)
    else:
        writer = None
        eval_summary_writer = None

    # ---------- Losses ----------
    if args.reg_loss == 'silog':
        loss_depth = silog_loss(variance_focus=args.variance_focus)
    elif args.reg_loss == 'l2':
        loss_depth = l2_loss()
    else:
        loss_depth = l1_loss()
    loss_classification = nn.CrossEntropyLoss()
    loss_smoothness = smoothness_loss_func()

    # ---------- LR schedule params ----------
    end_learning_rate = args.end_learning_rate if args.end_learning_rate != -1 else 0.1 * args.learning_rate

    # ---------- Make run folder on rank-0 ----------
    if is_main_process(args):
        os.makedirs(os.path.join(args.log_directory, args.model_name), exist_ok=True)

    # ---------- Train Loop ----------
    steps_per_epoch = len(train_loader)
    num_total_steps = max(1, args.num_epochs * steps_per_epoch)
    start_epoch = global_step // max(steps_per_epoch, 1)

    for epoch in range(start_epoch, args.num_epochs + 1):
        if args.distributed:
            td_train.set_epoch(epoch)

        loop = tqdm(train_loader, unit='batch', disable=not is_main_process(args))
        loop.set_description(f"Epoch {epoch}")

        for sample_batched in loop:
            optimizer.zero_grad(set_to_none=True)

            # ---- Move to device
            image                    = sample_batched['image'].to(device, non_blocking=True)
            text_feature_general     = sample_batched['text_feature_general'].to(device, non_blocking=True)
            text_feature_left        = sample_batched['text_feature_left'].to(device, non_blocking=True)
            text_feature_mid_left    = sample_batched['text_feature_mid_left'].to(device, non_blocking=True)
            text_feature_mid_right   = sample_batched['text_feature_mid_right'].to(device, non_blocking=True)
            text_feature_right       = sample_batched['text_feature_right'].to(device, non_blocking=True)
            text_length              = sample_batched['text_length'].to(device, non_blocking=True)
            text_mask                = sample_batched['text_mask'].to(device, non_blocking=True)
            depth_gt                 = sample_batched['depth'].to(device, non_blocking=True)
            single_depth_gt          = sample_batched['lidar'].to(device, non_blocking=True)
            label                    = sample_batched['label'].to(device, non_blocking=True)
            radar_channels           = sample_batched['radar_channels'].to(device, non_blocking=True)
            radar_points             = sample_batched['radar_points'].to(device, non_blocking=True)

            # ---- Forward
            depth_est, class_pred    = model(
                                    image, radar_channels, radar_points, text_feature_general, text_feature_left,
                                    text_feature_mid_left, text_feature_mid_right, text_feature_right, text_mask, text_length
                                    )

            # ---- Classification loss (handle one-hot or indices)
            if label.dim() == 2 and label.size(1) > 1:
                targets = label.argmax(dim=1)
            else:
                targets = label.view(-1).long()
            loss_c = loss_classification(class_pred, targets)

            # ---- Depth loss
            mask_single = single_depth_gt > 0.01
            mask = torch.logical_and(depth_gt > 0.01, mask_single == 0)
            loss_d = loss_depth(depth_est, depth_gt, mask.bool()) + \
                     loss_depth(depth_est, single_depth_gt, mask_single.bool())

            # ---- Smoothness (optional)
            if args.w_smoothness > 0.00:
                loss_s = loss_smoothness(depth_est, image) * args.w_smoothness
            else:
                loss_s = torch.tensor(0.0, device=device)

            loss = loss_d + loss_s + loss_c
            loss.backward()

            optimizer.step()

            # ---- LR decay (per step, consistent across ranks)
            current_lr = (args.learning_rate - end_learning_rate) * (1 - global_step / num_total_steps) ** 0.9 + end_learning_rate
            for pg in optimizer.param_groups:
                pg['lr'] = current_lr

            # ---- Save checkpoints (rank-0 only)
            if is_main_process(args) and (not args.do_online_eval) and global_step and (global_step % args.save_freq == 0):
                checkpoint = {
                    'global_step': global_step,
                    'model': (model.module.state_dict() if hasattr(model, 'module') else model.state_dict()),
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(checkpoint, os.path.join(args.log_directory, args.model_name, f'model-{global_step}'))

            # ---- Online eval (rank-0 only)
            if args.do_online_eval and global_step and (global_step % args.eval_freq == 0) and (not model_just_loaded):
                if args.distributed:
                    dist.barrier()
                if is_main_process(args):
                    eval_measures = online_eval(model, val_loader_rank0, args.local_rank if args.distributed else (torch.cuda.current_device() if torch.cuda.is_available() else None), args, post_process=True)
                    if eval_measures is not None and eval_summary_writer is not None:
                        for i in range(10):
                            eval_summary_writer.add_scalar(eval_metrics[i], eval_measures[i].cpu(), int(global_step))
                            measure = eval_measures[i].item()
                            is_best = False
                            if i < 7:
                                if measure < best_eval_measures_lower_better[i]:
                                    old_best = best_eval_measures_lower_better[i].item()
                                    best_eval_measures_lower_better[i] = measure
                                    is_best = True
                            else:
                                idx = i - 7
                                if measure > best_eval_measures_higher_better[idx]:
                                    old_best = best_eval_measures_higher_better[idx].item()
                                    best_eval_measures_higher_better[idx] = measure
                                    is_best = True
                            if is_best:
                                old_best_step = best_eval_steps[i]
                                old_best_name = f"/model-{old_best_step}-best_{eval_metrics[i]}_{old_best:.5f}"
                                old_path = os.path.join(args.log_directory, args.model_name + old_best_name)
                                if os.path.exists(old_path):
                                    os.remove(old_path)
                                best_eval_steps[i] = global_step
                                model_save_name = f"/model-{global_step}-best_{eval_metrics[i]}_{measure:.5f}"
                                print(f'New best for {eval_metrics[i]}. Saving model: {model_save_name}')
                                checkpoint = {
                                    'global_step': global_step,
                                    'model': (model.module.state_dict() if hasattr(model, 'module') else model.state_dict()),
                                    'optimizer': optimizer.state_dict(),
                                    'best_eval_measures_higher_better': best_eval_measures_higher_better,
                                    'best_eval_measures_lower_better': best_eval_measures_lower_better,
                                    'best_eval_steps': best_eval_steps
                                }
                                torch.save(checkpoint, os.path.join(args.log_directory, args.model_name + model_save_name))
                        eval_summary_writer.flush()
                model.train()
                if args.distributed:
                    dist.barrier()

            model_just_loaded = False
            global_step += 1

            if is_main_process(args):
                loop.set_postfix(loss=float(loss.item()), depth=float(loss_d.item()), classification=float(loss_c.item()))

    # ---------- Close writers ----------
    if is_main_process(args):
        if writer is not None:
            writer.close()
        if args.do_online_eval and eval_summary_writer is not None:
            eval_summary_writer.close()

    cleanup_distributed(args)


def main():
    if sys.argv.__len__() == 2:
        arg_filename_with_prefix = '@' + sys.argv[1]
        args = parser.parse_args([arg_filename_with_prefix])
    else:
        args = parser.parse_args()

    if args.mode != 'train':
        print('main.py is only for training. Use test.py instead.')
        return -1

    # runtime-based model name on rank-0
    runtime = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    args.model_name = runtime + '_' + args.model_name

    # pre-create run dir and copy config on rank-0 only
    # (we may not yet know distributed; create optimistically)
    os.makedirs(os.path.join(args.log_directory, args.model_name), exist_ok=True)
    if len(sys.argv) > 1 and not sys.argv[1].startswith('@'):
        args_out_path = os.path.join(args.log_directory, args.model_name, sys.argv[1])
        os.system(f'cp {sys.argv[1]} {args_out_path}')

    torch.cuda.empty_cache()
    if args.do_online_eval:
        print("You have specified --do_online_eval.")
        print(f"This will evaluate the model every eval_freq {args.eval_freq} steps and save best models for individual eval metrics.")

    main_worker(args)


if __name__ == '__main__':
    main()