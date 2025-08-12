import os
import sys
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from PIL import Image

from dataloaders.dataloader import TextDataLoader
from models.model import TRIDE
from utils.utils import post_process_depth, flip_lr

def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg

parser = argparse.ArgumentParser(description='TRIDE evaluation', fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args

# ----- Data -----
parser.add_argument('--mode',                                           type=str, default='test')
parser.add_argument('--main_path',                                      type=str, required=True)
parser.add_argument('--validation_image_path',                          type=str, required=True)
parser.add_argument('--validation_text_feature_general_path',           type=str, required=True)
parser.add_argument('--validation_text_feature_left_path',              type=str, required=True)
parser.add_argument('--validation_text_feature_mid_left_path',          type=str, required=True)
parser.add_argument('--validation_text_feature_mid_right_path',         type=str, required=True)
parser.add_argument('--validation_text_feature_right_path',             type=str, required=True)
parser.add_argument('--validation_weather_condition_path',              type=str, required=True)
parser.add_argument('--validation_ground_truth_path',                   type=str, required=True)
parser.add_argument('--validation_radar_path',                          type=str, required=True)

# ----- Model -----
parser.add_argument('--k',                                              type=int, default=4)
parser.add_argument('--encoder_radar',                                  type=str, default='resnet18')
parser.add_argument('--radar_input_channels',                           type=int, default=4)
parser.add_argument('--encoder',                                        type=str, default='resnet34_bts')
parser.add_argument('--n_filters_decoder',                              type=int, nargs='+', default=[256, 256, 128, 64, 32])
parser.add_argument('--input_height',                                   type=int, default=352)
parser.add_argument('--input_width',                                    type=int, default=704)
parser.add_argument('--max_depth',                                      type=float, default=100)
parser.add_argument('--fuse',                                           type=str, default='wafb')
parser.add_argument('--text_fuse',                                      type=str, default='cross_attention')
parser.add_argument('--text_hidden_dim',                                type=int, default=128)
parser.add_argument('--use_img_feat',                                   action='store_true')
parser.add_argument('--point_hidden_dim',                               type=int, default=128)

# ----- Eval / IO -----
parser.add_argument('--checkpoint_path',                                type=str, required=True)
parser.add_argument('--num_threads',                                    type=int, default=1)
parser.add_argument('--batch_size',                                     type=int, default=1)
parser.add_argument('--min_depth_eval',                                 type=float, default=1e-3)
parser.add_argument('--max_depth_eval',                                 type=float, default=80)
parser.add_argument('--store_prediction',                               action='store_true')
parser.add_argument('--save_dir',                                       type=str, default='./eval_result')
parser.add_argument('--post_process',                                   action='store_true') 

# ----- Distributed -----
parser.add_argument('--distributed',                                    action='store_true')
parser.add_argument('--dist_url',                                       default='env://', type=str)
parser.add_argument('--local_rank',                                     default=-1, type=int)
parser.add_argument('--seed',                                           default=42, type=int)

def init_distributed_mode(args):
    args.distributed = args.distributed or ("RANK" in os.environ and "WORLD_SIZE" in os.environ)
    if not args.distributed:
        args.rank = 0; args.world_size = 1
        args.local_rank = -1 if args.local_rank is None else args.local_rank
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.local_rank == -1:
        args.local_rank = int(os.environ["LOCAL_RANK"])
    args.rank = int(os.environ["RANK"])
    args.world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl', init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    dist.barrier()
    return torch.device('cuda', args.local_rank)

def is_main_process(args): return (not args.distributed) or args.rank == 0
def cleanup_distributed(args):
    if args.distributed and dist.is_initialized():
        dist.barrier(); dist.destroy_process_group()

def set_seed(seed):
    torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed_all(seed)

def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean(); d2 = (thresh < 1.25 * 2).mean(); d3 = (thresh < 1.25 * 3).mean()
    mae = np.mean(np.abs(gt - pred))
    rmse = np.sqrt(((gt - pred) ** 2).mean())
    rmse_log = np.sqrt(((np.log(gt) - np.log(pred)) ** 2).mean())
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)
    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - (np.mean(err) ** 2)) * 100
    log10 = np.mean(np.abs(np.log10(pred) - np.log10(gt)))
    # 返回顺序与打印一致
    return [silog, log10, abs_rel, sq_rel, rmse, rmse_log, d1, d2, d3, mae]

@torch.inference_mode()
def run_eval(args):
    device = init_distributed_mode(args)
    set_seed(args.seed); cudnn.benchmark = True

    # ----- DataLoader -----
    td = TextDataLoader(args, 'test')
    if args.distributed:
        ds = td.data.dataset
        sampler = DistributedSampler(ds, num_replicas=args.world_size, rank=args.rank, shuffle=False, drop_last=False)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, sampler=sampler,
                            num_workers=args.num_threads, pin_memory=True, drop_last=False)
    else:
        loader = td.data

    # ----- Model -----
    model = TRIDE(args).to(device)
    ckpt = torch.load(args.checkpoint_path, map_location=device)
    state = ckpt.get('model', ckpt) 
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        from collections import OrderedDict
        new_state = OrderedDict((k.replace('module.', ''), v) for k, v in state.items())
        model.load_state_dict(new_state, strict=True)

    if args.distributed:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank,
                    find_unused_parameters=False, broadcast_buffers=False)
    model.eval()

    # ----- store results -----
    if args.store_prediction and is_main_process(args):
        tag = os.path.basename(os.path.dirname(args.checkpoint_path)) or 'ckpt'
        save_dir = os.path.join(args.save_dir, tag)
        pred_depth_dir = os.path.join(save_dir, 'pred_depth')
        os.makedirs(pred_depth_dir, exist_ok=True)
    else:
        pred_depth_dir = None

    # ----- eval -----
    eval_measures = torch.zeros(11, device=device)   # 10 指标 + 计数
    cls_sums = torch.zeros(6, device=device)         # correct,total, rain_c,rain_t, night_c,night_t

    loop = tqdm(loader, disable=not is_main_process(args))
    for i, batch in enumerate(loop):
        image = batch['image'].to(device, non_blocking=True)
        text_mask = batch['text_mask'].to(device, non_blocking=True)
        t_gen = batch['text_feature_general'].to(device, non_blocking=True)
        t_l = batch['text_feature_left'].to(device, non_blocking=True)
        t_ml = batch['text_feature_mid_left'].to(device, non_blocking=True)
        t_mr = batch['text_feature_mid_right'].to(device, non_blocking=True)
        t_r = batch['text_feature_right'].to(device, non_blocking=True)
        t_len = batch['text_length'].to(device, non_blocking=True)

        radar_channels = batch['radar_channels'].to(device, non_blocking=True)
        radar_points = batch['radar_points'].to(device, non_blocking=True)

        label = batch['label'].to(device, non_blocking=True)
        gt_depth = batch['depth']

        pred_depth, class_pred = model(image, radar_channels, radar_points,
                                       t_gen, t_l, t_ml, t_mr, t_r, text_mask, t_len)

        if args.post_process:
            image_f = flip_lr(image)
            radar_f = flip_lr(radar_channels)
            mask_f = flip_lr(text_mask)
            pred_depth_f, _ = model(image_f, radar_f, radar_points,
                                    t_gen, t_l, t_ml, t_mr, t_r, mask_f, t_len)
            pred_depth = post_process_depth(pred_depth, pred_depth_f)

        # ---- 分类统计
        _, predicted = class_pred.max(1)
        if label.dim() == 2 and label.size(1) > 1:
            targets = label.argmax(dim=1)
        else:
            targets = label.view(-1).long()
        correct = (predicted == targets).sum().item()
        total = targets.numel()
        rain = (targets == 2); night = (targets == 1)
        cls_sums += torch.tensor([
            correct, total,
            ((predicted == 2) & rain).sum().item(), rain.sum().item(),
            ((predicted == 1) & night).sum().item(), night.sum().item()
        ], device=device, dtype=torch.float32)

        # ---- 深度指标
        pred_np = pred_depth.detach().cpu().numpy().squeeze()
        gt_np = gt_depth.cpu().numpy().squeeze()

        if args.store_prediction and is_main_process(args) and pred_depth_dir is not None:
            pd = np.uint32(np.clip(pred_np, 0, args.max_depth_eval) * 256.0)
            Image.fromarray(pd, mode='I').save(os.path.join(pred_depth_dir, f'{args.rank:02d}_{i:06d}.png'))

        pred_np[pred_np < args.min_depth_eval] = args.min_depth_eval
        pred_np[pred_np > args.max_depth_eval] = args.max_depth_eval
        pred_np[np.isinf(pred_np)] = args.max_depth_eval
        pred_np[np.isnan(pred_np)] = args.min_depth_eval

        valid = np.logical_and(gt_np > args.min_depth_eval, gt_np < args.max_depth_eval)
        measures = compute_errors(gt_np[valid], pred_np[valid])

        eval_measures[:-1] += torch.tensor(measures, device=device)
        eval_measures[-1] += 1

    if args.distributed:
        dist.all_reduce(eval_measures, op=dist.ReduceOp.SUM)
        dist.all_reduce(cls_sums, op=dist.ReduceOp.SUM)

    if is_main_process(args):
        cnt = int(eval_measures[-1].item())
        eval_measures /= max(cnt, 1)
        silog, log10, abs_rel, sq_rel, rmse, rmse_log, d1, d2, d3, mae = [m.item() for m in eval_measures[:10]]
        c, t, cr, tr, cn, tn = cls_sums.tolist()
        acc = 100.0 * (c / max(t, 1))
        acc_rain = 100.0 * (cr / max(tr, 1))
        acc_night = 100.0 * (cn / max(tn, 1))

        print('--------------Classification--------------')
        print('{:>7}, {:>13}, {:>14}'.format('Accuracy', 'Accuracy Rain', 'Accuracy Night'))
        print('{:.3f}%, {:.3f}%, {:.3f}%'.format(acc, acc_rain, acc_night))

        print('-------------Depth Estimation-------------')
        print('Computing errors for {} eval samples'.format(cnt))
        print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format(
              'silog', 'log10', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'd1', 'd2', 'd3', 'mae'))
        print("{:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}".format(
              silog, log10, abs_rel, sq_rel, rmse, rmse_log, d1, d2, d3, mae))

    cleanup_distributed(args)

def main():
    if sys.argv.__len__() == 2:
        arg_filename_with_prefix = '@' + sys.argv[1]
        args = parser.parse_args([arg_filename_with_prefix])
    else:
        args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_eval(args)

if __name__ == '__main__':
    main()