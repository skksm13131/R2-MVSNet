import argparse, os, sys, time, gc, datetime
import csv
import torch.nn as nn
import cv2
import numpy as np
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from datasets import find_dataset_def
from models import *
from utils import *
import torch.distributed as dist
# NOTE 2026-06-04: this file comes from the newer workflow, but this rebuild uses
# true-baseline models. Extra experimental model kwargs were intentionally
# filtered from the CascadeMVSNet constructor for compatibility.


cudnn.benchmark = True

parser = argparse.ArgumentParser(description='A PyTorch Implementation of Cascade Cost Volume MVSNet')
parser.add_argument('--mode', default='train', help='train or test', choices=['train', 'test', 'profile'])
parser.add_argument('--model', default='mvsnet', help='select model')
parser.add_argument('--device', default='cuda', help='select model')

parser.add_argument('--dataset', default='dtu_yao', help='select dataset')
parser.add_argument('--trainpath', default='/home/u104754251515/data/mvs_training/dtu', help='train datapath')
parser.add_argument('--testpath', default='/home/u104754251515/data/mvs_training/dtu', help='test datapath')
parser.add_argument('--trainlist', default='lists/dtu/train.txt', help='train list')
parser.add_argument('--testlist', default='lists/dtu/test.txt', help='test list')

parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lrepochs', type=str, default="10,12,14:2", help='epoch ids to downscale lr and the downscale rate')
parser.add_argument('--wd', type=float, default=0.0, help='weight decay')
parser.add_argument('--views_train', type=int, default=5, help='number of views to train')
parser.add_argument('--views_test', type=int, default=5, help='number of views to test')

parser.add_argument('--batch_size', type=int, default=4, help='train batch size')
parser.add_argument('--numdepth', type=int, default=192, help='the number of depth values')
parser.add_argument('--interval_scale', type=float, default=1.06, help='the number of depth values')

parser.add_argument('--loadckpt', default=None, help='load a specific checkpoint')
parser.add_argument('--logdir', default='./checkpoints/debug', help='the directory to save checkpoints/logs')
parser.add_argument('--metricsdir', default='./results', help='the directory to save metrics csv and loss plot')
parser.add_argument('--resume', action='store_true', help='continue to train the model')
parser.add_argument('--log111', default='./', help='the directory to save logs')

parser.add_argument('--summary_freq', type=int, default=100, help='print and summary frequency')
parser.add_argument('--save_freq', type=int, default=1, help='save checkpoint frequency')
parser.add_argument('--eval_freq', type=int, default=1, help='eval freq')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')
parser.add_argument('--pin_m', action='store_true', help='data loader pin memory')
parser.add_argument("--local_rank", type=int, default=0)

parser.add_argument('--share_cr', action='store_true', help='whether share the cost volume regularization')
parser.add_argument('--ndepths', type=str, default="48,32,8", help='ndepths')
parser.add_argument('--depth_inter_r', type=str, default="4,2,1", help='depth_intervals_ratio')
parser.add_argument('--dlossw', type=str, default="0.5,1.0,2.0", help='depth loss weight for different stage')
parser.add_argument('--cr_base_chs', type=str, default="8,8,8", help='cost regularization base channels')
parser.add_argument('--grad_method', type=str, default="detach", choices=["detach", "undetach"], help='grad method')
parser.add_argument('--use_eta', action='store_true', help='enable ETA visibility weighting')
parser.add_argument('--use_view_attention', action='store_true', help='enable warped-view reliability fusion in DepthNet')
parser.add_argument('--view_attention_mode', type=str, default='legacy', choices=['legacy', 'residual_fusion', 'progressive_residual_fusion', 'single_pass_reliability_weighted'], help='view attention / reliability fusion mode')
parser.add_argument('--use_reliability_guidance', action='store_true', help='enable confidence-guided stage interval/reliability propagation')
parser.add_argument('--reliability_stage_start', type=int, default=2, help='stage to start reliability guidance')
parser.add_argument('--reliability_interval_scale', type=float, default=0.5, help='reliability-guided interval scale')
parser.add_argument('--reliability_confidence_mix', type=float, default=0.5, help='photometric/texture confidence mix')
parser.add_argument('--reliability_min_interval_factor', type=float, default=0.5, help='minimum depth interval factor under reliability guidance')
parser.add_argument('--stage_consistency_start', type=int, default=2, help='stage to start SCRF stage-consistency gating')
parser.add_argument('--stage_consistency_weight', type=float, default=0.5, help='SCRF geometry agreement weight in stage-consistency prior')
parser.add_argument('--stage_consistency_temperature', type=float, default=1.0, help='SCRF geometry agreement temperature')
parser.add_argument('--use_hypothesis_sampling', action='store_true', help='enable RAHS reliability-aware hypothesis sampling')
parser.add_argument('--hypothesis_sampling_start', type=int, default=2, help='stage to start RAHS hypothesis sampling')
parser.add_argument('--hypothesis_sampling_strength', type=float, default=0.55, help='RAHS hypothesis redistribution strength')
parser.add_argument('--hypothesis_sampling_shift_scale', type=float, default=0.35, help='RAHS bounded center-shift scale in depth intervals')
parser.add_argument('--use_cadr', action='store_true', help='enable CADR confidence-aware distribution regression')
parser.add_argument('--cadr_stage_start', type=int, default=2, help='stage to start CADR distribution regression')
parser.add_argument('--cadr_window_radius', type=int, default=2, help='CADR local peak-window radius in hypothesis bins')
parser.add_argument('--cadr_max_residual_ratio', type=float, default=0.45, help='CADR maximum residual as a ratio of stage interval')
parser.add_argument('--cadr_confidence_mix', type=float, default=0.5, help='CADR blend between propagated confidence and peak confidence')
parser.add_argument('--use_rmfe', action='store_true', help='enable RMFE learnable multi-scale feature enhancement')
parser.add_argument('--use_rafe', action='store_true', help='enable reliability-aware feature extraction')
parser.add_argument('--use_adaptive_r2', action='store_true', help='enable difficulty-adaptive RAFE and SP-RWCV gating')
parser.add_argument('--use_ugdr', action='store_true', help='enable UGDR final-stage bounded depth refinement')
parser.add_argument('--ugdr_max_residual_ratio', type=float, default=0.5, help='UGDR maximum residual as a ratio of final-stage interval')

parser.add_argument('--using_apex', action='store_true', help='using apex, need to install apex')
parser.add_argument('--sync_bn', action='store_true', help='enabling apex sync BN.')
parser.add_argument('--opt-level', type=str, default="O0")
parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
parser.add_argument('--loss-scale', type=str, default=None)

# 设备 & 分布式
num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
is_distributed = num_gpus > 1


# ===============================================================================
# 【核心修改】新增：鲁棒性深度评估指标计算函数
# ===============================================================================
def compute_robust_metrics(depth_est, depth_gt, mask):
    """
    计算更能反映点云重建质量的指标。
    核心思想：最终的点云融合(Fusion)步骤会丢弃误差大的点，
    所以训练时的评估也应该忽略掉最差的那部分点(Outliers)，关注好点的精度。
    """
    # 确保 mask 是 bool 类型
    mask = mask > 0.5

    # 只取有效区域
    depth_est = depth_est[mask].detach()
    depth_gt = depth_gt[mask].detach()

    if depth_est.numel() == 0:
        return {
            "trimmed_mae_90": 0.0,
            "trimmed_mae_80": 0.0,
            "outlier_ratio_2mm": 0.0,
            "outlier_ratio_4mm": 0.0
        }

    # 计算绝对误差
    abs_error = torch.abs(depth_est - depth_gt)

    # 1. Trimmed MAE (90%): 排除掉误差最大的10%像素后计算平均误差
    # 这模拟了 Fusibile 过滤掉 outlier 后的情况
    num_samples = abs_error.numel()
    num_keep_90 = int(0.9 * num_samples)

    if num_keep_90 > 0:
        # topk largest=False 取最小的k个（即误差最小的90%）
        topk_values_90, _ = torch.topk(abs_error, k=num_keep_90, largest=False)
        trimmed_mae_90 = topk_values_90.mean()
    else:
        trimmed_mae_90 = abs_error.mean()

    # 2. Trimmed MAE (80%): 更严格，模拟更激进的滤波
    num_keep_80 = int(0.8 * num_samples)
    if num_keep_80 > 0:
        topk_values_80, _ = torch.topk(abs_error, k=num_keep_80, largest=False)
        trimmed_mae_80 = topk_values_80.mean()
    else:
        trimmed_mae_80 = abs_error.mean()

    # 3. Outlier Ratios (误差大于阈值的比例)
    # 比平均误差更能反映有多少点是"完全不可用"的
    outlier_ratio_2mm = (abs_error > 2.0).float().mean() * 100.0
    outlier_ratio_4mm = (abs_error > 4.0).float().mean() * 100.0

    return {
        "trimmed_mae_90": trimmed_mae_90,
        "trimmed_mae_80": trimmed_mae_80,
        "outlier_ratio_2mm": outlier_ratio_2mm,
        "outlier_ratio_4mm": outlier_ratio_4mm
    }


# main function
def train(model, model_loss, optimizer, TrainImgLoader, TestImgLoader, start_epoch, args):
    if (not is_distributed) or (dist.get_rank() == 0):
        tb_logger = SummaryWriter(args.logdir)

        # 1. 创建专门用于存放指标的文件夹
        os.makedirs(args.metricsdir, exist_ok=True)
        csv_path = os.path.join(args.metricsdir, 'all_epoch_metrics.csv')
        # 2. 定义完整的CSV表头，确保所有指标都被包含
        # 【核心修改】新增了鲁棒指标到 CSV 表头
        csv_header = [
            'epoch', 'lr', 'train_loss', 'train_depth_loss', 'train_abs_depth_error',
            'test_loss', 'test_depth_loss',
            'test_abs_depth_error', 'test_thres2mm_error',
            'test_trimmed_mae_90', 'test_trimmed_mae_80',  # 新增：截断误差（越低越好，代表有效点云精度）
            'test_out_ratio_2mm', 'test_out_ratio_4mm'  # 新增：坏点比例（越低越好）
        ]
        # 3. 如果是新训练，则创建CSV并写入表头
        if not args.resume and start_epoch == 0:
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(csv_header)

    milestones = [len(TrainImgLoader) * int(epoch_idx) for epoch_idx in args.lrepochs.split(':')[0].split(',')]
    lr_gamma = 1 / float(args.lrepochs.split(':')[1])
    lr_scheduler = WarmupMultiStepLR(optimizer, milestones, gamma=lr_gamma, warmup_factor=1.0 / 3, warmup_iters=500,
                                     last_epoch=len(TrainImgLoader) * start_epoch - 1)

    for epoch_idx in range(start_epoch, args.epochs):
        print('Epoch {}:'.format(epoch_idx))
        global_step = len(TrainImgLoader) * epoch_idx

        # training
        model.train()
        epoch_train_loss = 0.0
        num_train_batches = 0
        if (not is_distributed) or (dist.get_rank() == 0):
            avg_train_scalars = DictAverageMeter()
        for batch_idx, sample in enumerate(TrainImgLoader):
            start_time = time.time()
            loss, scalar_outputs, image_outputs = train_sample(model, model_loss, optimizer, sample, args)
            epoch_train_loss += loss
            num_train_batches += 1
            do_summary = global_step % args.summary_freq == 0
            lr_scheduler.step()
            avg_train_scalars.update(scalar_outputs)
            if (not is_distributed) or (dist.get_rank() == 0):
                if do_summary:
                    save_scalars(tb_logger, 'train', scalar_outputs, global_step)
                    save_images(tb_logger, 'train', image_outputs, global_step)
                    print(
                        "Epoch {}/{}, Iter {}/{}, lr {:.6f}, train loss = {:.3f}, depth loss = {:.3f}, time = {:.3f}".format(
                            epoch_idx, args.epochs, batch_idx, len(TrainImgLoader),
                            optimizer.param_groups[0]["lr"], loss,
                            scalar_outputs['depth_loss'],
                            time.time() - start_time))
            global_step += 1
            del scalar_outputs, image_outputs

        if (not is_distributed) or (dist.get_rank() == 0):
            avg_train_metrics = avg_train_scalars.mean()
            print(f"avg_train_scalars: {avg_train_metrics}")

        # checkpoint
        if (not is_distributed) or (dist.get_rank() == 0):
            if (epoch_idx + 1) % args.save_freq == 0:
                torch.save({
                    'epoch': epoch_idx,
                    'model': model.module.state_dict(),
                    'optimizer': optimizer.state_dict()},
                    "{}/model_{:0>6}.ckpt".format(args.logdir, epoch_idx))
        gc.collect()

        # testing
        if (epoch_idx % args.eval_freq == 0) or (epoch_idx == args.epochs - 1):
            avg_test_scalars = DictAverageMeter()
            for batch_idx, sample in enumerate(TestImgLoader):
                start_time = time.time()
                do_summary = global_step % args.summary_freq == 0
                loss, scalar_outputs, image_outputs = test_sample_depth(model, model_loss, sample, args)
                avg_test_scalars.update(scalar_outputs)
                if (not is_distributed) or (dist.get_rank() == 0):
                    if do_summary:
                        save_scalars(tb_logger, 'test', scalar_outputs, global_step)
                        save_images(tb_logger, 'test', image_outputs, global_step)
                        print(
                            "Epoch {}/{}, Iter {}/{}, test loss = {:.3f}, depth loss = {:.3f}, Trimmed90 = {:.3f}, Trimmed80 = {:.3f}".format(
                                epoch_idx, args.epochs,
                                batch_idx,
                                len(TestImgLoader), loss,
                                scalar_outputs["depth_loss"],
                                scalar_outputs["trimmed_mae_90"],
                                scalar_outputs["trimmed_mae_80"]))
                global_step += 1
                del scalar_outputs, image_outputs

            if (not is_distributed) or (dist.get_rank() == 0):
                test_metrics = avg_test_scalars.mean()
                save_scalars(tb_logger, 'fulltest', avg_test_scalars.mean(), global_step)
                print("avg_test_scalars:", avg_test_scalars.mean())
            gc.collect()

        if (not is_distributed) or (dist.get_rank() == 0):
            # 1. 准备一行待写入CSV的数据
            row_data = {'epoch': epoch_idx, 'lr': optimizer.param_groups[0]["lr"]}
            for key, val in avg_train_metrics.items():
                row_data[f"train_{key}"] = val

            if test_metrics:
                for key, val in test_metrics.items():
                    row_data[f"test_{key}"] = val

            # 【核心修改点】将写入操作移出循环，保证一个 Epoch 只写一行
            with open(csv_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=csv_header, extrasaction='ignore')
                writer.writerow(row_data)


def test(model, model_loss, TestImgLoader, args):
    avg_test_scalars = DictAverageMeter()
    for batch_idx, sample in enumerate(TestImgLoader):
        start_time = time.time()
        loss, scalar_outputs, image_outputs = test_sample_depth(model, model_loss, sample, args)
        avg_test_scalars.update(scalar_outputs)
        del scalar_outputs, image_outputs
        if (not is_distributed) or (dist.get_rank() == 0):
            print('Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(batch_idx, len(TestImgLoader), loss,
                                                                        time.time() - start_time))
            if batch_idx % 100 == 0:
                print("Iter {}/{}, test results = {}".format(batch_idx, len(TestImgLoader), avg_test_scalars.mean()))
    if (not is_distributed) or (dist.get_rank() == 0):
        print("final", avg_test_scalars.mean())


def unpack_model_loss(loss_outputs):
    if len(loss_outputs) == 4:
        return loss_outputs
    if len(loss_outputs) == 2:
        total_loss, depth_loss = loss_outputs
        zero_loss = total_loss.new_tensor(0.0)
        return total_loss, depth_loss, zero_loss, zero_loss
    raise ValueError('model_loss must return 2 or 4 values, got {}'.format(len(loss_outputs)))


def train_sample(model, model_loss, optimizer, sample, args):
    model.train()
    optimizer.zero_grad()

    sample_cuda = tocuda(sample)
    depth_gt_ms = sample_cuda["depth"]
    mask_ms = sample_cuda["mask"]

    num_stage = len([int(nd) for nd in args.ndepths.split(",") if nd])
    depth_gt = depth_gt_ms["stage{}".format(num_stage)]
    mask = mask_ms["stage{}".format(num_stage)]

    outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])
    depth_est = outputs["depth"]

    total_loss, depth_loss, normal_loss, dndr_loss = unpack_model_loss(model_loss(outputs, depth_gt_ms, mask_ms, dlossw=[float(e) for e in args.dlossw.split(",") if e]))

    if is_distributed and args.using_apex:
        with amp.scale_loss(total_loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        total_loss.backward()

    optimizer.step()

    scalar_outputs = {"loss": total_loss,
                      "depth_loss": depth_loss,
                      "normal_loss": normal_loss,
                      "dndr_loss": dndr_loss,
                      "abs_depth_error": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5)}

    image_outputs = {"depth_est": depth_est * mask,
                     "depth_est_nomask": depth_est,
                     "depth_gt": sample["depth"]["stage{}".format(num_stage)],
                     "ref_img": sample["imgs"][:, 0],
                     "mask": sample["mask"]["stage{}".format(num_stage)],
                     "errormap": (depth_est - depth_gt).abs() * mask,
                     }
    if is_distributed:
        scalar_outputs = reduce_scalar_outputs(scalar_outputs)

    return tensor2float(scalar_outputs["loss"]), tensor2float(scalar_outputs), tensor2numpy(image_outputs)


@make_nograd_func
def test_sample_depth(model, model_loss, sample, args):
    if is_distributed:
        model_eval = model.module
    else:
        model_eval = model.module
    model_eval.eval()

    sample_cuda = tocuda(sample)
    depth_gt_ms = sample_cuda["depth"]
    mask_ms = sample_cuda["mask"]

    num_stage = len([int(nd) for nd in args.ndepths.split(",") if nd])
    depth_gt = depth_gt_ms["stage{}".format(num_stage)]
    mask = mask_ms["stage{}".format(num_stage)]

    outputs = model_eval(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])
    depth_est = outputs["depth"]

    total_loss, depth_loss, normal_loss, dndr_loss = unpack_model_loss(model_loss(outputs, depth_gt_ms, mask_ms, dlossw=[float(e) for e in args.dlossw.split(",") if e]))

    # ===============================================================================
    # 【核心修改】计算鲁棒性指标
    # ===============================================================================
    robust_metrics = compute_robust_metrics(depth_est, depth_gt, mask > 0.5)

    scalar_outputs = {"loss": total_loss,
                      "depth_loss": depth_loss,
                      "normal_loss": normal_loss,
                      "dndr_loss": dndr_loss,
                      "abs_depth_error": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5),
                      "thres2mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 2),
                      # 新增的指标
                      "trimmed_mae_90": robust_metrics["trimmed_mae_90"],
                      "trimmed_mae_80": robust_metrics["trimmed_mae_80"],
                      "out_ratio_2mm": robust_metrics["outlier_ratio_2mm"],
                      "out_ratio_4mm": robust_metrics["outlier_ratio_4mm"],
                      }

    image_outputs = {"depth_est": depth_est * mask,
                     "depth_est_nomask": depth_est,
                     "depth_gt": sample["depth"]["stage{}".format(num_stage)],
                     "ref_img": sample["imgs"][:, 0],
                     "mask": sample["mask"]["stage{}".format(num_stage)],
                     "errormap": (depth_est - depth_gt).abs() * mask}

    if is_distributed:
        scalar_outputs = reduce_scalar_outputs(scalar_outputs)

    return tensor2float(scalar_outputs["loss"]), tensor2float(scalar_outputs), tensor2numpy(image_outputs)


def profile():
    warmup_iter = 5
    iter_dataloader = iter(TestImgLoader)

    @make_nograd_func
    def do_iteration():
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        test_sample_depth(model, model_loss, next(iter_dataloader), args)
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        return end_time - start_time

    for i in range(warmup_iter):
        t = do_iteration()
        print('WarpUp Iter {}, time = {:.4f}'.format(i, t))

    with torch.autograd.profiler.profile(enabled=True, use_cuda=True) as prof:
        for i in range(5):
            t = do_iteration()
            print('Profile Iter {}, time = {:.4f}'.format(i, t))
            time.sleep(0.02)

    if prof is not None:
        # print(prof)
        trace_fn = 'chrome-trace.bin'
        prof.export_chrome_trace(trace_fn)
        print("chrome trace file is written to: ", trace_fn)


if __name__ == '__main__':
    # parse arguments and check
    args = parser.parse_args()

    # using sync_bn by using nvidia-apex, need to install apex.
    if args.sync_bn:
        assert args.using_apex, "must set using apex and install nvidia-apex"
    if args.using_apex:
        try:
            from apex.parallel import DistributedDataParallel as DDP
            from apex.fp16_utils import *
            from apex import amp, optimizers
            from apex.multi_tensor_apply import multi_tensor_applier
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

    if args.resume:
        assert args.mode == "train"
        assert args.loadckpt is None
    if args.testpath is None:
        args.testpath = args.trainpath

    if is_distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    set_random_seed(args.seed)
    device = torch.device(args.device)

    if (not is_distributed) or (dist.get_rank() == 0):
        # create logger for mode "train" and "testall"
        if args.mode == "train":
            if not os.path.isdir(args.logdir):
                os.makedirs(args.logdir)
            current_time_str = str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
            print("current time", current_time_str)
            print("creating new summary file")
            logger = SummaryWriter(args.logdir)
        print("argv:", sys.argv[1:])
        print_args(args)

    # model, optimizer
    model = CascadeMVSNet(refine=False,
                          ndepths=[int(nd) for nd in args.ndepths.split(",") if nd],
                          depth_interals_ratio=[float(d_i) for d_i in args.depth_inter_r.split(",") if d_i],
                          share_cr=args.share_cr,
                          cr_base_chs=[int(ch) for ch in args.cr_base_chs.split(",") if ch],
                          grad_method=args.grad_method,
                          use_view_attention=args.use_view_attention,
                          view_attention_mode=args.view_attention_mode,
                          use_rafe=args.use_rafe,
                          use_adaptive_r2=args.use_adaptive_r2)
    model.to(device)
    model_loss = cas_mvsnet_loss

    if args.sync_bn:
        import apex

        print("using apex synced BN")
        model = apex.parallel.convert_syncbn_model(model)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.9, 0.999),
                           weight_decay=args.wd)

    # load parameters
    start_epoch = 0
    if args.resume:
        saved_models = [fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt")]
        saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        # use the latest checkpoint file
        loadckpt = os.path.join(args.logdir, saved_models[-1])
        print("resuming", loadckpt)
        state_dict = torch.load(loadckpt, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
        start_epoch = state_dict['epoch'] + 1
    elif args.loadckpt:
        # load checkpoint file specified by args.loadckpt
        print("loading model {}".format(args.loadckpt))
        state_dict = torch.load(args.loadckpt, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict['model'])

    if (not is_distributed) or (dist.get_rank() == 0):
        print("start at epoch {}".format(start_epoch))
        print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    if args.using_apex:
        # Initialize Amp
        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level=args.opt_level,
                                          keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                                          loss_scale=args.loss_scale
                                          )

    if is_distributed:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank,
            # find_unused_parameters=False,
            # this should be removed if we update BatchNorm stats
            # broadcast_buffers=False,
        )
    else:
        if torch.cuda.is_available():
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)

    # dataset, dataloader
    MVSDataset = find_dataset_def(args.dataset)
    train_dataset = MVSDataset(args.trainpath, args.trainlist, "train", args.views_train, args.numdepth, args.interval_scale)
    test_dataset = MVSDataset(args.testpath, args.testlist, "test", args.views_test, args.numdepth, args.interval_scale)

    if is_distributed:
        train_sampler = torch.utils.data.DistributedSampler(train_dataset, num_replicas=dist.get_world_size(),
                                                            rank=dist.get_rank())
        test_sampler = torch.utils.data.DistributedSampler(test_dataset, num_replicas=dist.get_world_size(),
                                                           rank=dist.get_rank())

        TrainImgLoader = DataLoader(train_dataset, args.batch_size, sampler=train_sampler, num_workers=1,
                                    drop_last=True,
                                    pin_memory=args.pin_m)
        TestImgLoader = DataLoader(test_dataset, args.batch_size, sampler=test_sampler, num_workers=1, drop_last=False,
                                   pin_memory=args.pin_m)
    else:
        TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=1, drop_last=True,
                                    pin_memory=args.pin_m)
        TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=1, drop_last=False,
                                   pin_memory=args.pin_m)

    if args.mode == "train":
        train(model, model_loss, optimizer, TrainImgLoader, TestImgLoader, start_epoch, args)
    elif args.mode == "test":
        test(model, model_loss, TestImgLoader, args)
    elif args.mode == "profile":
        profile()
    else:
        raise NotImplementedError
