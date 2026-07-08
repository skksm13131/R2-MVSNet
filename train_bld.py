import argparse, os, sys, time, gc, datetime
import csv
import torch.nn as nn
import cv2
import numpy as np
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from datasets.blend import BlendedMVSDataset
from models import *
from utils import *
import torch.distributed as dist

cudnn.benchmark = True

SHARED_ROOT = os.environ.get("R2MVSNET_SHARED_ROOT", "/root/shared-nvme")
BLD_ROOT = os.environ.get("R2MVSNET_BLENDED_PATH", os.path.join(SHARED_ROOT, "BlendedMVS"))

parser = argparse.ArgumentParser(description='R2-MVSNet training on BlendedMVS')
parser.add_argument('--mode', default='train', help='train or test', choices=['train', 'test', 'profile'])
parser.add_argument('--model', default='mvsnet', help='select model')
parser.add_argument('--device', default='cuda', help='select model')
parser.add_argument('--dataset', default='blend', help='select dataset')
parser.add_argument('--trainpath', default=BLD_ROOT, help='train datapath')
parser.add_argument('--testpath', help='test datapath')
parser.add_argument('--trainlist', default='lists/blendedmvs/train.txt', help='train list')
parser.add_argument('--testlist', default='lists/blendedmvs/val.txt', help='test list')
parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
parser.add_argument('--lrepochs', type=str, default="3,5,7:2", help='epoch ids to downscale lr and the downscale rate')
parser.add_argument('--wd', type=float, default=0.0, help='weight decay')
parser.add_argument('--nviews', type=int, default=7, help='number of views')
parser.add_argument('--batch_size', type=int, default=2, help='train batch size')
parser.add_argument('--numdepth', type=int, default=192, help='the number of depth values')
parser.add_argument('--interval_scale', type=float, default=1.06, help='the number of depth values')
parser.add_argument('--loadckpt', default='checkpoints/model_000000.ckpt', help='load a specific checkpoint')
parser.add_argument('--logdir', default='./checkpoints_bld', help='the directory to save checkpoints/logs')
parser.add_argument('--metricsdir', default='./results', help='the directory to save metrics csv and loss plot')
parser.add_argument('--resume', action='store_true', help='continue to train the model')
parser.add_argument('--log111', default='./', help='the directory to save logs')
parser.add_argument('--summary_freq', type=int, default=100, help='print and summary frequency')
parser.add_argument('--save_freq', type=int, default=1, help='save checkpoint frequency')
parser.add_argument('--eval_freq', type=int, default=1, help='eval freq')
parser.add_argument('--max_train_batches', type=int, default=0,
                    help='debug only: stop each training epoch after this many batches when > 0')
parser.add_argument('--max_test_batches', type=int, default=0,
                    help='debug only: stop each validation epoch after this many batches when > 0')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')
parser.add_argument('--pin_m', action='store_true', help='data loader pin memory')
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument('--share_cr', action='store_true', help='whether share the cost volume regularization')
parser.add_argument('--ndepths', type=str, default="96,32,8", help='ndepths')
parser.add_argument('--depth_inter_r', type=str, default="2,2,1", help='depth_intervals_ratio')
parser.add_argument('--dlossw', type=str, default="1.0,1.0,1.0", help='depth loss weight for different stage')
parser.add_argument('--cr_base_chs', type=str, default="8,8,8", help='cost regularization base channels')
parser.add_argument('--grad_method', type=str, default="detach", choices=["detach", "undetach"], help='grad method')

# R2 modules
parser.add_argument('--use_view_attention', action='store_true', help='enable SP-RWCV')
parser.add_argument('--view_attention_mode', type=str, default='single_pass_reliability_weighted',
                    choices=['single_pass_reliability_weighted'])
parser.add_argument('--use_rafe', action='store_true', help='enable RAFE')
parser.add_argument('--use_fgdr', action='store_true', help='enable FGDR')
parser.add_argument('--fgdr_max_radius_factor', type=float, default=2.0)
parser.add_argument('--fgdr_anchor_base', action='store_true')
parser.add_argument('--fgdr_loss_weight', type=float, default=0.05)
parser.add_argument('--fgdr_radius_weight', type=float, default=0.1)
parser.add_argument('--fgdr_center_weight', type=float, default=0.25)

parser.add_argument('--using_apex', action='store_true', help='using apex, need to install apex')
parser.add_argument('--sync_bn', action='store_true', help='enabling apex sync BN.')
parser.add_argument('--opt-level', type=str, default="O0")
parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
parser.add_argument('--loss-scale', type=str, default=None)

num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
is_distributed = num_gpus > 1


def unpack_model_loss(loss_outputs):
    if len(loss_outputs) == 4:
        return loss_outputs
    if len(loss_outputs) == 2:
        total_loss, depth_loss = loss_outputs
        zero_loss = total_loss.new_tensor(0.0)
        return total_loss, depth_loss, zero_loss, zero_loss
    raise ValueError('model_loss must return 2 or 4 values, got {}'.format(len(loss_outputs)))


def train(model, model_loss, optimizer, TrainImgLoader, TestImgLoader, start_epoch, args):
    if (not is_distributed) or (dist.get_rank() == 0):
        tb_logger = SummaryWriter(args.logdir)
        os.makedirs(args.metricsdir, exist_ok=True)
        csv_path = os.path.join(args.metricsdir, 'all_epoch_metrics.csv')
        csv_header = [
            'epoch', 'lr', 'train_loss', 'train_depth_loss',
            'test_loss', 'test_depth_loss', 'test_abs_depth_error', 'test_thres2mm_error'
        ]
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
        if (not is_distributed) or (dist.get_rank() == 0):
            avg_train_scalars = DictAverageMeter()
        for batch_idx, sample in enumerate(TrainImgLoader):
            loss, scalar_outputs, image_outputs = train_sample(model, model_loss, optimizer, sample, args)
            epoch_train_loss += loss
            start_time = time.time()
            do_summary = global_step % args.summary_freq == 0
            lr_scheduler.step()
            avg_train_scalars.update(scalar_outputs)
            if (not is_distributed) or (dist.get_rank() == 0):
                if do_summary:
                    save_scalars(tb_logger, 'train', scalar_outputs, global_step)
                    save_images(tb_logger, 'train', image_outputs, global_step)
                    print("Epoch {}/{}, Iter {}/{}, lr {:.6f}, train loss = {:.3f}, time = {:.3f}".format(
                        epoch_idx, args.epochs, batch_idx, len(TrainImgLoader),
                        optimizer.param_groups[0]["lr"], loss, time.time() - start_time))
            global_step += 1
            del scalar_outputs, image_outputs
            if args.max_train_batches > 0 and (batch_idx + 1) >= args.max_train_batches:
                break

        if (not is_distributed) or (dist.get_rank() == 0):
            avg_train_metrics = avg_train_scalars.mean()

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
        test_metrics = None
        if (epoch_idx % args.eval_freq == 0) or (epoch_idx == args.epochs - 1):
            avg_test_scalars = DictAverageMeter()
            for batch_idx, sample in enumerate(TestImgLoader):
                do_summary = global_step % args.summary_freq == 0
                loss, scalar_outputs, image_outputs = test_sample_depth(model, model_loss, sample, args)
                avg_test_scalars.update(scalar_outputs)
                if (not is_distributed) or (dist.get_rank() == 0):
                    if do_summary:
                        save_scalars(tb_logger, 'test', scalar_outputs, global_step)
                        save_images(tb_logger, 'test', image_outputs, global_step)
                        print("Epoch {}/{}, Iter {}/{}, test loss = {:.3f}".format(
                            epoch_idx, args.epochs, batch_idx, len(TestImgLoader), loss))
                global_step += 1
                del scalar_outputs, image_outputs
                if args.max_test_batches > 0 and (batch_idx + 1) >= args.max_test_batches:
                    break

            if (not is_distributed) or (dist.get_rank() == 0):
                test_metrics = avg_test_scalars.mean()
                save_scalars(tb_logger, 'fulltest', test_metrics, global_step)
            gc.collect()

        if (not is_distributed) or (dist.get_rank() == 0):
            row_data = {'epoch': epoch_idx, 'lr': optimizer.param_groups[0]["lr"]}
            for key, val in avg_train_metrics.items():
                row_data[f"train_{key}"] = val
            if test_metrics:
                for key, val in test_metrics.items():
                    row_data[f"test_{key}"] = val
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
        if args.max_test_batches > 0 and (batch_idx + 1) >= args.max_test_batches:
            break
    if (not is_distributed) or (dist.get_rank() == 0):
        print("final", avg_test_scalars.mean())


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

    total_loss, depth_loss, _, fgdr_loss = unpack_model_loss(model_loss(
        outputs, depth_gt_ms, mask_ms,
        dlossw=[float(e) for e in args.dlossw.split(",") if e],
        fgdr_loss_weight=args.fgdr_loss_weight if args.use_fgdr else 0.0,
        fgdr_radius_weight=args.fgdr_radius_weight,
        fgdr_center_weight=args.fgdr_center_weight,
    ))

    if is_distributed and args.using_apex:
        with amp.scale_loss(total_loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        total_loss.backward()
    optimizer.step()

    scalar_outputs = {"loss": total_loss, "depth_loss": depth_loss,
                      "abs_depth_error": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5)}
    if args.use_fgdr:
        scalar_outputs["fgdr_loss"] = fgdr_loss

    image_outputs = {"depth_est": depth_est * mask,
                     "depth_est_nomask": depth_est,
                     "depth_gt": sample["depth"]["stage{}".format(num_stage)],
                     "ref_img": sample["imgs"][:, 0],
                     "mask": sample["mask"]["stage{}".format(num_stage)],
                     "errormap": (depth_est - depth_gt).abs() * mask}
    if is_distributed:
        scalar_outputs = reduce_scalar_outputs(scalar_outputs)
    return tensor2float(scalar_outputs["loss"]), tensor2float(scalar_outputs), tensor2numpy(image_outputs)


@make_nograd_func
def test_sample_depth(model, model_loss, sample, args):
    model_eval = model.module if is_distributed else model
    model_eval.eval()
    sample_cuda = tocuda(sample)
    depth_gt_ms = sample_cuda["depth"]
    mask_ms = sample_cuda["mask"]
    num_stage = len([int(nd) for nd in args.ndepths.split(",") if nd])
    depth_gt = depth_gt_ms["stage{}".format(num_stage)]
    mask = mask_ms["stage{}".format(num_stage)]
    outputs = model_eval(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])
    depth_est = outputs["depth"]

    total_loss, depth_loss, _, _ = unpack_model_loss(model_loss(
        outputs, depth_gt_ms, mask_ms,
        dlossw=[float(e) for e in args.dlossw.split(",") if e],
        fgdr_loss_weight=args.fgdr_loss_weight if args.use_fgdr else 0.0,
        fgdr_radius_weight=args.fgdr_radius_weight,
        fgdr_center_weight=args.fgdr_center_weight,
    ))

    scalar_outputs = {"loss": total_loss, "depth_loss": depth_loss,
                      "abs_depth_error": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5),
                      "thres2mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 2)}
    image_outputs = {"depth_est": depth_est * mask, "depth_est_nomask": depth_est,
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
        trace_fn = 'chrome-trace.bin'
        prof.export_chrome_trace(trace_fn)
        print("chrome trace file is written to: ", trace_fn)


if __name__ == '__main__':
    args = parser.parse_args()

    if args.sync_bn:
        assert args.using_apex, "must set using apex and install nvidia-apex"
    if args.using_apex:
        try:
            from apex.parallel import DistributedDataParallel as DDP
            from apex.fp16_utils import *
            from apex import amp, optimizers
            from apex.multi_tensor_apply import multi_tensor_applier
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex")

    if args.resume:
        assert args.mode == "train"
        assert args.loadckpt is None
    if args.testpath is None:
        args.testpath = args.trainpath

    if is_distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    set_random_seed(args.seed)
    device = torch.device(args.device)

    if (not is_distributed) or (dist.get_rank() == 0):
        if args.mode == "train":
            if not os.path.isdir(args.logdir):
                os.makedirs(args.logdir)
            print("current time", str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')))
            logger = SummaryWriter(args.logdir)
        print("argv:", sys.argv[1:])
        print_args(args)

    model = CascadeMVSNet(
        ndepths=[int(nd) for nd in args.ndepths.split(",") if nd],
        depth_interals_ratio=[float(d_i) for d_i in args.depth_inter_r.split(",") if d_i],
        share_cr=args.share_cr,
        cr_base_chs=[int(ch) for ch in args.cr_base_chs.split(",") if ch],
        grad_method=args.grad_method,
        use_rafe=args.use_rafe,
        use_view_attention=args.use_view_attention,
        view_attention_mode=args.view_attention_mode,
        use_fgdr=args.use_fgdr,
        fgdr_anchor_base=args.fgdr_anchor_base,
        fgdr_max_radius_factor=args.fgdr_max_radius_factor,
    )
    model.to(device)
    model_loss = cas_mvsnet_loss

    if args.sync_bn:
        import apex
        print("using apex synced BN")
        model = apex.parallel.convert_syncbn_model(model)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=args.lr, betas=(0.9, 0.999), weight_decay=args.wd)

    start_epoch = 0
    if args.loadckpt:
        print("loading model {}".format(args.loadckpt))
        state_dict = torch.load(args.loadckpt, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict['model'])
        if args.resume:
            optimizer.load_state_dict(state_dict['optimizer'])
            start_epoch = state_dict['epoch'] + 1
    elif args.resume:
        saved_models = [fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt")]
        saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        loadckpt = os.path.join(args.logdir, saved_models[-1])
        print("resuming", loadckpt)
        state_dict = torch.load(loadckpt, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
        start_epoch = state_dict['epoch'] + 1

    if (not is_distributed) or (dist.get_rank() == 0):
        print("start at epoch {}".format(start_epoch))
        print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    if args.using_apex:
        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level=args.opt_level,
                                          keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                                          loss_scale=args.loss_scale)

    if is_distributed:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank)
    else:
        if torch.cuda.is_available():
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)

    train_dataset = BlendedMVSDataset(args.trainpath, args.trainlist, "train", args.nviews, args.numdepth,
                                      args.interval_scale)
    test_dataset = BlendedMVSDataset(args.testpath, args.testlist, "val", args.nviews, args.numdepth,
                                     args.interval_scale)

    if is_distributed:
        train_sampler = torch.utils.data.DistributedSampler(train_dataset, num_replicas=dist.get_world_size(),
                                                            rank=dist.get_rank())
        test_sampler = torch.utils.data.DistributedSampler(test_dataset, num_replicas=dist.get_world_size(),
                                                           rank=dist.get_rank())
        TrainImgLoader = DataLoader(train_dataset, args.batch_size, sampler=train_sampler,
                                    num_workers=1, drop_last=True, pin_memory=args.pin_m)
        TestImgLoader = DataLoader(test_dataset, args.batch_size, sampler=test_sampler,
                                   num_workers=1, drop_last=False, pin_memory=args.pin_m)
    else:
        TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True,
                                    num_workers=1, drop_last=True, pin_memory=args.pin_m)
        TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False,
                                   num_workers=1, drop_last=False, pin_memory=args.pin_m)

    if args.mode == "train":
        train(model, model_loss, optimizer, TrainImgLoader, TestImgLoader, start_epoch, args)
    elif args.mode == "test":
        test(model, model_loss, TestImgLoader, args)
    elif args.mode == "profile":
        profile()
    else:
        raise NotImplementedError
