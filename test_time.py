import argparse
import gc
import os
import statistics
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from datasets import find_dataset_def
from datasets.data_io import save_pfm
from models import *
from utils import *


cudnn.benchmark = True

SHARED_ROOT = os.environ.get("R2MVSNET_SHARED_ROOT", "/root/shared-nvme")
DTU_TEST_ROOT = os.environ.get(
    "R2MVSNET_DTU_TEST_PATH",
    os.path.join(SHARED_ROOT, "datasets", "dtu_testing"),
)

parser = argparse.ArgumentParser(description="Measure R2-MVSNet inference time and memory")
parser.add_argument("--model", default="mvsnet", help="select model")

parser.add_argument("--dataset", default="general_eval", help="select dataset")
parser.add_argument("--testpath", default=DTU_TEST_ROOT, help="testing data dir")
parser.add_argument("--testpath_single_scene", help="testing data path for single scene")
parser.add_argument("--testlist", default="lists/dtu/test.txt", help="testing scene list")

parser.add_argument("--batch_size", type=int, default=1, help="testing batch size")
parser.add_argument("--numdepth", type=int, default=192, help="the number of depth values")

parser.add_argument("--loadckpt", default="checkpoints/debug/model_000013.ckpt", help="load a specific checkpoint")
parser.add_argument("--outdir", default="./outputs_time", help="output dir, only used with --save_outputs")
parser.add_argument("--save_outputs", action="store_true", help="save depth/confidence maps after timing")

parser.add_argument("--share_cr", action="store_true", help="whether share the cost volume regularization")
parser.add_argument("--ndepths", type=str, default="48,32,8", help="ndepths")
parser.add_argument("--depth_inter_r", type=str, default="4,2,1", help="depth_intervals_ratio")
parser.add_argument("--cr_base_chs", type=str, default="8,8,8", help="cost regularization base channels")
parser.add_argument("--grad_method", type=str, default="detach", choices=["detach", "undetach"], help="grad method")

parser.add_argument("--use_view_attention", action="store_true", help="enable SP-RWCV source-view reliability weighting")
parser.add_argument(
    "--view_attention_mode",
    type=str,
    default="single_pass_reliability_weighted",
    choices=["single_pass_reliability_weighted", "decoupled_reliability_weighted"],
    help="SP-RWCV weighting variant",
)
parser.add_argument("--use_rafe", action="store_true", help="enable reliability-aware feature extraction")
parser.add_argument("--use_fgdr", action="store_true", help="enable progressive fusion-guided depth refinement")
parser.add_argument(
    "--fgdr_max_radius_factor",
    type=float,
    default=2.0,
    help="maximum FGDR candidate radius in local depth intervals",
)
parser.add_argument("--fgdr_anchor_base", action="store_true", help="keep original R2 depth as cascade/fusion anchor")

parser.add_argument("--interval_scale", type=float, default=1.06, help="the depth interval scale")
parser.add_argument("--num_view", type=int, default=5, help="num of view")
parser.add_argument("--max_h", type=int, default=864, help="testing max h")
parser.add_argument("--max_w", type=int, default=1152, help="testing max w")
parser.add_argument("--fix_res", action="store_true", help="scene all using same res")

parser.add_argument("--num_worker", type=int, default=4, help="dataloader workers")
parser.add_argument("--warmup_batches", type=int, default=1, help="number of first batches excluded from averages")
parser.add_argument("--max_batches", type=int, default=0, help="maximum measured batches, 0 means all")


args = parser.parse_args()
print("argv:", sys.argv[1:])
print_args(args)
if args.testpath_single_scene:
    args.testpath = os.path.dirname(args.testpath_single_scene)

num_stage = len([int(nd) for nd in args.ndepths.split(",") if nd])
Interval_Scale = args.interval_scale
print("***********Interval_Scale**********\n", Interval_Scale)


def write_cam(file, cam):
    with open(file, "w") as f:
        f.write("extrinsic\n")
        for i in range(0, 4):
            for j in range(0, 4):
                f.write(str(cam[0][i][j]) + " ")
            f.write("\n")
        f.write("\n")

        f.write("intrinsic\n")
        for i in range(0, 3):
            for j in range(0, 3):
                f.write(str(cam[1][i][j]) + " ")
            f.write("\n")

        f.write(
            "\n"
            + str(cam[1][3][0])
            + " "
            + str(cam[1][3][1])
            + " "
            + str(cam[1][3][2])
            + " "
            + str(cam[1][3][3])
            + "\n"
        )


def build_model():
    model = CascadeMVSNet(
        refine=False,
        ndepths=[int(nd) for nd in args.ndepths.split(",") if nd],
        depth_interals_ratio=[float(d_i) for d_i in args.depth_inter_r.split(",") if d_i],
        share_cr=args.share_cr,
        cr_base_chs=[int(ch) for ch in args.cr_base_chs.split(",") if ch],
        grad_method=args.grad_method,
        use_view_attention=args.use_view_attention,
        view_attention_mode=args.view_attention_mode,
        use_rafe=args.use_rafe,
        use_fgdr=args.use_fgdr,
        fgdr_max_radius_factor=args.fgdr_max_radius_factor,
        fgdr_anchor_base=args.fgdr_anchor_base,
    )
    return model


def save_batch_outputs(sample, outputs):
    outputs_np = tensor2numpy(outputs)
    filenames = sample["filename"]
    cams = sample["proj_matrices"]["stage{}".format(num_stage)].numpy()
    imgs = sample["imgs"].numpy()

    for sample_idx, (filename, cam, img, depth_est, photometric_confidence) in enumerate(
        zip(filenames, cams, imgs, outputs_np["depth"], outputs_np["photometric_confidence"])
    ):
        img = img[0]
        cam = cam[0]
        depth_filename = os.path.join(args.outdir, filename.format("depth_est", ".pfm"))
        confidence_filename = os.path.join(args.outdir, filename.format("confidence", ".pfm"))
        cam_filename = os.path.join(args.outdir, filename.format("cams", "_cam.txt"))
        img_filename = os.path.join(args.outdir, filename.format("images", ".jpg"))
        os.makedirs(depth_filename.rsplit("/", 1)[0], exist_ok=True)
        os.makedirs(confidence_filename.rsplit("/", 1)[0], exist_ok=True)
        os.makedirs(cam_filename.rsplit("/", 1)[0], exist_ok=True)
        os.makedirs(img_filename.rsplit("/", 1)[0], exist_ok=True)

        save_pfm(depth_filename, depth_est)
        save_pfm(confidence_filename, photometric_confidence)
        write_cam(cam_filename, cam)
        img = np.clip(np.transpose(img, (1, 2, 0)) * 255, 0, 255).astype(np.uint8)
        cv2.imwrite(img_filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        if args.use_fgdr:
            fgdr_outputs = {
                "depth_candidate_main": outputs_np["fgdr_depth_main"][sample_idx],
                "depth_near": outputs_np["fgdr_depth_near"][sample_idx],
                "depth_far": outputs_np["fgdr_depth_far"][sample_idx],
                "geometry_gate": outputs_np["fgdr_geometry_gate"][sample_idx],
                "uncertainty": outputs_np["fgdr_uncertainty"][sample_idx],
                "depth_delta": outputs_np["fgdr_delta"][sample_idx],
                "depth_base": outputs_np["fgdr_depth_base"][sample_idx],
            }
            for folder, value in fgdr_outputs.items():
                fgdr_filename = os.path.join(args.outdir, filename.format(folder, ".pfm"))
                os.makedirs(fgdr_filename.rsplit("/", 1)[0], exist_ok=True)
                save_pfm(fgdr_filename, value)


def percentile(values, pct):
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * pct / 100.0))))
    return ordered[idx]


def measure_scene_depth(testlist):
    MVSDataset = find_dataset_def(args.dataset)
    test_dataset = MVSDataset(
        args.testpath,
        testlist,
        "test",
        args.num_view,
        args.numdepth,
        Interval_Scale,
        max_h=args.max_h,
        max_w=args.max_w,
        fix_res=args.fix_res,
    )
    test_loader = DataLoader(
        test_dataset,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_worker,
        drop_last=False,
    )

    model = build_model()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params / 1e6:.3f} M")
    print(f"Trainable params: {trainable_params / 1e6:.3f} M")

    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict["model"], strict=True)
    model = nn.DataParallel(model)
    model.cuda()
    model.eval()

    measured_times = []
    all_times = []
    sample_count = 0
    max_batches = len(test_loader) if args.max_batches <= 0 else min(args.max_batches, len(test_loader))

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print("Start inference timing...")
    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):
            if batch_idx >= max_batches:
                break

            sample_cuda = tocuda(sample)
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time

            batch_size_actual = sample_cuda["imgs"].shape[0]
            sample_count += batch_size_actual
            all_times.append(elapsed)
            if batch_idx >= args.warmup_batches:
                measured_times.append(elapsed)

            imgs = sample["imgs"].numpy()
            warmup_tag = "warmup" if batch_idx < args.warmup_batches else "measured"
            print(
                "Iter {}/{}, Time:{:.4f}s, PerSample:{:.4f}s, Res:{}, {}".format(
                    batch_idx + 1,
                    max_batches,
                    elapsed,
                    elapsed / max(batch_size_actual, 1),
                    imgs[0].shape,
                    warmup_tag,
                )
            )

            if args.save_outputs:
                save_batch_outputs(sample, outputs)

            del sample_cuda, outputs

    if measured_times:
        avg_batch_time = sum(measured_times) / len(measured_times)
        median_batch_time = statistics.median(measured_times)
        p90_batch_time = percentile(measured_times, 90)
        avg_sample_time = avg_batch_time / max(args.batch_size, 1)
        fps = max(args.batch_size, 1) / avg_batch_time if avg_batch_time > 0 else 0.0
    elif all_times:
        avg_batch_time = sum(all_times) / len(all_times)
        median_batch_time = statistics.median(all_times)
        p90_batch_time = percentile(all_times, 90)
        avg_sample_time = avg_batch_time / max(args.batch_size, 1)
        fps = max(args.batch_size, 1) / avg_batch_time if avg_batch_time > 0 else 0.0
    else:
        avg_batch_time = median_batch_time = p90_batch_time = avg_sample_time = fps = 0.0

    peak_allocated_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    peak_reserved_mb = torch.cuda.max_memory_reserved() / (1024 * 1024)

    scene_name = testlist[0] if len(testlist) == 1 else f"{len(testlist)} scenes"
    print("\n--- Inference timing summary ---")
    print(f"Scenes: {scene_name}")
    print(
        "Modules: RAFE={}, SP-RWCV={}, FGDR={}, Anchor={}".format(
            args.use_rafe, args.use_view_attention, args.use_fgdr, args.fgdr_anchor_base
        )
    )
    print(f"Batch size: {args.batch_size}")
    print(f"Warmup batches skipped: {min(args.warmup_batches, len(all_times))}")
    print(f"Measured batches: {len(measured_times) if measured_times else len(all_times)}")
    print(f"Processed samples: {sample_count}")
    print(f"Avg batch inference time: {avg_batch_time:.4f} s")
    print(f"Median batch inference time: {median_batch_time:.4f} s")
    print(f"P90 batch inference time: {p90_batch_time:.4f} s")
    print(f"Avg per-sample inference time: {avg_sample_time:.4f} s")
    print(f"Throughput: {fps:.3f} samples/s")
    print(f"Peak GPU allocated: {peak_allocated_mb:.2f} MB")
    print(f"Peak GPU reserved: {peak_reserved_mb:.2f} MB")
    print(f"Total params: {total_params / 1e6:.3f} M")
    print("--------------------------------\n")

    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    if args.testlist != "all":
        with open(args.testlist) as f:
            testlist = [line.rstrip() for line in f.readlines()]
    else:
        testlist = (
            [e for e in os.listdir(args.testpath) if os.path.isdir(os.path.join(args.testpath, e))]
            if not args.testpath_single_scene
            else [os.path.basename(args.testpath_single_scene)]
        )

    measure_scene_depth(testlist)
