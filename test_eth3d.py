import argparse, os, time, sys, gc, cv2
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import numpy as np

from datasets.eth3d import ETH3DDataset
from models import CascadeMVSNet
from utils import *
from datasets.data_io_e3 import save_pfm, read_cam_file

cudnn.benchmark = True

SHARED_ROOT = os.environ.get("R2MVSNET_SHARED_ROOT", "/root/shared-nvme")
ETH3D_ROOT = os.environ.get("R2MVSNET_ETH3D_PATH", os.path.join(SHARED_ROOT, "datasets", "eth3d_high_res_test"))

parser = argparse.ArgumentParser(description='Predict depth and confidence maps for ETH3D dataset')

parser.add_argument('--dataset', default='eth3d', help='select dataset', choices=['eth3d'])
parser.add_argument('--testpath', default=ETH3D_ROOT, help='testing data dir for ETH3D')
parser.add_argument('--testlist', default='lists/eth3d/test.txt', help='testing scene list for ETH3D')
parser.add_argument('--outdir', default='./outputs_eth3d', help='output dir for ETH3D results')
parser.add_argument('--loadckpt', default='checkpoints/model_000000.ckpt', help='load a specific checkpoint')
parser.add_argument('--batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--num_view', type=int, default=7, help='num of views')
parser.add_argument('--max_h', type=int, default=1792, help='testing max h')
parser.add_argument('--max_w', type=int, default=2688, help='testing max w')
parser.add_argument('--numdepth', type=int, default=192, help='number of depth values')
parser.add_argument('--ndepths', type=str, default="96,32,8", help='ndepths')
parser.add_argument('--depth_inter_r', type=str, default="2,2,1", help='depth_intervals_ratio')
parser.add_argument('--cr_base_chs', type=str, default="8,8,8", help='cost regularization base channels')
parser.add_argument('--grad_method', type=str, default="detach", choices=["detach", "undetach"])
parser.add_argument('--refine', action='store_true', help='use refinement module')
parser.add_argument('--share_cr', action='store_true', help='share cost volume regularization')

# R2 modules
parser.add_argument('--use_view_attention', action='store_true',
                    help='enable SP-RWCV source-view reliability weighting')
parser.add_argument('--use_rafe', action='store_true', help='enable reliability-aware feature extraction')
parser.add_argument('--use_fgdr', action='store_true', help='enable fusion-guided depth refinement')
parser.add_argument('--fgdr_max_radius_factor', type=float, default=2.0)
parser.add_argument('--fgdr_anchor_base', action='store_true',
                    help='keep original R2 depth as the cascade/fusion anchor')

args = parser.parse_args()
print("argv:", sys.argv[1:])
print_args(args)

num_stage = len([int(nd) for nd in args.ndepths.split(",") if nd])


def write_cam_eth3d_format(file, intrinsics, extrinsics, depth_min, depth_max):
    try:
        with open(file, "w") as f:
            f.write('extrinsic\n')
            for i in range(4):
                f.write(' '.join(map(str, extrinsics[i])) + '\n')
            f.write('\n')
            f.write('intrinsic\n')
            for i in range(3):
                f.write(' '.join(map(str, intrinsics[i])) + '\n')
            f.write('\n')
            f.write(f'{float(depth_min)} {float(depth_max)}\n')
    except Exception as e:
        print(f"Error writing camera file {file}: {e}")


def save_scene_depth(testlist):
    test_dataset = ETH3DDataset(datapath=args.testpath,
                                listfile=args.testlist,
                                nviews=args.num_view,
                                ndepths=args.numdepth,
                                img_wh=(args.max_w, args.max_h))
    TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=4, drop_last=False)

    model = CascadeMVSNet(
        ndepths=[int(nd) for nd in args.ndepths.split(",") if nd],
        depth_interals_ratio=[float(d_i) for d_i in args.depth_inter_r.split(",") if d_i],
        share_cr=args.share_cr,
        cr_base_chs=[int(ch) for ch in args.cr_base_chs.split(",") if ch],
        grad_method=args.grad_method,
        refine=args.refine,
        use_rafe=args.use_rafe,
        use_view_attention=args.use_view_attention,
        use_fgdr=args.use_fgdr,
        fgdr_anchor_base=args.fgdr_anchor_base,
        fgdr_max_radius_factor=args.fgdr_max_radius_factor,
    )

    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict['model'], strict=True)
    model = nn.DataParallel(model)
    model.cuda()
    model.eval()

    with torch.no_grad():
        for batch_idx, sample in enumerate(TestImgLoader):
            if sample is None:
                print(f"Skipping batch {batch_idx} due to loading error.")
                continue
            sample_cuda = tocuda(sample)
            start_time = time.time()
            outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])
            end_time = time.time()
            outputs = tensor2numpy(outputs)
            imgs = sample["imgs"].numpy()
            intrinsics_scaled = sample["intrinsics_scaled"].numpy()
            extrinsics = sample["extrinsics"].numpy()
            depth_min = sample["depth_min"].numpy()
            depth_max = sample["depth_max"].numpy()
            del sample_cuda
            filenames = sample["filename"]
            print(f'Iter {batch_idx + 1}/{len(TestImgLoader)}, Time:{end_time - start_time:.4f} Res:{imgs[0, 0].shape}')

            for i in range(len(filenames)):
                filename = filenames[i]
                depth = outputs["depth"][i]
                confidence = outputs["photometric_confidence"][i]
                img_ref = imgs[i, 0]
                intr_ref = intrinsics_scaled[i]
                extr_ref = extrinsics[i]
                dmin_ref = depth_min[i]
                dmax_ref = depth_max[i]

                depth_filename = os.path.join(args.outdir, filename.format('depth_est', '.pfm'))
                confidence_filename = os.path.join(args.outdir, filename.format('confidence', '.pfm'))
                cam_filename = os.path.join(args.outdir, filename.format('cams', '_cam.txt'))
                img_filename = os.path.join(args.outdir, filename.format('images', '.jpg'))
                os.makedirs(os.path.dirname(depth_filename), exist_ok=True)
                os.makedirs(os.path.dirname(confidence_filename), exist_ok=True)
                os.makedirs(os.path.dirname(cam_filename), exist_ok=True)
                os.makedirs(os.path.dirname(img_filename), exist_ok=True)
                save_pfm(depth_filename, depth.squeeze())
                save_pfm(confidence_filename, confidence.squeeze())
                write_cam_eth3d_format(cam_filename, intr_ref, extr_ref, dmin_ref, dmax_ref)
                img_np = (img_ref.transpose(1, 2, 0) * 255).astype(np.uint8)
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                cv2.imwrite(img_filename, img_bgr)

                if args.use_fgdr:
                    fgdr_outputs = {
                        'depth_candidate_main': outputs.get('fgdr_depth_main', [None])[i],
                        'depth_near': outputs.get('fgdr_depth_near', [None])[i],
                        'depth_far': outputs.get('fgdr_depth_far', [None])[i],
                        'geometry_gate': outputs.get('fgdr_geometry_gate', [None])[i],
                        'uncertainty': outputs.get('fgdr_uncertainty', [None])[i],
                        'depth_delta': outputs.get('fgdr_delta', [None])[i],
                        'depth_base': outputs.get('fgdr_depth_base', [None])[i],
                    }
                    for folder, value in fgdr_outputs.items():
                        fgdr_filename = os.path.join(args.outdir, filename.format(folder, '.pfm'))
                        os.makedirs(os.path.dirname(fgdr_filename), exist_ok=True)
                        save_pfm(fgdr_filename, value)

    torch.cuda.empty_cache()
    gc.collect()


if __name__ == '__main__':
    if not os.path.exists(args.testlist):
        print(f"Error: Test list file not found at {args.testlist}")
        sys.exit(1)

    if not os.path.exists(args.testpath):
        print(f"Error: Test data path not found at {args.testpath}")
        sys.exit(1)

    if not os.path.exists(args.loadckpt):
        print(f"Error: Model checkpoint not found at {args.loadckpt}")
        sys.exit(1)

    with open(args.testlist) as f:
        testlist = [line.strip() for line in f.readlines() if line.strip()]

    print("Step 1: Generating depth and confidence maps for ETH3D...")
    save_scene_depth(testlist)
    print(f"Results saved in: {args.outdir}")
