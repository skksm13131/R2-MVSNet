import argparse, os, time, sys, gc, cv2
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import numpy as np
from datasets import find_dataset_def
from models import *
from utils import *
from datasets.data_io import read_pfm, save_pfm
from plyfile import PlyData, PlyElement
from PIL import Image
from multiprocessing import Pool
from functools import partial
import signal

cudnn.benchmark = True

SHARED_ROOT = os.environ.get("R2MVSNET_SHARED_ROOT", "/root/shared-nvme")
TNT_ROOT = os.environ.get("R2MVSNET_TNT_PATH", os.path.join(SHARED_ROOT, "datasets", "TankandTemples"))

parser = argparse.ArgumentParser(
    description='Predict depth for Tanks and Temples Advanced set')

parser.add_argument('--model', default='mvsnet', help='select model')
parser.add_argument('--dataset', default='tanks', help='select dataset')
parser.add_argument('--testpath', default=TNT_ROOT, help='testing data dir')
parser.add_argument('--testlist', default='lists/tnt/advanced.txt', help='testing scene list')
parser.add_argument('--outdir', default='./outputs_tnt', help='output dir')
parser.add_argument('--split', type=str, default='advanced', help='intermediate or advanced')
parser.add_argument('--max_test_batches', type=int, default=0,
                    help='debug only: stop after this many batches when > 0')
parser.add_argument('--loadckpt', default='checkpoints/model_000000.ckpt', help='load a specific checkpoint')
parser.add_argument('--batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--num_view', type=int, default=11, help='num of view')
parser.add_argument('--max_h', type=int, default=1080, help='testing max h')
parser.add_argument('--max_w', type=int, default=1920, help='testing max w')
parser.add_argument('--numdepth', type=int, default=192, help='the number of depth values')
parser.add_argument('--ndepths', type=str, default="96,32,8", help='ndepths for cascade')
parser.add_argument('--depth_inter_r', type=str, default="2,2,1", help='depth_intervals_ratio for cascade')
parser.add_argument('--cr_base_chs', type=str, default="8,8,8", help='cost regularization base channels')
parser.add_argument('--grad_method', type=str, default="detach", choices=["detach", "undetach"], help='grad method')
parser.add_argument('--share_cr', action='store_true', help='whether share the cost volume regularization')

parser.add_argument('--use_view_attention', action='store_true',
                    help='enable SP-RWCV source-view reliability weighting')
parser.add_argument('--view_attention_mode', type=str, default='single_pass_reliability_weighted',
                    choices=['single_pass_reliability_weighted'],
                    help='SP-RWCV weighting variant')
parser.add_argument('--use_rafe', action='store_true', help='enable reliability-aware feature extraction')
parser.add_argument('--use_fgdr', action='store_true', help='enable fusion-guided depth refinement')
parser.add_argument('--fgdr_max_radius_factor', type=float, default=2.0,
                    help='maximum FGDR candidate radius in local depth intervals')
parser.add_argument('--fgdr_anchor_base', action='store_true',
                    help='keep original R2 depth as the cascade/fusion anchor')

args = parser.parse_args()
print("argv:", sys.argv[1:])
print_args(args)
num_stage = len([int(nd) for nd in args.ndepths.split(",") if nd])


def write_cam(file, cam):
    f = open(file, "w")
    f.write('extrinsic\n')
    for i in range(4):
        for j in range(4):
            f.write(str(cam[0][i][j]) + ' ')
        f.write('\n')
    f.write('\n')
    f.write('intrinsic\n')
    for i in range(3):
        for j in range(3):
            f.write(str(cam[1][i][j]) + ' ')
        f.write('\n')
    f.write(
        '\n' + str(cam[1][3][0]) + ' ' + str(cam[1][3][1]) + ' ' + str(cam[1][3][2]) + ' ' + str(cam[1][3][3]) + '\n')
    f.close()


def save_scene_depth(testlist):
    MVSDataset = find_dataset_def(args.dataset)
    test_dataset = MVSDataset(datapath=args.testpath,
                              listfile=testlist,
                              split=args.split,
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
        use_rafe=args.use_rafe,
        use_view_attention=args.use_view_attention,
        view_attention_mode=args.view_attention_mode,
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
            sample_cuda = tocuda(sample)
            start_time = time.time()
            outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])
            end_time = time.time()
            outputs = tensor2numpy(outputs)
            del sample_cuda
            filenames = sample["filename"]
            cams = sample["proj_matrices"][f"stage{num_stage}"].numpy()
            imgs = sample["imgs"].numpy()
            print(f'Iter {batch_idx}/{len(TestImgLoader)}, Time:{end_time - start_time:.4f} Res:{imgs[0].shape}')

            for filename, cam, img, depth_est, photometric_confidence in zip(
                    filenames, cams, imgs,
                    outputs["depth"],
                    outputs["photometric_confidence"]):
                img = img[0]
                cam = cam[0]
                depth_filename = os.path.join(args.outdir, filename.format('depth_est', '.pfm'))
                confidence_filename = os.path.join(args.outdir, filename.format('confidence', '.pfm'))
                cam_filename = os.path.join(args.outdir, filename.format('cams', '_cam.txt'))
                img_filename = os.path.join(args.outdir, filename.format('images', '.jpg'))
                os.makedirs(os.path.dirname(depth_filename), exist_ok=True)
                os.makedirs(os.path.dirname(confidence_filename), exist_ok=True)
                os.makedirs(os.path.dirname(cam_filename), exist_ok=True)
                os.makedirs(os.path.dirname(img_filename), exist_ok=True)
                save_pfm(depth_filename, depth_est)
                save_pfm(confidence_filename, photometric_confidence)
                write_cam(cam_filename, cam)
                img = np.clip(np.transpose(img, (1, 2, 0)) * 255, 0, 255).astype(np.uint8)
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(img_filename, img_bgr)

                if args.use_fgdr:
                    fgdr_outputs = {
                        'depth_candidate_main': outputs.get('fgdr_depth_main', [None])[0],
                        'depth_near': outputs.get('fgdr_depth_near', [None])[0],
                        'depth_far': outputs.get('fgdr_depth_far', [None])[0],
                        'geometry_gate': outputs.get('fgdr_geometry_gate', [None])[0],
                        'uncertainty': outputs.get('fgdr_uncertainty', [None])[0],
                        'depth_delta': outputs.get('fgdr_delta', [None])[0],
                        'depth_base': outputs.get('fgdr_depth_base', [None])[0],
                    }
                    for folder, value in fgdr_outputs.items():
                        fgdr_filename = os.path.join(args.outdir, filename.format(folder, '.pfm'))
                        os.makedirs(os.path.dirname(fgdr_filename), exist_ok=True)
                        save_pfm(fgdr_filename, value)

            if args.max_test_batches > 0 and (batch_idx + 1) >= args.max_test_batches:
                break

    torch.cuda.empty_cache()
    gc.collect()


if __name__ == '__main__':
    with open(args.testlist) as f:
        testlist = [line.strip() for line in f.readlines() if line.strip()]

    print("Step 1: Generating depth and confidence maps...")
    save_scene_depth(testlist)
