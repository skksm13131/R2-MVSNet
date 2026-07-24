import argparse, os, time, sys, gc, cv2
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from datasets import find_dataset_def
from models import *
from utils import *
from datasets.data_io import read_pfm, save_pfm
from plyfile import PlyData, PlyElement
from PIL import Image
from gipuma import gipuma_filter
# NOTE 2026-06-04: this file comes from the newer workflow, but this rebuild uses
# true-baseline models. Extra experimental model kwargs were intentionally
# filtered from the CascadeMVSNet constructor for compatibility.


from multiprocessing import Pool
from functools import partial
import signal

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Predict depth, filter, and fuse')
parser.add_argument('--model', default='mvsnet', help='select model')

parser.add_argument('--dataset', default='general_eval', help='select dataset')
parser.add_argument('--testpath', default='/home/u104754251515/data/dtu_testing', help='testing data dir for some scenes')
parser.add_argument('--testpath_single_scene', help='testing data path for single scene')
parser.add_argument('--testlist', default='lists/dtu/test.txt', help='testing scene list')

parser.add_argument('--batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--numdepth', type=int, default=192, help='the number of depth values')

parser.add_argument('--loadckpt', default='checkpoints/debug/model_000013.ckpt', help='load a specific checkpoint')
parser.add_argument('--outdir', default='./outputs', help='output dir')
parser.add_argument('--display', action='store_true', help='display depth images and masks')

parser.add_argument('--share_cr', action='store_true', help='whether share the cost volume regularization')

parser.add_argument('--ndepths', type=str, default="48,32,8", help='ndepths')
parser.add_argument('--depth_inter_r', type=str, default="4,2,1", help='depth_intervals_ratio')
parser.add_argument('--cr_base_chs', type=str, default="8,8,8", help='cost regularization base channels')
parser.add_argument('--grad_method', type=str, default="detach", choices=["detach", "undetach"], help='grad method')
parser.add_argument('--use_view_attention', action='store_true', help='enable SP-RWCV reliability-weighted cost volume in DepthNet')
parser.add_argument('--use_rafe', action='store_true', help='enable reliability-aware feature extraction')
parser.add_argument('--use_adaptive_r2', action='store_true', help='enable difficulty-adaptive RAFE and SP-RWCV gating')
parser.add_argument('--use_fgdr', action='store_true', help='enable progressive fusion-guided depth refinement')
parser.add_argument('--fgdr_max_radius_factor', type=float, default=2.0, help='maximum FGDR candidate radius in local depth intervals')
parser.add_argument('--fgdr_anchor_base', action='store_true', help='keep original R2 depth as the cascade/fusion anchor')

parser.add_argument('--interval_scale', type=float, default=1.06, help='the depth interval scale')
parser.add_argument('--num_view', type=int, default=5, help='num of view')
parser.add_argument('--max_h', type=int, default=864, help='testing max h')
parser.add_argument('--max_w', type=int, default=1152, help='testing max w')
parser.add_argument('--fix_res', action='store_true', help='scene all using same res')

parser.add_argument('--num_worker', type=int, default=4, help='depth_filer worker')
parser.add_argument('--save_freq', type=int, default=20, help='save freq of local pcd')

parser.add_argument('--filter_method', type=str, default='gipuma', choices=["gipuma", "normal"], help="filter method")


# parse arguments and check
args = parser.parse_args()
print("argv:", sys.argv[1:])
print_args(args)
if args.testpath_single_scene:
    args.testpath = os.path.dirname(args.testpath_single_scene)

num_stage = len([int(nd) for nd in args.ndepths.split(",") if nd])

Interval_Scale = args.interval_scale
print("***********Interval_Scale**********\n", Interval_Scale)


def write_cam(file, cam):
    f = open(file, "w")
    f.write('extrinsic\n')
    for i in range(0, 4):
        for j in range(0, 4):
            f.write(str(cam[0][i][j]) + ' ')
        f.write('\n')
    f.write('\n')

    f.write('intrinsic\n')
    for i in range(0, 3):
        for j in range(0, 3):
            f.write(str(cam[1][i][j]) + ' ')
        f.write('\n')

    f.write(
        '\n' + str(cam[1][3][0]) + ' ' + str(cam[1][3][1]) + ' ' + str(cam[1][3][2]) + ' ' + str(cam[1][3][3]) + '\n')

    f.close()


def save_depth(testlist):
    for scene in testlist:
        save_scene_depth([scene])


# run CasMVS model to save depth maps and confidence maps
def save_scene_depth(testlist):
    # dataset, dataloader
    MVSDataset = find_dataset_def(args.dataset)
    test_dataset = MVSDataset(args.testpath, testlist, "test", args.num_view, args.numdepth, Interval_Scale,
                              max_h=args.max_h, max_w=args.max_w, fix_res=args.fix_res)
    TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=4, drop_last=False)

    # model
    model = CascadeMVSNet(refine=False,
                          ndepths=[int(nd) for nd in args.ndepths.split(",") if nd],
                          depth_interals_ratio=[float(d_i) for d_i in args.depth_inter_r.split(",") if d_i],
                          share_cr=args.share_cr,
                          cr_base_chs=[int(ch) for ch in args.cr_base_chs.split(",") if ch],
                          grad_method=args.grad_method,
                          use_view_attention=args.use_view_attention,
                          use_rafe=args.use_rafe,
                          use_adaptive_r2=args.use_adaptive_r2,
                          use_fgdr=args.use_fgdr,
                          fgdr_max_radius_factor=args.fgdr_max_radius_factor,
                          fgdr_anchor_base=args.fgdr_anchor_base)

    # load checkpoint file specified by args.loadckpt
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
            cams = sample["proj_matrices"]["stage{}".format(num_stage)].numpy()
            imgs = sample["imgs"].numpy()
            print('Iter {}/{}, Time:{} Res:{}'.format(batch_idx, len(TestImgLoader), end_time - start_time,
                                                      imgs[0].shape))

            # save depth maps and confidence maps
            for sample_idx, (filename, cam, img, depth_est, photometric_confidence) in enumerate(
                    zip(filenames, cams, imgs, outputs["depth"], outputs["photometric_confidence"])):
                img = img[0]  # ref view
                cam = cam[0]  # ref cam
                depth_filename = os.path.join(args.outdir, filename.format('depth_est', '.pfm'))
                confidence_filename = os.path.join(args.outdir, filename.format('confidence', '.pfm'))
                cam_filename = os.path.join(args.outdir, filename.format('cams', '_cam.txt'))
                img_filename = os.path.join(args.outdir, filename.format('images', '.jpg'))
                ply_filename = os.path.join(args.outdir, filename.format('ply_local', '.ply'))
                os.makedirs(depth_filename.rsplit('/', 1)[0], exist_ok=True)
                os.makedirs(confidence_filename.rsplit('/', 1)[0], exist_ok=True)
                os.makedirs(cam_filename.rsplit('/', 1)[0], exist_ok=True)
                os.makedirs(img_filename.rsplit('/', 1)[0], exist_ok=True)
                os.makedirs(ply_filename.rsplit('/', 1)[0], exist_ok=True)
                # save depth maps
                save_pfm(depth_filename, depth_est)
                # save confidence maps
                save_pfm(confidence_filename, photometric_confidence)
                # save cams, img
                write_cam(cam_filename, cam)
                img = np.clip(np.transpose(img, (1, 2, 0)) * 255, 0, 255).astype(np.uint8)
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(img_filename, img_bgr)

                if args.use_fgdr:
                    fgdr_outputs = {
                        'depth_candidate_main': outputs['fgdr_depth_main'][sample_idx],
                        'depth_near': outputs['fgdr_depth_near'][sample_idx],
                        'depth_far': outputs['fgdr_depth_far'][sample_idx],
                        'geometry_gate': outputs['fgdr_geometry_gate'][sample_idx],
                        'uncertainty': outputs['fgdr_uncertainty'][sample_idx],
                        'depth_delta': outputs['fgdr_delta'][sample_idx],
                        'depth_base': outputs['fgdr_depth_base'][sample_idx],
                    }
                    for folder, value in fgdr_outputs.items():
                        fgdr_filename = os.path.join(args.outdir, filename.format(folder, '.pfm'))
                        os.makedirs(fgdr_filename.rsplit('/', 1)[0], exist_ok=True)
                        save_pfm(fgdr_filename, value)

                # vis disabled for headless server

    torch.cuda.empty_cache()
    gc.collect()


if __name__ == '__main__':

    if args.testlist != "all":
        with open(args.testlist) as f:
            content = f.readlines()
            testlist = [line.rstrip() for line in content]
    else:
        # for tanks & temples or eth3d or colmap
        testlist = [e for e in os.listdir(args.testpath) if os.path.isdir(os.path.join(args.testpath, e))] \
            if not args.testpath_single_scene else [os.path.basename(args.testpath_single_scene)]

    # step1. save all the depth maps and the masks in outputs directory
    save_depth(testlist)
