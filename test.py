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
parser.add_argument('--use_eta', action='store_true', help='enable ETA visibility weighting')
parser.add_argument('--use_view_attention', action='store_true', help='enable warped-view attention in DepthNet')
parser.add_argument('--view_attention_mode', type=str, default='legacy', choices=['legacy', 'residual_fusion', 'progressive_residual_fusion', 'single_pass_reliability_weighted'], help='view attention mode')
parser.add_argument('--use_normal_visibility', action='store_true', help='enable normal-derived visibility prior')
parser.add_argument('--visibility_weight', type=float, default=0.3, help='normal visibility prior weight')
parser.add_argument('--visibility_stage_start', type=int, default=2, help='stage to start normal visibility')
parser.add_argument('--visibility_residual_only', action='store_true', help='apply visibility prior only to residual branch')
parser.add_argument('--use_normal_head', action='store_true', help='enable normal prediction heads')
parser.add_argument('--normal_loss_weight', type=float, default=0.0, help='normal loss weight, kept for checkpoint-compatible construction')
parser.add_argument('--use_geometry_guidance', action='store_true', help='enable geometry-guided cost regularization')
parser.add_argument('--geometry_guidance_weight', type=float, default=0.1, help='geometry guidance weight')
parser.add_argument('--geometry_stage_start', type=int, default=2, help='stage to start geometry guidance')
parser.add_argument('--normal_prior_stage_start', type=int, default=3, help='stage to start normal prior blending')
parser.add_argument('--normal_prior_blend_weight', type=float, default=0.35, help='normal prior blend weight')
parser.add_argument('--use_normal_consistency_gate', action='store_true', help='enable normal consistency gating')
parser.add_argument('--normal_consistency_threshold', type=float, default=0.3, help='normal consistency threshold')
parser.add_argument('--normal_consistency_power', type=float, default=2.0, help='normal consistency power')
parser.add_argument('--use_reliability_guidance', action='store_true', help='enable reliability-guided interval/prior weighting')
parser.add_argument('--reliability_stage_start', type=int, default=2, help='stage to start reliability guidance')
parser.add_argument('--reliability_interval_scale', type=float, default=0.5, help='reliability interval scale')
parser.add_argument('--reliability_confidence_mix', type=float, default=0.5, help='photometric/texture confidence mix')
parser.add_argument('--reliability_min_interval_factor', type=float, default=0.5, help='minimum interval factor under reliability guidance')
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
parser.add_argument('--use_ugdr', action='store_true', help='enable UGDR final-stage bounded depth refinement')
parser.add_argument('--ugdr_max_residual_ratio', type=float, default=0.5, help='UGDR maximum residual as a ratio of final-stage interval')
parser.add_argument('--use_sparse_feature_attention', action='store_true', help='enable sparse feature attention')
parser.add_argument('--feature_attention_type', type=str, default='eca', help='sparse feature attention type')
parser.add_argument('--use_gcp', action='store_true', help='enable GCP cost regularization')
parser.add_argument('--use_tpfe', action='store_true', help='enable TPFE feature extraction')
parser.add_argument('--use_dndr', action='store_true', help='enable DNDR regularization')
parser.add_argument('--dndr_weight', type=float, default=0.1, help='DNDR loss/regularization weight')

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
                          view_attention_mode=args.view_attention_mode,
                          use_rafe=args.use_rafe)

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
            for filename, cam, img, depth_est, photometric_confidence in zip(filenames, cams, imgs,
                                                                             outputs["depth"],
                                                                             outputs["photometric_confidence"]):
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
