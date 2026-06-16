import argparse
import math
import os
import sys
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from functools import partial
from PIL import Image
from datasets.data_io import read_pfm, save_pfm
import signal

parser = argparse.ArgumentParser(
    description='Filter depth maps and fuse point cloud with NORMAL method, independently.')
parser.add_argument('--conf', type=float, default=0.8, help='Photometric confidence threshold.')
parser.add_argument('--conf_stage', type=float, default=0.99,
                    help='High confidence threshold to keep original depth without averaging.')
parser.add_argument('--s_view', type=int, default=3, help='Start of consistent view count for dynamic check.')
parser.add_argument('--e_view', type=int, default=11, help='End of consistent view count for dynamic check.')
parser.add_argument('--dist_base', type=float, default=0.25,
                    help='Base unit for pixel distance threshold. Final threshold is i * dist_base.')
parser.add_argument('--diff_base', type=float, default=0.001,
                    help='Base unit for depth difference threshold. Final threshold is log(i) * diff_base.')

parser.add_argument('--outdir', default='./outputs',
                    help='Directory where scan folders with depth/confidence are located.')
parser.add_argument('--testpath', default='/home/u104754251515/data/dtu_testing',
                    help='Original testing data dir (for camera and pair files).')
parser.add_argument('--testpath_single_scene', help='testing data path for single scene')
parser.add_argument('--testlist', default='lists/dtu/test.txt', help='List of scans to process.')
parser.add_argument('--num_worker', type=int, default=4,
                    help='Number of workers for scenes multiprocessing (Watch out for CUDA OOM!).')
parser.add_argument('--ndepths', type=str, default="32,16,8,8", help='ndepths')
parser.add_argument('--filter_method', type=str, default='normal', choices=["gipuma", "normal"], help="filter method")
parser.add_argument('--display', action='store_true', help='display depth images and masks')

args = parser.parse_args()
if args.testpath_single_scene:
    args.testpath = os.path.dirname(args.testpath_single_scene)
num_stage = len([int(nd) for nd in args.ndepths.split(",") if nd])


def save_mask(filename, mask):
    assert mask.dtype == np.bool_ or mask.dtype == bool
    mask = mask.astype(np.uint8) * 255
    Image.fromarray(mask).save(filename)


def read_camera_parameters(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
    return intrinsics, extrinsics


def read_img(filename):
    img = Image.open(filename)
    np_img = np.array(img, dtype=np.float32) / 255.
    return np_img


def read_pair_file(filename):
    data = []
    with open(filename) as f:
        num_viewpoint = int(f.readline())
        for view_idx in range(num_viewpoint):
            ref_view = int(f.readline().rstrip())
            src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
            if len(src_views) > 0:
                data.append((ref_view, src_views))
    return data


# 【终极优化：原生极速二进制 PLY 保存器】
def save_ply_fast(filename, vertexs, colors):
    if len(vertexs) == 0:
        return
    # 构建适合 PLY 二进制规范的结构化 NumPy 数组
    vertex_all = np.empty(len(vertexs), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                               ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    vertex_all['x'] = vertexs[:, 0]
    vertex_all['y'] = vertexs[:, 1]
    vertex_all['z'] = vertexs[:, 2]
    vertex_all['red'] = colors[:, 0]
    vertex_all['green'] = colors[:, 1]
    vertex_all['blue'] = colors[:, 2]

    # 按照标准生成文本 Header
    header = f"ply\nformat binary_little_endian 1.0\nelement vertex {len(vertex_all)}\n"
    header += "property float x\nproperty float y\nproperty float z\n"
    header += "property uchar red\nproperty uchar green\nproperty uchar blue\n"
    header += "end_header\n"

    # 以二进制模式一次性将 Header 和内存数据冲刷到硬盘，速度极快
    with open(filename, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(vertex_all.tobytes())


# 【核心引擎】构建完全基于张量广播的极速 GPU 并行重投影引擎
def batch_reproject_gpu(depth_ref, intrinsics_ref, extrinsics_ref, depths_src, intrinsics_src, extrinsics_src, device):
    N, H, W = depths_src.shape

    y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    x_ref = x.flatten().unsqueeze(0).expand(N, -1).float()
    y_ref = y.flatten().unsqueeze(0).expand(N, -1).float()
    ones = torch.ones_like(x_ref)

    xy1_ref = torch.stack([x_ref, y_ref, ones], dim=1)
    depth_ref_flat = depth_ref.view(1, 1, -1).expand(N, -1, -1)

    inv_K_ref = torch.inverse(intrinsics_ref).unsqueeze(0).expand(N, -1, -1)
    xyz_ref = torch.bmm(inv_K_ref, xy1_ref) * depth_ref_flat
    xyz_ref_4 = torch.cat([xyz_ref, ones.unsqueeze(1)], dim=1)

    P_src = extrinsics_src @ torch.inverse(extrinsics_ref).unsqueeze(0)
    xyz_src = torch.bmm(P_src, xyz_ref_4)[:, :3, :]

    K_xyz_src = torch.bmm(intrinsics_src, xyz_src)
    xy_src = K_xyz_src[:, :2, :] / (K_xyz_src[:, 2:3, :] + 1e-6)

    # 转换为 grid_sample 接受的 [-1, 1] 区间
    grid_x = (xy_src[:, 0, :] / (W - 1)) * 2.0 - 1.0
    grid_y = (xy_src[:, 1, :] / (H - 1)) * 2.0 - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1).view(N, H, W, 2)

    sampled_depth_src = F.grid_sample(depths_src.unsqueeze(1), grid, mode='bilinear', padding_mode='zeros',
                                      align_corners=True).squeeze(1)

    sampled_depth_flat = sampled_depth_src.view(N, 1, -1)
    xyz_src_sampled = torch.bmm(torch.inverse(intrinsics_src),
                                torch.stack([xy_src[:, 0, :], xy_src[:, 1, :], ones], dim=1)) * sampled_depth_flat
    xyz_src_sampled_4 = torch.cat([xyz_src_sampled, ones.unsqueeze(1)], dim=1)

    P_reproj = extrinsics_ref.unsqueeze(0) @ torch.inverse(extrinsics_src)
    xyz_reproj = torch.bmm(P_reproj, xyz_src_sampled_4)[:, :3, :]

    depth_reproj = xyz_reproj[:, 2, :].view(N, H, W)
    K_xyz_reproj = torch.bmm(intrinsics_ref.unsqueeze(0).expand(N, -1, -1), xyz_reproj)
    xy_reproj = K_xyz_reproj[:, :2, :] / (K_xyz_reproj[:, 2:3, :] + 1e-6)

    return depth_reproj, xy_reproj[:, 0, :].view(N, H, W), xy_reproj[:, 1, :].view(N, H, W)


def filter_depth_gpu(pair_folder, scan_folder, out_folder, plyfilename, args, device):
    pair_file = os.path.join(pair_folder, "pair.txt")
    vertexs = []
    vertex_colors = []

    pair_data = read_pair_file(pair_file)

    for ref_view, src_views in pair_data:
        dy_range = min(args.e_view, len(src_views))
        src_views = src_views[:dy_range]
        if len(src_views) == 0: continue

        # 加载主视图并送入 GPU
        ref_in_np, ref_ex_np = read_camera_parameters(os.path.join(scan_folder, 'cams/{:0>8}_cam.txt'.format(ref_view)))
        ref_img = read_img(os.path.join(scan_folder, 'images/{:0>8}.jpg'.format(ref_view)))

        # 强制 NumPy 分配连续正向内存，解决 negative stride 问题
        ref_depth_est = torch.from_numpy(
            read_pfm(os.path.join(out_folder, 'depth_est/{:0>8}.pfm'.format(ref_view)))[0].copy()).to(device)
        confidence = torch.from_numpy(
            read_pfm(os.path.join(out_folder, 'confidence/{:0>8}.pfm'.format(ref_view)))[0].copy()).to(device)
        photo_mask = confidence > args.conf

        ref_in_t = torch.from_numpy(ref_in_np.copy()).to(device)
        ref_ex_t = torch.from_numpy(ref_ex_np.copy()).to(device)
        H, W = ref_depth_est.shape

        # 将该视角下的所有源视图数据堆叠（Stack），打包送进显存
        src_depths, src_ins, src_exs = [], [], []
        for src_view in src_views:
            i_np, e_np = read_camera_parameters(os.path.join(scan_folder, 'cams/{:0>8}_cam.txt'.format(src_view)))
            d_np = read_pfm(os.path.join(out_folder, 'depth_est/{:0>8}.pfm'.format(src_view)))[0]
            src_ins.append(torch.from_numpy(i_np.copy()))
            src_exs.append(torch.from_numpy(e_np.copy()))
            src_depths.append(torch.from_numpy(d_np.copy()))

        src_depths_t = torch.stack(src_depths).to(device)
        src_ins_t = torch.stack(src_ins).to(device)
        src_exs_t = torch.stack(src_exs).to(device)

        # 一键调用重投影
        depth_reproj, x2d, y2d = batch_reproject_gpu(ref_depth_est, ref_in_t, ref_ex_t, src_depths_t, src_ins_t,
                                                     src_exs_t, device)

        y_ref, x_ref = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        x_ref_flat = x_ref.unsqueeze(0).float()
        y_ref_flat = y_ref.unsqueeze(0).float()

        dist = torch.sqrt((x2d - x_ref_flat) ** 2 + (y2d - y_ref_flat) ** 2)
        depth_diff = torch.abs(depth_reproj - ref_depth_est.unsqueeze(0)) / (ref_depth_est.unsqueeze(0) + 1e-6)

        geo_mask_sums = torch.zeros((dy_range - args.s_view, H, W), dtype=torch.int32, device=device)

        # 极速计算各个梯度的容忍掩码
        for i_idx, i in enumerate(range(args.s_view, dy_range)):
            thresh_d = math.log(max(i, 1.05), 10) * args.diff_base
            mask = (dist < i * args.dist_base) & (depth_diff < thresh_d)
            geo_mask_sums[i_idx] = mask.sum(dim=0, dtype=torch.int32)

        broad_mask = (dist < (dy_range - 1) * args.dist_base) & (
                    depth_diff < math.log(max(dy_range - 1, 1.05), 10) * args.diff_base)
        depth_reproj[~broad_mask] = 0.0
        geo_mask_sum_for_avg = broad_mask.sum(dim=0, dtype=torch.float32)

        depth_est_averaged = (depth_reproj.sum(dim=0) + ref_depth_est) / (geo_mask_sum_for_avg + 1.0)
        depth_est_averaged[confidence > args.conf_stage] = ref_depth_est[confidence > args.conf_stage]

        geo_mask = geo_mask_sums[-1] >= dy_range
        for i_idx, i in enumerate(range(args.s_view, dy_range)):
            geo_mask = geo_mask | (geo_mask_sums[i_idx] >= i)

        final_mask = photo_mask & geo_mask

        # -----------------------------------------------------------
        # 【全量 GPU 矩阵运算：点云 3D 投影】
        # -----------------------------------------------------------

        # 1. 在 GPU 上直接提取合法点
        x_val = x_ref[final_mask].float()
        y_val = y_ref[final_mask].float()
        depth_val = depth_est_averaged[final_mask]

        if x_val.numel() == 0:
            continue

        # 2. 在 GPU 上构建齐次坐标并反投影为世界坐标 [X, Y, Z]
        ones_val = torch.ones_like(x_val)
        xy1_val = torch.stack([x_val, y_val, ones_val], dim=0)  # [3, N_points]

        inv_in_t = torch.inverse(ref_in_t)
        inv_ex_t = torch.inverse(ref_ex_t)

        xyz_ref_val = torch.matmul(inv_in_t, xy1_val) * depth_val
        xyz_ref_val_4 = torch.cat([xyz_ref_val, ones_val.unsqueeze(0)], dim=0)
        xyz_world = torch.matmul(inv_ex_t, xyz_ref_val_4)[:3, :]  # [3, N_points]

        # 3. 颜色提取
        if num_stage == 1:
            color_img = ref_img[1::8, 1::8, :]
        elif num_stage == 2:
            color_img = ref_img[1::4, 1::4, :]
        elif num_stage == 3:
            color_img = ref_img[1::2, 1::2, :]
        else:
            color_img = ref_img

        color_img_t = torch.from_numpy(color_img).to(device)
        color_val = color_img_t[final_mask]  # [N_points, 3]

        # 4. 将该视角的合法点集存入列表 (暂时停留在显存中)
        vertexs.append(xyz_world.t())  # 转置为 [N_points, 3]
        vertex_colors.append(color_val)

        # -----------------------------------------------------------

        # 依旧保存一些必要的 2D 掩码结果（可选，可注释以进一步加速）
        final_mask_np = final_mask.cpu().numpy()
        os.makedirs(os.path.join(out_folder, "mask"), exist_ok=True)
        save_mask(os.path.join(out_folder, "mask/{:0>8}_photo.png".format(ref_view)), photo_mask.cpu().numpy())
        save_mask(os.path.join(out_folder, "mask/{:0>8}_geo.png".format(ref_view)), geo_mask.cpu().numpy())
        save_mask(os.path.join(out_folder, "mask/{:0>8}_final.png".format(ref_view)), final_mask_np)

        print("processing {}, ref-view{:0>2}, Final Mask Ratio: {:.4f} [⚡ 3D GPU]".format(
            scan_folder, ref_view, final_mask_np.mean()))

    # 循环遍历所有视角结束，将显存里所有的点一次性打包给 CPU 硬盘
    if len(vertexs) > 0:
        all_vertexs = torch.cat(vertexs, dim=0).cpu().numpy()
        all_colors = torch.cat(vertex_colors, dim=0).cpu().numpy()
        all_colors = (all_colors * 255).clip(0, 255).astype(np.uint8)

        # 调用极速保存器
        save_ply_fast(plyfilename, all_vertexs, all_colors)
        print(f"\n✅ 成功极速保存了 {len(all_vertexs)} 个点云，文件路径: {plyfilename}")
    else:
        print(f"\n⚠️ 警告: 该场景未能生成任何有效点云！")

    torch.cuda.empty_cache()  # 释放显存


def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def pcd_filter_worker(scan, args):
    # 【核心修改】为每个进程独立初始化 device，避免 CUDA Context 冲突
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.testlist != "all":
        scan_id = int(scan[4:])
        save_name = 'mvsnet{:0>3}_l3.ply'.format(scan_id)
    else:
        save_name = '{}.ply'.format(scan)

    pair_folder = os.path.join(args.testpath, scan)
    scan_folder = os.path.join(args.outdir, scan)
    out_folder = os.path.join(args.outdir, scan)

    filter_depth_gpu(pair_folder, scan_folder, out_folder, os.path.join(args.outdir, save_name), args, device)


if __name__ == '__main__':
    # 【安全多进程】PyTorch 在 CUDA 下使用 multiprocessing 必须使用 'spawn' 模式
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    if args.testlist != "all":
        with open(args.testlist) as f:
            content = f.readlines()
            testlist = [line.rstrip() for line in content]
    else:
        testlist = [e for e in os.listdir(args.testpath) if os.path.isdir(os.path.join(args.testpath, e))] \
            if not args.testpath_single_scene else [os.path.basename(args.testpath_single_scene)]

    print(f"--- Starting Point Cloud Fusion (PyTorch GPU Accelerated ⚡) ---")

    if args.filter_method != "gipuma":
        # 如果设置的 worker 数量超过场景总数，限制它
        workers = min(args.num_worker, len(testlist))
        print(f"🚀 Using {workers} parallel processes for {len(testlist)} scenes...")
        print(f"⚠️ Warning: If you run into 'CUDA Out of Memory', please reduce --num_worker")

        partial_func = partial(pcd_filter_worker, args=args)
        p = mp.Pool(workers, init_worker)
        try:
            p.map(partial_func, testlist)
        except KeyboardInterrupt:
            print("....\nCaught KeyboardInterrupt, terminating workers")
            p.terminate()
        else:
            p.close()
        p.join()