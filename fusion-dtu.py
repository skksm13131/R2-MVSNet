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

SHARED_ROOT = os.environ.get("R2MVSNET_SHARED_ROOT", "/root/shared-nvme")
DTU_TEST_ROOT = os.environ.get(
    "R2MVSNET_DTU_TEST_PATH",
    os.path.join(SHARED_ROOT, "datasets", "dtu_testing"),
)

parser = argparse.ArgumentParser(
    description='DTU depth filtering and candidate-aware point cloud fusion.')
parser.add_argument('--conf', type=float, default=0.8, help='Photometric confidence threshold.')
parser.add_argument('--conf_stage', type=float, default=0.99,
                    help='High confidence threshold to keep original depth without averaging.')
parser.add_argument('--s_view', type=int, default=3, help='Start of consistent view count for dynamic check.')
parser.add_argument('--e_view', type=int, default=11, help='End of consistent view count for dynamic check.')
parser.add_argument('--dist_base', type=float, default=0.25,
                    help='Base unit for pixel distance threshold. Final threshold is i * dist_base.')
parser.add_argument('--diff_base', type=float, default=0.001,
                    help='Base unit for depth difference threshold. Final threshold is log(i) * diff_base.')
parser.add_argument('--use_fgdr_candidates', action='store_true',
                    help='enable conservative FGDR near/far candidate selection before fusion')
parser.add_argument('--fgdr_candidate_gate_threshold', type=float, default=0.5,
                    help='minimum FGDR geometry gate required to consider near/far candidates')
parser.add_argument('--fgdr_candidate_min_support_gain', type=int, default=1,
                    help='minimum additional consistent source views required to replace main depth')

parser.add_argument('--outdir', default='./outputs',
                    help='Directory where scan folders with depth/confidence are located.')
parser.add_argument('--testpath', default=DTU_TEST_ROOT,
                    help='Original testing data dir (for camera and pair files).')
parser.add_argument('--testpath_single_scene', help='testing data path for single scene')
parser.add_argument('--testlist', default='lists/dtu/test.txt', help='List of scans to process.')
parser.add_argument('--num_worker', type=int, default=4,
                    help='Number of workers for scenes multiprocessing (Watch out for CUDA OOM!).')
parser.add_argument('--ndepths', type=str, default="48,32,8", help='ndepths')
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


def evaluate_depth_candidate_gpu(depth_candidate, ref_in_t, ref_ex_t, src_depths_t, src_ins_t, src_exs_t,
                                 args, device):
    num_sources, height, width = src_depths_t.shape
    effective_s_view = max(1, min(args.s_view, num_sources))
    effective_e_view = max(effective_s_view + 1, num_sources)

    depth_reproj, x2d, y2d = batch_reproject_gpu(
        depth_candidate, ref_in_t, ref_ex_t, src_depths_t, src_ins_t, src_exs_t, device)

    y_ref, x_ref = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing='ij',
    )
    dist = torch.sqrt(
        (x2d - x_ref.unsqueeze(0).float()) ** 2 +
        (y2d - y_ref.unsqueeze(0).float()) ** 2
    )
    depth_diff = torch.abs(depth_reproj - depth_candidate.unsqueeze(0)) / (
        depth_candidate.unsqueeze(0).abs() + 1e-6)

    geo_mask_sums = []
    for consistent_views in range(effective_s_view, effective_e_view):
        depth_threshold = math.log(max(consistent_views, 1.05), 10) * args.diff_base
        consistency = (
            (dist < consistent_views * args.dist_base) &
            (depth_diff < depth_threshold)
        )
        geo_mask_sums.append(consistency.sum(dim=0, dtype=torch.int32))

    geo_mask = geo_mask_sums[-1] >= num_sources
    for support_sum, consistent_views in zip(
            geo_mask_sums, range(effective_s_view, effective_e_view)):
        geo_mask = geo_mask | (support_sum >= consistent_views)

    broad_level = max(1, num_sources - 1)
    broad_mask = (
        (dist < broad_level * args.dist_base) &
        (depth_diff < math.log(max(broad_level, 1.05), 10) * args.diff_base)
    )
    support_count = broad_mask.sum(dim=0, dtype=torch.int32)
    depth_reproj_for_average = depth_reproj.masked_fill(~broad_mask, 0.0)
    depth_averaged = (
        depth_reproj_for_average.sum(dim=0) + depth_candidate
    ) / (support_count.float() + 1.0)

    residual_sum = (depth_diff * broad_mask.float()).sum(dim=0)
    mean_residual = residual_sum / support_count.clamp(min=1).float()
    mean_residual = torch.where(
        support_count > 0,
        mean_residual,
        torch.full_like(mean_residual, float('inf')),
    )
    return depth_averaged, geo_mask, support_count, mean_residual, x_ref, y_ref


def gather_candidate(candidate_stack, candidate_choice):
    return torch.gather(candidate_stack, 0, candidate_choice.unsqueeze(0)).squeeze(0)


def select_fgdr_candidate(support_stack, residual_stack, geo_stack, candidate_depth_stack,
                          geometry_gate, confidence, args):
    if support_stack.size(0) < 2:
        raise ValueError("FGDR candidate selection requires at least one alternative depth")

    best_alt_choice = torch.ones_like(support_stack[0], dtype=torch.long)
    best_alt_support = support_stack[1]
    best_alt_residual = residual_stack[1]
    for candidate_idx in range(2, support_stack.size(0)):
        candidate_is_better = (
            (support_stack[candidate_idx] > best_alt_support) |
            ((support_stack[candidate_idx] == best_alt_support) &
             (residual_stack[candidate_idx] < best_alt_residual))
        )
        best_alt_choice = torch.where(
            candidate_is_better,
            torch.full_like(best_alt_choice, candidate_idx),
            best_alt_choice,
        )
        best_alt_support = torch.where(
            candidate_is_better,
            support_stack[candidate_idx],
            best_alt_support,
        )
        best_alt_residual = torch.where(
            candidate_is_better,
            residual_stack[candidate_idx],
            best_alt_residual,
        )

    best_alt_support = gather_candidate(support_stack, best_alt_choice)
    best_alt_geo = gather_candidate(geo_stack, best_alt_choice)
    best_alt_depth = gather_candidate(candidate_depth_stack, best_alt_choice)

    switch_mask = (
        (geometry_gate >= args.fgdr_candidate_gate_threshold) &
        (confidence <= args.conf_stage) &
        best_alt_geo &
        torch.isfinite(best_alt_depth) &
        (best_alt_depth > 0.0) &
        (best_alt_support >= (
            support_stack[0] + args.fgdr_candidate_min_support_gain))
    )
    candidate_choice = torch.where(
        switch_mask,
        best_alt_choice,
        torch.zeros_like(best_alt_choice),
    )
    return candidate_choice, switch_mask


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

        candidate_choice = None
        candidate_switch_mask = None
        if args.use_fgdr_candidates:
            candidate_paths = {
                'main': os.path.join(out_folder, 'depth_candidate_main/{:0>8}.pfm'.format(ref_view)),
                'near': os.path.join(out_folder, 'depth_near/{:0>8}.pfm'.format(ref_view)),
                'far': os.path.join(out_folder, 'depth_far/{:0>8}.pfm'.format(ref_view)),
                'gate': os.path.join(out_folder, 'geometry_gate/{:0>8}.pfm'.format(ref_view)),
            }
            required_paths = [candidate_paths['near'], candidate_paths['far'], candidate_paths['gate']]
            missing = [path for path in required_paths if not os.path.isfile(path)]
            if missing:
                raise FileNotFoundError(
                    "FGDR candidate fusion requested, but candidate maps are missing: " + ", ".join(missing))

            ref_depth_near = torch.from_numpy(read_pfm(candidate_paths['near'])[0].copy()).to(device)
            ref_depth_far = torch.from_numpy(read_pfm(candidate_paths['far'])[0].copy()).to(device)
            geometry_gate = torch.from_numpy(read_pfm(candidate_paths['gate'])[0].copy()).to(device)
            candidate_depths = [ref_depth_est]
            if os.path.isfile(candidate_paths['main']):
                ref_depth_candidate_main = torch.from_numpy(
                    read_pfm(candidate_paths['main'])[0].copy()).to(device)
                candidate_depths.append(ref_depth_candidate_main)
            candidate_depths.extend([ref_depth_near, ref_depth_far])
            candidate_results = [
                evaluate_depth_candidate_gpu(
                    candidate_depth,
                    ref_in_t,
                    ref_ex_t,
                    src_depths_t,
                    src_ins_t,
                    src_exs_t,
                    args,
                    device,
                )
                for candidate_depth in candidate_depths
            ]

            averaged_stack = torch.stack([result[0] for result in candidate_results], dim=0)
            geo_stack = torch.stack([result[1] for result in candidate_results], dim=0)
            support_stack = torch.stack([result[2] for result in candidate_results], dim=0)
            residual_stack = torch.stack([result[3] for result in candidate_results], dim=0)
            x_ref, y_ref = candidate_results[0][4], candidate_results[0][5]

            candidate_choice, candidate_switch_mask = select_fgdr_candidate(
                support_stack,
                residual_stack,
                geo_stack,
                torch.stack(candidate_depths, dim=0),
                geometry_gate,
                confidence,
                args,
            )
            depth_est_averaged = gather_candidate(averaged_stack, candidate_choice)
            geo_mask = gather_candidate(geo_stack, candidate_choice)
        else:
            depth_est_averaged, geo_mask, _, _, x_ref, y_ref = evaluate_depth_candidate_gpu(
                ref_depth_est,
                ref_in_t,
                ref_ex_t,
                src_depths_t,
                src_ins_t,
                src_exs_t,
                args,
                device,
            )

        depth_est_averaged[confidence > args.conf_stage] = ref_depth_est[confidence > args.conf_stage]

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
        color_img = ref_img
        if color_img.shape[0] != H or color_img.shape[1] != W:
            color_img = cv2.resize(color_img, (W, H), interpolation=cv2.INTER_LINEAR)

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

        candidate_message = ""
        if candidate_choice is not None:
            switch_final = candidate_switch_mask & final_mask
            final_count = final_mask.sum().clamp(min=1).float()
            switch_ratio = (switch_final.sum().float() / final_count).item()
            save_mask(
                os.path.join(out_folder, "mask/{:0>8}_fgdr_switch.png".format(ref_view)),
                candidate_switch_mask.cpu().numpy(),
            )
            choice_scale = 255 // max(len(candidate_depths) - 1, 1)
            choice_image = (
                candidate_choice.cpu().numpy().astype(np.uint8) * choice_scale
            )
            Image.fromarray(choice_image).save(
                os.path.join(out_folder, "mask/{:0>8}_fgdr_choice.png".format(ref_view)))
            if len(candidate_depths) == 4:
                refined_ratio = (((candidate_choice == 1) & final_mask).sum().float() / final_count).item()
                near_ratio = (((candidate_choice == 2) & final_mask).sum().float() / final_count).item()
                far_ratio = (((candidate_choice == 3) & final_mask).sum().float() / final_count).item()
                candidate_message = (
                    f", FGDR Switch: {switch_ratio:.4f} "
                    f"(refined={refined_ratio:.4f}, near={near_ratio:.4f}, far={far_ratio:.4f})"
                )
            else:
                near_ratio = (((candidate_choice == 1) & final_mask).sum().float() / final_count).item()
                far_ratio = (((candidate_choice == 2) & final_mask).sum().float() / final_count).item()
                candidate_message = (
                    f", FGDR Switch: {switch_ratio:.4f} "
                    f"(near={near_ratio:.4f}, far={far_ratio:.4f})"
                )

        print("processing {}, ref-view{:0>2}, Final Mask Ratio: {:.4f}{} [⚡ 3D GPU]".format(
            scan_folder, ref_view, final_mask_np.mean(), candidate_message))

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

    if scan.startswith("scan") and scan[4:].isdigit():
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
