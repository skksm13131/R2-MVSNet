import argparse
import os
import cv2
import numpy as np
import re
import sys
from multiprocessing import Pool


# ==========================================
# 1. 基础工具函数
# ==========================================

def read_pfm(filename):
    try:
        file = open(filename, 'rb')
    except FileNotFoundError:
        return None, None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        file.close()
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        file.close()
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale


def read_cam_file(filename):
    """
    读取 BlendedMVS 的 cam 文件获取深度范围
    """
    with open(filename) as f:
        lines = [line.rstrip() for line in f.readlines()]

    # 解析 depth_ranges (第11行，索引11)
    # 格式通常是: depth_min depth_interval num_depths depth_max
    # 或者: depth_min depth_max (取决于具体的 cam 文件版本，BlendedMVS 通常是 depth_min depth_max)
    words = lines[11].split()
    depth_min = float(words[0])
    depth_max = float(words[-1])

    return depth_min, depth_max


# ==========================================
# 2. 核心评估 Worker (核心修改部分)
# ==========================================

def eval_worker(scan_info):
    scan, dataset_root, pred_root = scan_info

    # 路径配置
    scan_gt_dir = os.path.join(dataset_root, scan, 'rendered_depth_maps')
    scan_cam_dir = os.path.join(dataset_root, scan, 'cams')  # 需要读取cam

    # 兼容两种预测路径结构
    base_pred_dir = os.path.join(pred_root, scan)
    if os.path.isdir(os.path.join(base_pred_dir, 'depth_est')):
        scan_pred_dir = os.path.join(base_pred_dir, 'depth_est')
    else:
        scan_pred_dir = base_pred_dir

    if not os.path.exists(scan_pred_dir):
        return None

    # 获取所有预测文件
    pred_files = [f for f in os.listdir(scan_pred_dir)
                  if f.endswith('.pfm') and 'prob' not in f and 'conf' not in f]

    if not pred_files:
        return None

    # 指标列表
    epe_list = []
    e1_list = []
    e3_list = []

    for pred_name in pred_files:
        # 1. 构造文件路径
        # pred_name 格式通常是 00000000.pfm
        image_id_str = pred_name.split('.')[0]

        # 读取 预测深度
        depth_est, _ = read_pfm(os.path.join(scan_pred_dir, pred_name))

        # 读取 GT 深度
        gt_path = os.path.join(scan_gt_dir, pred_name)
        depth_gt, _ = read_pfm(gt_path)

        # 读取 Cam 参数 (用于归一化)
        cam_path = os.path.join(scan_cam_dir, f'{image_id_str}_cam.txt')

        if depth_gt is None or not os.path.exists(cam_path):
            continue

        # 2. 读取深度范围用于归一化
        depth_min, depth_max = read_cam_file(cam_path)

        # 【核心公式】根据 PPMNet/BlendedMVS 论文标准：
        # 误差归一化因子 = (d_max - d_min) / 128
        # 这里的 128 是标准化的假设平面数，无论模型实际用了多少平面
        depth_interval_unit = (depth_max - depth_min) / 128.0

        # 3. 尺寸对齐
        h_gt, w_gt = depth_gt.shape[:2]
        h_est, w_est = depth_est.shape[:2]
        if h_gt != h_est or w_gt != w_est:
            depth_est = cv2.resize(depth_est, (w_gt, h_gt), interpolation=cv2.INTER_NEAREST)

        # 4. 生成 Mask
        # 有效像素必须满足：
        # a. GT 中有值 (> 1e-3)
        # b. 预测中有值 (> 1e-3)
        # c. GT 在有效深度范围内 (有些GT会有背景无穷远的情况)
        mask = (depth_gt > 1e-3) & (depth_est > 1e-3) & \
               (depth_gt >= depth_min) & (depth_gt <= depth_max)

        # 防止无效 mask 导致计算错误
        if np.sum(mask) == 0:
            continue

        valid_est = depth_est[mask]
        valid_gt = depth_gt[mask]

        # 5. 计算指标
        # 绝对误差
        abs_error = np.abs(valid_est - valid_gt)

        # 【关键步骤】归一化误差
        if depth_interval_unit <= 1e-6:
            depth_interval_unit = 1.0  # 防止除零，虽然理论上不会发生

        normalized_error = abs_error / depth_interval_unit

        # EPE: 平均归一化误差
        epe_list.append(np.mean(normalized_error))

        # e1: 误差 > 1 个 interval unit 的比例
        e1_list.append(np.mean((normalized_error > 1).astype(np.float32)) * 100)

        # e3: 误差 > 3 个 interval unit 的比例
        e3_list.append(np.mean((normalized_error > 3).astype(np.float32)) * 100)

    if len(epe_list) == 0:
        return None

    return {
        'scan': scan,
        'epe': np.mean(epe_list),
        'e1': np.mean(e1_list),
        'e3': np.mean(e3_list)
    }


# ==========================================
# 3. 主程序入口
# ==========================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BlendedMVS Evaluation Script')
    parser.add_argument('--dataset_root', default='E:\\DATA\\BlendedMVS', help='数据集根目录')
    parser.add_argument('--outdir', default='./outputs_bld', help='预测结果目录')
    parser.add_argument('--testlist', default='lists/blendedmvs/val.txt', help='验证集列表')
    parser.add_argument('--num_workers', type=int, default=8, help='进程数')

    args = parser.parse_args()

    print(f"[*] 数据集: {args.dataset_root}")
    print(f"[*] 结果路径: {args.outdir}")

    # 读取列表
    if not os.path.exists(args.testlist):
        # 兼容直接传入单个scan的情况，或者简单的列表
        print(f"[提示] 未找到列表文件 {args.testlist}，尝试将其视为目录下的所有文件夹...")
        if os.path.isdir(args.outdir):
            scans = [d for d in os.listdir(args.outdir) if os.path.isdir(os.path.join(args.outdir, d))]
        else:
            sys.exit(1)
    else:
        with open(args.testlist) as f:
            scans = [line.rstrip() for line in f.readlines()]

    print(f"[*] 待评估场景数: {len(scans)}")

    with Pool(processes=args.num_workers) as pool:
        results = pool.map(eval_worker, [(scan, args.dataset_root, args.outdir) for scan in scans])

    # 过滤 None
    results = [r for r in results if r is not None]

    all_epe = []
    all_e1 = []
    all_e3 = []

    # 收集输出内容
    output_lines = []
    output_lines.append("\n" + "=" * 65)
    output_lines.append(f"{'Scan':<25} | {'EPE':<10} | {'e1 (%)':<10} | {'e3 (%)':<10}")
    output_lines.append("-" * 65)

    for res in results:
        line = f"{res['scan']:<25} | {res['epe']:.4f}     | {res['e1']:.4f}     | {res['e3']:.4f}"
        output_lines.append(line)
        all_epe.append(res['epe'])
        all_e1.append(res['e1'])
        all_e3.append(res['e3'])

    output_lines.append("-" * 65)

    if len(all_epe) > 0:
        output_lines.append(f"Overall Metrics ({len(all_epe)} scans):")
        output_lines.append(f"  > Mean EPE : {np.mean(all_epe):.4f}")
        output_lines.append(f"  > Mean e1  : {np.mean(all_e1):.4f} %")
        output_lines.append(f"  > Mean e3  : {np.mean(all_e3):.4f} %")

        # 简单对比参考 (基于 PPMNet 论文 Table 2)
        output_lines.append("\n[参考指标 (Lower is better)]")
        output_lines.append("  PPMNet (Ours)  : EPE ~0.67 | e1 ~8.47 | e3 ~3.46")
        output_lines.append("  MVSNet         : EPE ~1.49 | e1 ~21.9 | e3 ~8.32")
    else:
        output_lines.append("[错误] 无有效评估结果。")
    output_lines.append("=" * 65)

    # 打印并保存
    output_str = "\n".join(output_lines)
    print(output_str)

    # 保存到 result_m/bld.txt
    save_dir = 'results_m'
    save_filename = 'bld.txt'
    save_path = os.path.join(save_dir, save_filename)
    
    try:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(output_str)
        print(f"\n[*] 结果已保存至: {save_path}")
    except Exception as e:
        print(f"\n[!] 保存文件失败: {e}")
