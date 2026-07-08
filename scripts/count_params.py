"""
统计 CascadeMVSNet 各模块参数量和推理时间
用法: python scripts/count_params.py [--config baseline|r2|full]
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import time
import argparse
from models import CascadeMVSNet


def count_parameters(model, verbose=True):
    """统计模型参数量，按模块分解"""
    total = 0
    trainable = 0
    module_stats = {}

    for name, param in model.named_parameters():
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n

        # 按一级模块归类
        module_name = name.split('.')[0]
        if module_name not in module_stats:
            module_stats[module_name] = {'total': 0, 'trainable': 0}
        module_stats[module_name]['total'] += n
        if param.requires_grad:
            module_stats[module_name]['trainable'] += n

    if verbose:
        print(f"\n{'='*60}")
        print(f"{'Module':<30} {'Total':>12} {'Trainable':>12}")
        print(f"{'='*60}")
        for name, stats in sorted(module_stats.items()):
            print(f"{name:<30} {stats['total']:>10,} {stats['trainable']:>10,}")
        print(f"{'='*60}")
        print(f"{'TOTAL':<30} {total:>10,} {trainable:>10,}")
        print(f"{'='*60}")

    return total, trainable, module_stats


def measure_inference_time(model, device='cuda', n_views=5, img_size=(512, 640),
                           n_warmup=10, n_iter=50):
    """测量推理时间"""
    model.eval()
    model.to(device)

    B = 1  # batch size
    H, W = img_size

    # Create dummy inputs matching CascadeMVSNet
    imgs = torch.randn(B, n_views, 3, H, W).to(device)
    proj_matrices = {
        'stage1': torch.randn(B, n_views, 4, 4).to(device),
        'stage2': torch.randn(B, n_views, 4, 4).to(device),
        'stage3': torch.randn(B, n_views, 4, 4).to(device),
    }

    if device == 'cuda':
        torch.cuda.synchronize()

    with torch.no_grad():
        # Warmup
        for _ in range(n_warmup):
            _ = model(imgs, proj_matrices, {'stage1': 425.0, 'stage2': 935.0, 'stage3': 935.0},
                      torch.ones(B).to(device) * 425.0, torch.ones(B).to(device) * 935.0)
        if device == 'cuda':
            torch.cuda.synchronize()

        # Timed iterations
        times = []
        for _ in range(n_iter):
            if device == 'cuda':
                torch.cuda.synchronize()
                start = time.perf_counter()
                _ = model(imgs, proj_matrices, {'stage1': 425.0, 'stage2': 935.0, 'stage3': 935.0},
                          torch.ones(B).to(device) * 425.0, torch.ones(B).to(device) * 935.0)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
            else:
                start = time.perf_counter()
                _ = model(imgs, proj_matrices, {'stage1': 425.0, 'stage2': 935.0, 'stage3': 935.0},
                          torch.ones(B).to(device) * 425.0, torch.ones(B).to(device) * 935.0)
                elapsed = time.perf_counter() - start
            times.append(elapsed)

    avg_time = sum(times) / len(times) * 1000  # ms
    min_time = min(times) * 1000
    max_time = max(times) * 1000

    # GPU memory
    if device == 'cuda':
        mem_alloc = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
        mem_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
        torch.cuda.reset_peak_memory_stats()
    else:
        mem_alloc, mem_reserved = 0, 0

    print(f"\n推理时间 (n_views={n_views}, size={H}x{W}, {n_iter} iters):")
    print(f"  Avg: {avg_time:.1f} ms")
    print(f"  Min: {min_time:.1f} ms")
    print(f"  Max: {max_time:.1f} ms")
    if device == 'cuda':
        print(f"  GPU 显存峰值: {mem_alloc:.1f} MB (allocated), {mem_reserved:.1f} MB (reserved)")

    return avg_time, mem_alloc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='full',
                        choices=['baseline', 'r2', 'full', 'all'],
                        help='模型配置: baseline(无模块), r2(RAFE+SP-RWCV), full(全部), all(全部配置)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--n_views', type=int, default=5)
    args = parser.parse_args()

    configs = {
        'baseline': {'use_rafe': False, 'use_view_attention': False, 'use_fgdr': False, 'label': 'Baseline (CasMVSNet)'},
        'r2':       {'use_rafe': True,  'use_view_attention': True,  'use_fgdr': False, 'label': 'R2-MVSNet (RAFE+SP-RWCV)'},
        'full':     {'use_rafe': True,  'use_view_attention': True,  'use_fgdr': True,  'label': 'R2-MVSNet Full'},
    }

    if args.config == 'all':
        cfgs = configs.values()
    else:
        cfgs = [configs[args.config]]

    for cfg in cfgs:
        print(f"\n{'#'*60}")
        print(f"# {cfg['label']}")
        print(f"{'#'*60}")

        model = CascadeMVSNet(
            ndepths=[48, 32, 8],
            depth_interals_ratio=[4, 2, 1],
            share_cr=False,
            cr_base_chs=[8, 8, 8],
            refine=False,
            use_rafe=cfg['use_rafe'],
            use_view_attention=cfg['use_view_attention'],
            use_fgdr=cfg['use_fgdr'],
            fgdr_anchor_base=cfg['use_fgdr'],
            view_attention_mode='single_pass_reliability_weighted',
            fgdr_max_radius_factor=2.0,
        )

        total, trainable, _ = count_parameters(model)
        print(f"\n总参数: {total:,}  可训练: {trainable:,}")

        if torch.cuda.is_available() and args.device == 'cuda':
            measure_inference_time(model, device=args.device, n_views=args.n_views)
        else:
            print(f"\n(跳过推理时间测量: CUDA不可用)")
            # CPU 近似测量
            measure_inference_time(model, device='cpu', n_views=args.n_views, n_iter=5)


if __name__ == '__main__':
    main()
