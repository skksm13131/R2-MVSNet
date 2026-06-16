import traceback
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import open3d as o3d
import numpy as np
import scipy.io as sio
import argparse
import datetime
import csv
import sys
from collections import defaultdict
import concurrent.futures


def evaluate_scan(scan_id, args):
    print(f"--- Processing Scan {scan_id} ---", flush=True)

    # 1. Read reconstructed point cloud
    try:
        recon_ply_path = os.path.join(args.plyPath, f"mvsnet{scan_id:03d}_l3.ply")
        pcd_recon = o3d.io.read_point_cloud(recon_ply_path)
        if not pcd_recon.has_points():
            print(f"  - Error: empty reconstruction: {recon_ply_path}", flush=True)
            return None
    except Exception as e:
        print(f"  - Error reading reconstruction: {e}", flush=True)
        return None

    q_data_raw = np.asarray(pcd_recon.points)

    # voxel_down_sample
    pcd_recon_down = pcd_recon.voxel_down_sample(voxel_size=args.dst)
    q_data = np.asarray(pcd_recon_down.points)
    print(f"  - Reconstruction: {len(q_data_raw)} -> {len(q_data)} (voxel_down_sample {args.dst}mm)", flush=True)

    # 2. Read GT
    try:
        gt_ply_path = os.path.join(args.dataPath, "Points/stl", f"stl{scan_id:03d}_total.ply")
        pcd_gt = o3d.io.read_point_cloud(gt_ply_path)
        q_stl = np.asarray(pcd_gt.points)
        print(f"  - GT: {len(q_stl)} points", flush=True)
    except Exception as e:
        print(f"  - Error reading GT: {e}", flush=True)
        return None

    # 3. Read Mask and Plane
    try:
        mask_path = os.path.join(args.dataPath, "ObsMask", f"ObsMask{scan_id}_{args.margin}.mat")
        mask_mat = sio.loadmat(mask_path)
        obs_mask = mask_mat["ObsMask"]
        bb = mask_mat["BB"]
        res = mask_mat["Res"][0][0]

        plane_path = os.path.join(args.dataPath, "ObsMask", f"Plane{scan_id}.mat")
        plane_mat = sio.loadmat(plane_path)
        p = plane_mat["P"]
    except Exception as e:
        print(f"  - Error reading Mask/Plane: {e}", flush=True)
        return None

    # 4. Compute distances
    print("  - Computing Accuracy (recon -> GT)...", flush=True)
    dist_data = pcd_recon_down.compute_point_cloud_distance(pcd_gt)
    d_data = np.asarray(dist_data)

    print("  - Computing Completeness (GT -> recon)...", flush=True)
    dist_stl = pcd_gt.compute_point_cloud_distance(pcd_recon_down)
    d_stl = np.asarray(dist_stl)

    # 5. Apply Mask and Plane
    q_data_homo = np.hstack((q_data, np.ones((q_data.shape[0], 1))))
    bb_min = bb[0, :]
    q_v = np.round((q_data - bb_min) / res).astype(int) + 1

    valid_indices = (q_v[:, 0] >= 1) & (q_v[:, 0] <= obs_mask.shape[0]) & \
                    (q_v[:, 1] >= 1) & (q_v[:, 1] <= obs_mask.shape[1]) & \
                    (q_v[:, 2] >= 1) & (q_v[:, 2] <= obs_mask.shape[2])

    q_v_valid = q_v[valid_indices]
    mask_values = obs_mask[q_v_valid[:, 0] - 1, q_v_valid[:, 1] - 1, q_v_valid[:, 2] - 1]

    data_in_mask = np.zeros(len(q_data), dtype=bool)
    valid_mask_indices = np.where(valid_indices)[0][mask_values == 1]
    data_in_mask[valid_mask_indices] = True

    q_stl_homo = np.hstack((q_stl, np.ones((len(q_stl), 1))))
    stl_above_plane = (p.T @ q_stl_homo.T).squeeze() > 0

    # 6. Compute metrics
    max_dist = 20.0

    filtered_d_data = d_data[data_in_mask]
    filtered_d_data = filtered_d_data[filtered_d_data < max_dist]

    filtered_d_stl = d_stl[stl_above_plane]
    filtered_d_stl = filtered_d_stl[filtered_d_stl < max_dist]

    if len(filtered_d_data) == 0 or len(filtered_d_stl) == 0:
        print(f"  - Error: No valid points after filtering.", flush=True)
        return None

    acc_mean = float(np.mean(filtered_d_data))
    comp_mean = float(np.mean(filtered_d_stl))

    print(f"  => Scan {scan_id}: Acc={acc_mean:.6f}, Comp={comp_mean:.6f}, Overall={(acc_mean+comp_mean)/2:.6f}", flush=True)

    return {
        "scan_id": scan_id,
        "acc_mean": acc_mean,
        "comp_mean": comp_mean,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DTU Evaluation Script")
    parser.add_argument("--dataPath", type=str, default="/home/u104754251515/data/MVS Data")
    parser.add_argument("--plyPath", type=str, default="./outputs/")
    parser.add_argument("--resultPath", type=str, default="./results_m/")
    parser.add_argument("--dst", type=float, default=0.2, help="Voxel down-sample size (mm)")
    parser.add_argument("--margin", type=int, default=10)
    parser.add_argument("--max_workers", type=int, default=4, help="Max parallel workers")

    args = parser.parse_args()

    if not os.path.exists(args.resultPath):
        os.makedirs(args.resultPath)

    UsedSets = [1, 4, 9, 10, 11, 12, 13, 15, 23, 24, 29, 32, 33, 34, 48, 49, 62, 75, 77, 110, 114, 118]

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = os.path.join(args.resultPath, f"evaluation_results_{timestamp}.csv")

    print(f"Starting evaluation... Results: {csv_filename}")
    print(f"Using max_workers={args.max_workers} for {len(UsedSets)} scans")

    results_list = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_scan = {executor.submit(evaluate_scan, scan_id, args): scan_id for scan_id in UsedSets}

        for future in concurrent.futures.as_completed(future_to_scan):
            scan_id = future_to_scan[future]
            try:
                result = future.result()
                if result:
                    results_list.append(result)
                    print(f"[DONE] Scan {scan_id} | Progress: {len(results_list)}/{len(UsedSets)}", flush=True)
            except Exception as e:
                print(f"[FAIL] Scan {scan_id}: {e}", flush=True)
                traceback.print_exc()

    # Sort and write CSV
    results_list.sort(key=lambda x: x["scan_id"])

    with open(csv_filename, mode="w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["ScanID", "Acc_Mean", "Comp_Mean", "Overall"])
        writer.writeheader()

        all_acc, all_comp = [], []
        for result in results_list:
            overall = (result["acc_mean"] + result["comp_mean"]) / 2.0
            writer.writerow({
                "ScanID": result["scan_id"],
                "Acc_Mean": f"{result['acc_mean']:.6f}",
                "Comp_Mean": f"{result['comp_mean']:.6f}",
                "Overall": f"{overall:.6f}"
            })
            all_acc.append(result["acc_mean"])
            all_comp.append(result["comp_mean"])

        if all_acc:
            overall_acc_mean = np.mean(all_acc)
            overall_comp_mean = np.mean(all_comp)
            total_overall = (overall_acc_mean + overall_comp_mean) / 2.0
            writer.writerow({})
            writer.writerow({
                "ScanID": "AVERAGE",
                "Acc_Mean": f"{overall_acc_mean:.6f}",
                "Comp_Mean": f"{overall_comp_mean:.6f}",
                "Overall": f"{total_overall:.6f}"
            })

    print(f"\n=== FINAL RESULTS ===")
    print(f"Overall Acc  Mean: {overall_acc_mean:.6f}")
    print(f"Overall Comp Mean: {overall_comp_mean:.6f}")
    print(f"Total Overall    : {total_overall:.6f}")
