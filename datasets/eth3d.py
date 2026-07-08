from torch.utils.data import Dataset
from datasets.data_io_e3 import read_cam_file
import os
import numpy as np
import cv2
from PIL import Image
import torch


class ETH3DDataset(Dataset):
    def __init__(self, datapath, listfile, nviews=7, ndepths=192, img_wh=(2688, 1792)):
        super(ETH3DDataset, self).__init__()
        self.datapath = datapath
        self.listfile = listfile  # Path to the scan list file (e.g., test.txt)
        self.n_views = nviews
        self.ndepths = ndepths
        self.img_wh = img_wh
        self.stage_scales = {"stage1": 4.0, "stage2": 2.0, "stage3": 1.0}  # Match UWEMVSNet stages
        assert os.path.exists(self.datapath), f"Data path {self.datapath} does not exist."
        assert os.path.exists(self.listfile), f"List file {self.listfile} does not exist."
        self.metas = self.build_metas()

    def build_metas(self):
        metas = []
        with open(self.listfile) as f:
            scans = [line.rstrip() for line in f.readlines()]

        for scan in scans:
            pair_file = os.path.join(self.datapath, scan, 'pair.txt')
            if not os.path.exists(pair_file):
                print(f"Warning: pair.txt not found for scan {scan}. Skipping.")
                continue

            try:
                # Read pairs and determine original image size from the first image found
                with open(pair_file) as f_pair:
                    num_viewpoint = int(f_pair.readline())
                    # Find first valid reference view to get image size
                    first_ref_view = -1
                    temp_lines = f_pair.readlines()  # Read rest of the lines
                    for i in range(0, len(temp_lines), 2):
                        ref_view_candidate = int(temp_lines[i].rstrip())

                        img_path_candidate = os.path.join(self.datapath, scan, f'images/{ref_view_candidate:08d}.jpg')
                        if os.path.exists(img_path_candidate):
                            first_ref_view = ref_view_candidate
                            break
                    if first_ref_view == -1:
                        print(f"Warning: Could not find any valid image for scan {scan} based on pair.txt. Skipping.")
                        continue

                    img_path = os.path.join(self.datapath, scan, f'images/{first_ref_view:08d}.jpg')
                    img_pil = Image.open(img_path)
                    orig_w, orig_h = img_pil.size
                    img_pil.close()  # Close file handle

                    # Process pairs
                    for i in range(0, len(temp_lines), 2):
                        ref_view = int(temp_lines[i].rstrip())
                        src_views = [int(x) for x in temp_lines[i + 1].rstrip().split()[1::2]]

                        if len(src_views) >= self.n_views - 1:
                            metas.append((scan, ref_view, src_views, orig_w, orig_h))

            except Exception as e:
                print(f"Error processing scan {scan}: {e}")
                continue
        print(f"Built {len(metas)} items for ETH3D dataset.")
        return metas

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        scan, ref_view, src_views, orig_w, orig_h = self.metas[idx]
        view_ids = [ref_view] + src_views[:self.n_views - 1]
        imgs = []
        proj_matrices = []
        depth_values = None
        intrinsics_scaled_ref = None
        extrinsics_ref = None
        depth_min_ref = 0
        depth_max_ref = 0
        target_w, target_h = self.img_wh
        for i, vid in enumerate(view_ids):
            img_filename = os.path.join(self.datapath, scan, f'images/{vid:08d}.jpg')
            cam_filename = os.path.join(self.datapath, scan, f'cams_1/{vid:08d}_cam.txt')
            if not os.path.exists(img_filename):
                print(f"Image file not found: {img_filename}")
                return None
            if not os.path.exists(cam_filename):
                print(f"Camera file not found: {cam_filename}")
                return None
            try:
                img = Image.open(img_filename)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img_resized = img.resize(self.img_wh, Image.BILINEAR)
                img_np = np.array(img_resized, dtype=np.float32) / 255.0
                imgs.append(img_np)
                img.close()
            except Exception as e:
                print(f"Error reading or processing image {img_filename}: {e}")
                return None
            try:
                intrinsics, extrinsics, depth_params = read_cam_file(cam_filename)
                if len(depth_params) >= 2:
                    depth_min_ = depth_params[0]
                    depth_max_ = depth_params[1]
                else:
                    print(f"Warning: No depth range found in {cam_filename}. Using defaults.")
                    depth_min_ = 0.1
                    depth_max_ = 100.0
            except Exception as e:
                print(f"Error reading camera file {cam_filename}: {e}")
                return None
            x_scale = target_w / orig_w
            y_scale = target_h / orig_h
            intrinsics_scaled = intrinsics.copy()
            intrinsics_scaled[0] *= x_scale
            intrinsics_scaled[1] *= y_scale
            proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)
            proj_mat[0, :4, :4] = extrinsics
            proj_mat[1, :3, :3] = intrinsics_scaled[:3, :3]
            proj_mat[1, 3, 0] = depth_min_
            proj_mat[1, 3, 1] = depth_max_
            proj_mat[1, 3, 2] = self.ndepths
            proj_mat[1, 3, 3] = (depth_max_ - depth_min_) / (self.ndepths - 1) if self.ndepths > 1 else 0
            proj_matrices.append(proj_mat)
            if i == 0:
                intrinsics_scaled_ref = intrinsics_scaled
                extrinsics_ref = extrinsics
                depth_min_ref = depth_min_
                depth_max_ref = depth_max_
                if depth_min_ <= 1e-6:
                    print(f"Warning: Scan {scan}, view {vid} has depth_min <= 0 ({depth_min_}). Clamping to 0.1.")
                    depth_min_ = 0.1
                if depth_max_ <= depth_min_ + 1e-6:
                    depth_max_ = depth_min_ + 100.0
                    print(
                        f"Warning: Invalid or zero depth range in {cam_filename}. Using default range ({depth_min_:.2f} - {depth_max_:.2f}).")
                inv_depth_start = 1.0 / depth_min_
                inv_depth_end = 1.0 / depth_max_
                # Linear sampling in inverse depth space
                inv_depth_values = np.linspace(inv_depth_start, inv_depth_end, self.ndepths, dtype=np.float32)
                # Convert back to depth space (Note: this will order depths from near to far)
                depth_values = 1.0 / inv_depth_values
        proj_matrices = np.stack(proj_matrices)
        proj_matrices_ms = {}
        for stage_name, scale in self.stage_scales.items():
            proj_mat_stage = proj_matrices.copy()
            # Adjust intrinsics based on the feature map scale relative to the *target* image size
            proj_mat_stage[:, 1, :2, :3] /= scale
            proj_matrices_ms[stage_name] = torch.from_numpy(proj_mat_stage).float()
        imgs = np.stack(imgs).transpose(0, 3, 1, 2)  # (V, H, W, C) -> (V, C, H, W)
        return {
            "imgs": torch.from_numpy(imgs).float(),
            "proj_matrices": proj_matrices_ms,
            "depth_values": torch.from_numpy(depth_values.copy()).float() if depth_values is not None else torch.tensor(
                []),
            "filename": scan + '/{}/' + '{:0>8}'.format(view_ids[0]) + "{}",

            "intrinsics_scaled": torch.from_numpy(intrinsics_scaled_ref.copy()).float(),
            "extrinsics": torch.from_numpy(extrinsics_ref.copy()).float(),
            "depth_min": depth_min_ref,
            "depth_max": depth_max_ref,
        }