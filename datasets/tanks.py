from torch.utils.data import Dataset
from datasets.data_io import *
import os
import numpy as np
import cv2
from PIL import Image
import torch


class MVSDataset(Dataset):
    def __init__(self, datapath, listfile, split='intermediate', nviews=11, ndepths=192,
                 img_wh=(1920, 1080)):
        super(MVSDataset, self).__init__()
        self.datapath = datapath
        self.listfile = listfile
        self.split = split
        self.n_views = nviews
        self.ndepths = ndepths
        self.img_wh = img_wh
        self.stage_scales = {"stage1": 4.0, "stage2": 2.0, "stage3": 1.0}
        self.inter_scans = ['Family', 'Francis', 'Horse', 'Lighthouse', 'M60', 'Panther', 'Playground', 'Train']
        self.adv_scans = ['Auditorium', 'Ballroom', 'Courtroom', 'Museum', 'Palace', 'Temple']
        self.image_sizes = {
            'Family': (1920, 1080), 'Francis': (1920, 1080), 'Horse': (1920, 1080),
            'Lighthouse': (2048, 1080), 'M60': (2048, 1080), 'Panther': (2048, 1080),
            'Playground': (1920, 1080), 'Train': (1920, 1080), 'Auditorium': (1920, 1080),
            'Ballroom': (1920, 1080), 'Courtroom': (1920, 1080), 'Museum': (1920, 1080),
            'Palace': (1920, 1080), 'Temple': (1920, 1080)
        }
        self.metas = self.build_metas()

    def build_metas(self):
        metas = []
        for scan in self.listfile:

            if scan in self.inter_scans:
                scan_split = 'intermediate'
            elif scan in self.adv_scans:
                scan_split = 'advanced'
            else:
                print(f"Warning: Scan {scan} not found in predefined lists. Assuming split is '{self.split}'.")
                scan_split = self.split

            pair_file = os.path.join(self.datapath, scan_split, scan, 'pair.txt')
            with open(pair_file) as f:
                num_viewpoint = int(f.readline())
                for _ in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    if len(src_views) > 0:
                        metas.append((scan, ref_view, src_views, scan_split))
        return metas

    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = [line.rstrip() for line in f.readlines()]
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
        depth_min = float(lines[11].split()[0])
        depth_max = float(lines[11].split()[1])
        return intrinsics, extrinsics, depth_min, depth_max

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        scan, ref_view, src_views, split = self.metas[idx]
        view_ids = [ref_view] + src_views[:self.n_views - 1]

        imgs = []
        proj_matrices = []
        depth_values = None
        orig_w, orig_h = self.image_sizes[scan]
        target_w, target_h = self.img_wh
        for i, vid in enumerate(view_ids):
            img_filename = os.path.join(self.datapath, split, scan, f'images/{vid:08d}.jpg')
            cam_filename = os.path.join(self.datapath, split, scan, f'cams_1/{vid:08d}_cam.txt')
            img = Image.open(img_filename)
            img = img.resize(self.img_wh, Image.BILINEAR)
            imgs.append(np.array(img, dtype=np.float32) / 255.0)
            intrinsics, extrinsics, depth_min_, depth_max_ = self.read_cam_file(cam_filename)
            x_scale = target_w / orig_w
            y_scale = target_h / orig_h
            intrinsics[0] *= x_scale
            intrinsics[1] *= y_scale
            proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)
            proj_mat[0, :4, :4] = extrinsics
            proj_mat[1, :3, :3] = intrinsics
            proj_matrices.append(proj_mat)
            if i == 0:
                if depth_min_ <= 1e-6:
                    depth_min_ = 0.1
                if depth_max_ <= depth_min_ + 1e-6:
                    depth_max_ = depth_min_ + 100.0  # Bounding box
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
            proj_mat_stage[:, 1, :2, :] /= scale
            proj_matrices_ms[stage_name] = torch.from_numpy(proj_mat_stage).float()
        imgs = np.stack(imgs).transpose(0, 3, 1, 2)  # (V, H, W, C) -> (V, C, H, W)
        return {
            "imgs": torch.from_numpy(imgs).float(),
            "proj_matrices": proj_matrices_ms,
            "depth_values": torch.from_numpy(depth_values.copy()).float(),
            "filename": scan + '/{}/' + '{:0>8}'.format(view_ids[0]) + "{}"
        }