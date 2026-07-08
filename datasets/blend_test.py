from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
from datasets.data_io import *
import cv2


class MVSDataset(Dataset):
    def __init__(self, datapath, listfile, mode, nviews, ndepths=192, interval_scale=1.0, **kwargs):
        super(MVSDataset, self).__init__()
        self.datapath = datapath
        self.listfile = listfile
        self.mode = mode
        self.nviews = nviews
        self.ndepths = ndepths
        self.interval_scale = interval_scale
        # assert self.mode in ["train", "val", "test"]
        self.metas = self.build_list()

    def build_list(self):
        metas = []
        scans = self.listfile
        for scan in scans:
            pair_file = "{}/cams/pair.txt".format(scan)
            with open(os.path.join(self.datapath, pair_file)) as f:
                num_viewpoint = int(f.readline())
                for _ in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    if len(src_views) < self.nviews - 1:
                        continue
                    metas.append((scan, ref_view, src_views))
        print("数据集模式: {}, 场景元数据数量: {}".format(self.mode, len(metas)))
        return metas

    def __len__(self):
        return len(self.metas)

    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = [line.rstrip() for line in f.readlines()]
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
        intrinsics[:2, :] /= 4.0
        depth_min = float(lines[11].split()[0])
        depth_max = float(lines[11].split()[-1])
        depth_interval = (depth_max - depth_min) / self.ndepths
        return intrinsics, extrinsics, depth_min, depth_interval

    def read_img(self, filename):
        img = Image.open(filename)
        np_img = np.array(img, dtype=np.float32) / 255.
        return np_img

    def read_depth(self, filename):
        return np.array(read_pfm(filename)[0], dtype=np.float32)

    def __getitem__(self, idx):
        meta = self.metas[idx]
        scan, ref_view, src_views = meta
        view_ids = [ref_view] + src_views[:self.nviews - 1]
        imgs = []
        proj_matrices = []
        depth_ms = None
        mask_ms = None
        depth_values = None
        for i, vid in enumerate(view_ids):
            img_filename = os.path.join(self.datapath, '{}/blended_images/{:0>8}.jpg'.format(scan, vid))
            proj_mat_filename = os.path.join(self.datapath, '{}/cams/{:0>8}_cam.txt'.format(scan, vid))
            imgs.append(self.read_img(img_filename).transpose(2, 0, 1))
            intrinsics, extrinsics, depth_min, depth_interval_ = self.read_cam_file(proj_mat_filename)
            proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)
            proj_mat[0, :4, :4] = extrinsics
            proj_mat[1, :3, :3] = intrinsics
            proj_matrices.append(proj_mat)
            if i == 0:
                depth_max = depth_min + (self.ndepths - 1) * depth_interval_
                if depth_min <= 1e-6:
                    depth_min = 0.1
                if depth_max <= depth_min:
                    depth_max = depth_min + 100.0
                inv_depth_start = 1.0 / depth_min
                inv_depth_end = 1.0 / depth_max
                inv_depth_values = np.linspace(inv_depth_start, inv_depth_end, self.ndepths, dtype=np.float32)
                depth_values = 1.0 / inv_depth_values
                depth_ms = {"stage1": np.zeros((1, 1))}
                mask_ms = {"stage1": np.zeros((1, 1))}
        imgs = np.stack(imgs)
        proj_matrices = np.stack(proj_matrices)
        stage2_pjmats = proj_matrices.copy()
        stage2_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 2
        stage3_pjmats = proj_matrices.copy()
        stage3_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 4
        proj_matrices_ms = {
            "stage1": proj_matrices,
            "stage2": stage2_pjmats,
            "stage3": stage3_pjmats,
        }
        filename_template = f"{scan}/{{}}/{ref_view:0>8}{{}}"
        return {
            "imgs": imgs,
            "proj_matrices": proj_matrices_ms,
            "depth": depth_ms,
            "depth_values": depth_values,
            "mask": mask_ms,
            "filename": filename_template
        }