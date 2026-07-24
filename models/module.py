import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
sys.path.append("..")
from utils import local_pcd

"""
module.py
│
├── 1. 基础工具函数
│   ├── init_bn()
│   └── init_uniform()
│
├── 2. 基础卷积模块
│   ├── Conv2d
│   ├── Deconv2d
│   ├── Conv3d
│   ├── Deconv3d
│   ├── ConvBnReLU
│   ├── ConvBn
│   ├── ConvBnReLU3D
│   └── ConvBn3D
│
├── 3. 基础网络结构模块
│   ├── BasicBlock
│   └── Hourglass3d
│
├── 4. 单应性变换 / 特征投影
│   └── homo_warping()
│
├── 5. 2D 上采样融合模块
│   └── DeConv2dFuse
│
├── 6. RAFE 可靠性感知特征模块
│   └── ReliabilityAwareFeatureAdapter
│
├── 7. 特征提取主网络
│   └── FeatureNet
│
├── 8. 代价体正则化网络
│   └── CostRegNet
│
├── 9. 深度图细化网络
│   └── RefineNet
│
├── 10. FGDR 候选深度细化模块
│   └── FusionGuidedDepthRefinement
│
└── 11. 深度采样、深度回归和损失函数
    ├── depth_regression()
    ├── cas_mvsnet_loss()
    ├── get_cur_depth_range_samples()
    └── get_depth_range_samples()

"""

def init_bn(module):
    if module.weight is not None:
        nn.init.ones_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    return


def init_uniform(module, init_method):
    if module.weight is not None:
        if init_method == "kaiming":
            nn.init.kaiming_uniform_(module.weight)
        elif init_method == "xavier":
            nn.init.xavier_uniform_(module.weight)
    return

class Conv2d(nn.Module):
    """Applies a 2D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu

    Notes:
        Default momentum for batch normalization is set to be 0.01,

    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Conv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=(not bn), **kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)


class Deconv2d(nn.Module):
    """Applies a 2D deconvolution (optionally with batch normalization and relu activation)
       over an input signal composed of several input planes.

       Attributes:
           conv (nn.Module): convolution module
           bn (nn.Module): batch normalization module
           relu (bool): whether to activate by relu

       Notes:
           Default momentum for batch normalization is set to be 0.01,

       """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Deconv2d, self).__init__()
        self.out_channels = out_channels
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                                       bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        y = self.conv(x)
        if self.stride == 2:
            h, w = list(x.size())[2:]
            y = y[:, :, :2 * h, :2 * w].contiguous()
        if self.bn is not None:
            x = self.bn(y)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)

class Conv3d(nn.Module):
    """Applies a 3D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu

    Notes:
        Default momentum for batch normalization is set to be 0.01,

    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Conv3d, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm3d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)

class Deconv3d(nn.Module):
    """Applies a 3D deconvolution (optionally with batch normalization and relu activation)
       over an input signal composed of several input planes.

       Attributes:
           conv (nn.Module): convolution module
           bn (nn.Module): batch normalization module
           relu (bool): whether to activate by relu

       Notes:
           Default momentum for batch normalization is set to be 0.01,

       """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Deconv3d, self).__init__()
        self.out_channels = out_channels
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                                       bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm3d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        y = self.conv(x)
        if self.bn is not None:
            x = self.bn(y)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)



class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class ConvBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBn, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class ConvBn3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBn3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = ConvBnReLU(in_channels, out_channels, kernel_size=3, stride=stride, pad=1)
        self.conv2 = ConvBn(out_channels, out_channels, kernel_size=3, stride=1, pad=1)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        return out


class Hourglass3d(nn.Module):
    def __init__(self, channels):
        super(Hourglass3d, self).__init__()

        self.conv1a = ConvBnReLU3D(channels, channels * 2, kernel_size=3, stride=2, pad=1)
        self.conv1b = ConvBnReLU3D(channels * 2, channels * 2, kernel_size=3, stride=1, pad=1)

        self.conv2a = ConvBnReLU3D(channels * 2, channels * 4, kernel_size=3, stride=2, pad=1)
        self.conv2b = ConvBnReLU3D(channels * 4, channels * 4, kernel_size=3, stride=1, pad=1)

        self.dconv2 = nn.Sequential(
            nn.ConvTranspose3d(channels * 4, channels * 2, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(channels * 2))

        self.dconv1 = nn.Sequential(
            nn.ConvTranspose3d(channels * 2, channels, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(channels))

        self.redir1 = ConvBn3D(channels, channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = ConvBn3D(channels * 2, channels * 2, kernel_size=1, stride=1, pad=0)

    def forward(self, x):
        conv1 = self.conv1b(self.conv1a(x))
        conv2 = self.conv2b(self.conv2a(conv1))
        dconv2 = F.relu(self.dconv2(conv2) + self.redir2(conv1), inplace=True)
        dconv1 = F.relu(self.dconv1(dconv2) + self.redir1(x), inplace=True)
        return dconv1


def homo_warping(src_fea, src_proj, ref_proj, depth_values):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, Ndepth] o [B, Ndepth, H, W]
    # out: [B, C, Ndepth, H, W]
    batch, channels = src_fea.shape[0], src_fea.shape[1]
    num_depth = depth_values.shape[1]
    height, width = src_fea.shape[2], src_fea.shape[3]

    with torch.no_grad():
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, width, dtype=torch.float32, device=src_fea.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_values.view(batch, 1, num_depth,
                                                                                            -1)  # [B, 3, Ndepth, H*W]
        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]
        proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]
        proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
        proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        grid = proj_xy

    warped_src_fea = F.grid_sample(src_fea, grid.view(batch, num_depth * height, width, 2), mode='bilinear',
                                   padding_mode='zeros')
    warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)

    return warped_src_fea

class DeConv2dFuse(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, relu=True, bn=True,
                 bn_momentum=0.1):
        super(DeConv2dFuse, self).__init__()

        self.deconv = Deconv2d(in_channels, out_channels, kernel_size, stride=2, padding=1, output_padding=1,
                               bn=True, relu=relu, bn_momentum=bn_momentum)

        self.conv = Conv2d(2*out_channels, out_channels, kernel_size, stride=1, padding=1,
                           bn=bn, relu=relu, bn_momentum=bn_momentum)

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x_pre, x):
        x = self.deconv(x)
        x = torch.cat((x, x_pre), dim=1)
        x = self.conv(x)
        return x


class ReliabilityAwareFeatureAdapter(nn.Module):
    """Feature-side reliability estimator with residual structure-aware enhancement."""

    def __init__(self, channels, prior_channels=7, min_hidden_channels=8):
        super(ReliabilityAwareFeatureAdapter, self).__init__()
        hidden_channels = max(min_hidden_channels, channels // 2)
        self.prior_proj = nn.Sequential(
            Conv2d(prior_channels, hidden_channels, 3, 1, padding=1),
            nn.Conv2d(hidden_channels, channels, 1, bias=False),
        )
        self.gate = nn.Sequential(
            Conv2d(channels * 2, hidden_channels, 3, 1, padding=1),
            nn.Conv2d(hidden_channels, channels, 1, bias=True),
        )
        self.reliability_head = nn.Sequential(
            Conv2d(channels * 2, hidden_channels, 3, 1, padding=1),
            nn.Conv2d(hidden_channels, 1, 1, bias=True),
        )
        self.residual_scale_logit = nn.Parameter(torch.tensor(-2.2))
        nn.init.constant_(self.gate[-1].bias, -2.0)
        nn.init.constant_(self.reliability_head[-1].bias, 0.0)

    def forward(self, feature, structure_prior, parent_reliability=None):
        structure_prior = F.interpolate(
            structure_prior,
            size=feature.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        prior_feature = self.prior_proj(structure_prior)
        reliability_input = torch.cat([feature, prior_feature], dim=1)
        reliability = torch.sigmoid(self.reliability_head(reliability_input))
        if parent_reliability is not None:
            parent_reliability = F.interpolate(
                parent_reliability,
                size=feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            reliability = 0.7 * reliability + 0.3 * parent_reliability

        gate = torch.sigmoid(self.gate(reliability_input))
        residual_scale = torch.sigmoid(self.residual_scale_logit)
        reliability_gate = (0.5 + reliability).clamp(min=0.5, max=1.5)
        enhanced_feature = feature + residual_scale * gate * reliability_gate * prior_feature
        return enhanced_feature, reliability


class FeatureNet(nn.Module):
    def __init__(self, base_channels, num_stage=3, stride=4, arch_mode="unet", use_rafe=False):
        super(FeatureNet, self).__init__()
        assert arch_mode in ["unet", "fpn"], print("mode must be in 'unet' or 'fpn', but get:{}".format(arch_mode))
        print("*************feature extraction arch mode:{}****************".format(arch_mode))
        self.arch_mode = arch_mode
        self.stride = stride
        self.base_channels = base_channels
        self.num_stage = num_stage
        self.use_rafe = use_rafe
        if self.use_rafe:
            print("*************RAFE reliability-aware feature extraction enabled****************")

        self.conv0 = nn.Sequential(
            Conv2d(3, base_channels, 3, 1, padding=1),
            Conv2d(base_channels, base_channels, 3, 1, padding=1),
        )

        self.conv1 = nn.Sequential(
            Conv2d(base_channels, base_channels * 2, 5, stride=2, padding=2),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
        )

        self.conv2 = nn.Sequential(
            Conv2d(base_channels * 2, base_channels * 4, 5, stride=2, padding=2),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
        )

        self.out1 = nn.Conv2d(base_channels * 4, base_channels * 4, 1, bias=False)
        self.out_channels = [4 * base_channels]

        if self.arch_mode == "fpn":
            final_chs = base_channels * 4
            if num_stage == 3:
                self.inner1 = nn.Conv2d(base_channels * 2, final_chs, 1, bias=True)
                self.inner2 = nn.Conv2d(base_channels * 1, final_chs, 1, bias=True)

                self.out2 = nn.Conv2d(final_chs, base_channels * 2, 3, padding=1, bias=False)
                self.out3 = nn.Conv2d(final_chs, base_channels, 3, padding=1, bias=False)
                self.out_channels.append(base_channels * 2)
                self.out_channels.append(base_channels)

        if self.use_rafe:
            self.rafe_adapters = nn.ModuleDict({
                "stage1": ReliabilityAwareFeatureAdapter(base_channels * 4),
                "stage2": ReliabilityAwareFeatureAdapter(base_channels * 2),
                "stage3": ReliabilityAwareFeatureAdapter(base_channels),
            })

    def _normalize_prior_channel(self, x):
        mean = x.mean(dim=[2, 3], keepdim=True)
        std = x.std(dim=[2, 3], keepdim=True).clamp(min=1e-4)
        return (x - mean) / std

    def build_structure_prior(self, x):
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        grad_x = F.pad(gray[:, :, :, 1:] - gray[:, :, :, :-1], (0, 1, 0, 0))
        grad_y = F.pad(gray[:, :, 1:, :] - gray[:, :, :-1, :], (0, 0, 0, 1))
        grad_mag = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-6)
        local_mean = F.avg_pool2d(gray, kernel_size=5, stride=1, padding=2)
        texture = F.avg_pool2d((gray - local_mean).pow(2), kernel_size=5, stride=1, padding=2)

        batch, _, height, width = gray.shape
        y_coord = torch.linspace(-1.0, 1.0, height, device=x.device, dtype=x.dtype).view(1, 1, height, 1)
        x_coord = torch.linspace(-1.0, 1.0, width, device=x.device, dtype=x.dtype).view(1, 1, 1, width)
        y_coord = y_coord.expand(batch, 1, height, width)
        x_coord = x_coord.expand(batch, 1, height, width)

        return torch.cat([
            self._normalize_prior_channel(gray),
            self._normalize_prior_channel(grad_x),
            self._normalize_prior_channel(grad_y),
            self._normalize_prior_channel(grad_mag),
            self._normalize_prior_channel(texture),
            x_coord,
            y_coord,
        ], dim=1)

    def inject_reliability_feature(self, stage_name, feature, structure_prior, parent_reliability=None):
        if not self.use_rafe:
            return feature, None
        return self.rafe_adapters[stage_name](feature, structure_prior, parent_reliability=parent_reliability)

    def forward(self, x):
        structure_prior = self.build_structure_prior(x) if self.use_rafe else None
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)

        intra_feat = conv2
        outputs = {}
        out = self.out1(intra_feat)
        out, reliability = self.inject_reliability_feature("stage1", out, structure_prior)
        outputs["stage1"] = out
        if reliability is not None:
            outputs["stage1_reliability"] = reliability
       
        if self.arch_mode == "fpn":
            if self.num_stage == 3:
                intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner1(conv1)
                out = self.out2(intra_feat)
                out, reliability = self.inject_reliability_feature("stage2", out, structure_prior, reliability)
                outputs["stage2"] = out
                if reliability is not None:
                    outputs["stage2_reliability"] = reliability

                intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner2(conv0)
                out = self.out3(intra_feat)
                out, reliability = self.inject_reliability_feature("stage3", out, structure_prior, reliability)
                outputs["stage3"] = out
                if reliability is not None:
                    outputs["stage3_reliability"] = reliability

        return outputs

class CostRegNet(nn.Module):
    def __init__(self, in_channels, base_channels):
        super(CostRegNet, self).__init__()
        self.conv0 = Conv3d(in_channels, base_channels, padding=1)

        self.conv1 = Conv3d(base_channels, base_channels * 2, stride=2, padding=1)
        self.conv2 = Conv3d(base_channels * 2, base_channels * 2, padding=1)

        self.conv3 = Conv3d(base_channels * 2, base_channels * 4, stride=2, padding=1)
        self.conv4 = Conv3d(base_channels * 4, base_channels * 4, padding=1)

        self.conv5 = Conv3d(base_channels * 4, base_channels * 8, stride=2, padding=1)
        self.conv6 = Conv3d(base_channels * 8, base_channels * 8, padding=1)

        self.conv7 = Deconv3d(base_channels * 8, base_channels * 4, stride=2, padding=1, output_padding=1)

        self.conv9 = Deconv3d(base_channels * 4, base_channels * 2, stride=2, padding=1, output_padding=1)

        self.conv11 = Deconv3d(base_channels * 2, base_channels * 1, stride=2, padding=1, output_padding=1)

        self.prob = nn.Conv3d(base_channels, 1, 3, stride=1, padding=1, bias=False)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        x = conv2 + self.conv9(x)
        x = conv0 + self.conv11(x)
        x = self.prob(x)
        return x

class RefineNet(nn.Module):
    def __init__(self):
        super(RefineNet, self).__init__()
        self.conv1 = ConvBnReLU(4, 32)
        self.conv2 = ConvBnReLU(32, 32)
        self.conv3 = ConvBnReLU(32, 32)
        self.res = ConvBnReLU(32, 1)

    def forward(self, img, depth_init):
        concat = F.cat((img, depth_init), dim=1)
        depth_residual = self.res(self.conv3(self.conv2(self.conv1(concat))))
        depth_refined = depth_init + depth_residual
        return depth_refined


class FusionGuidedDepthRefinement(nn.Module):
    def __init__(self, in_channels, hidden_channels=16, max_radius_factor=2.0):
        super(FusionGuidedDepthRefinement, self).__init__()
        self.max_radius_factor = max_radius_factor
        self.context = nn.Sequential(
            ConvBnReLU(in_channels + 4, hidden_channels),
            ConvBnReLU(hidden_channels, hidden_channels),
        )
        self.residual_head = nn.Conv2d(hidden_channels, 1, 3, padding=1)
        self.radius_head = nn.Conv2d(hidden_channels, 1, 3, padding=1)
        self.gate_head = nn.Conv2d(hidden_channels, 1, 3, padding=1)

    def forward(self, ref_feature, depth, depth_values, confidence,
                ref_reliability=None, view_weights=None):
        depth_min = depth_values.min(dim=1)[0]
        depth_max = depth_values.max(dim=1)[0]
        depth_span = (depth_max - depth_min).clamp(min=1e-6)
        if depth_values.size(1) > 1:
            depth_interval = (depth_values[:, 1:] - depth_values[:, :-1]).abs().mean(dim=1)
        else:
            depth_interval = depth_span

        depth_norm = ((depth - depth_min) / depth_span).clamp(0.0, 1.0).unsqueeze(1)
        confidence = confidence.unsqueeze(1).clamp(0.0, 1.0)

        if ref_reliability is None:
            ref_reliability = ref_feature.new_ones(depth_norm.shape)
        elif ref_reliability.shape[-2:] != ref_feature.shape[-2:]:
            ref_reliability = F.interpolate(
                ref_reliability,
                size=ref_feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        if view_weights is None:
            view_uncertainty = ref_feature.new_zeros(depth_norm.shape)
        else:
            if view_weights.shape[-2:] != ref_feature.shape[-2:]:
                view_weights = F.interpolate(view_weights, size=ref_feature.shape[-2:], mode="nearest")
            view_mean = view_weights.mean(dim=1, keepdim=True)
            view_uncertainty = view_weights.std(dim=1, keepdim=True, unbiased=False) / (view_mean.abs() + 1e-6)
            view_uncertainty = view_uncertainty.clamp(0.0, 1.0)

        fgdr_input = torch.cat([ref_feature, depth_norm, confidence, ref_reliability, view_uncertainty], dim=1)
        context = self.context(fgdr_input)
        geometry_gate = torch.sigmoid(self.gate_head(context))
        uncertainty = torch.sigmoid(self.radius_head(context))

        max_radius = depth_interval.unsqueeze(1) * self.max_radius_factor
        delta = (0.25 + 0.75 * uncertainty * geometry_gate) * max_radius
        residual = torch.tanh(self.residual_head(context)) * delta * geometry_gate

        depth_main = (depth.unsqueeze(1) + residual).squeeze(1)
        depth_near = (depth_main.unsqueeze(1) - delta).squeeze(1)
        depth_far = (depth_main.unsqueeze(1) + delta).squeeze(1)

        return {
            "depth": depth_main,
            "fgdr_depth_base": depth,
            "fgdr_depth_main": depth_main,
            "fgdr_depth_near": depth_near,
            "fgdr_depth_far": depth_far,
            "fgdr_delta": delta.squeeze(1),
            "fgdr_geometry_gate": geometry_gate.squeeze(1),
            "fgdr_uncertainty": uncertainty.squeeze(1),
        }


def depth_regression(p, depth_values):
    if depth_values.dim() <= 2:
        # print("regression dim <= 2")
        depth_values = depth_values.view(*depth_values.shape, 1, 1)
    depth = torch.sum(p * depth_values, 1)

    return depth

def cas_mvsnet_loss(inputs, depth_gt_ms, mask_ms, **kwargs):
    depth_loss_weights = kwargs.get("dlossw", None)
    fgdr_loss_weight = kwargs.get("fgdr_loss_weight", 0.0)
    fgdr_radius_weight = kwargs.get("fgdr_radius_weight", 0.1)
    fgdr_center_weight = kwargs.get("fgdr_center_weight", 0.25)

    total_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
    fgdr_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)

    for (stage_inputs, stage_key) in [(inputs[k], k) for k in inputs.keys() if "stage" in k]:
        depth_est = stage_inputs["depth"]
        depth_gt = depth_gt_ms[stage_key]
        mask = mask_ms[stage_key]
        mask = mask > 0.5

        depth_loss = F.smooth_l1_loss(depth_est[mask], depth_gt[mask], reduction='mean')

        if depth_loss_weights is not None:
            stage_idx = int(stage_key.replace("stage", "")) - 1
            total_loss += depth_loss_weights[stage_idx] * depth_loss
        else:
            total_loss += 1.0 * depth_loss

        if fgdr_loss_weight > 0.0 and "fgdr_depth_near" in stage_inputs and "fgdr_depth_far" in stage_inputs:
            depth_near = stage_inputs["fgdr_depth_near"]
            depth_far = stage_inputs["fgdr_depth_far"]
            delta = stage_inputs["fgdr_delta"]
            gate = stage_inputs["fgdr_geometry_gate"]
            confidence = stage_inputs.get("photometric_confidence", None)
            if confidence is None:
                confidence = depth_est.new_ones(depth_est.shape)
            cover_loss = F.smooth_l1_loss(
                F.relu(depth_near[mask] - depth_gt[mask]) + F.relu(depth_gt[mask] - depth_far[mask]),
                torch.zeros_like(depth_gt[mask]),
                reduction='mean',
            )
            high_confidence_radius = F.smooth_l1_loss(
                (confidence[mask].detach() * gate[mask] * delta[mask]),
                torch.zeros_like(delta[mask]),
                reduction='mean',
            )
            candidate_center = stage_inputs.get("fgdr_depth_main", stage_inputs["depth"])
            candidate_center_loss = F.smooth_l1_loss(
                candidate_center[mask],
                depth_gt[mask],
                reduction='mean',
            )
            stage_fgdr_loss = (
                cover_loss +
                fgdr_center_weight * candidate_center_loss +
                fgdr_radius_weight * high_confidence_radius
            )
            fgdr_loss = fgdr_loss + stage_fgdr_loss
            if depth_loss_weights is not None:
                stage_idx = int(stage_key.replace("stage", "")) - 1
                total_loss += depth_loss_weights[stage_idx] * fgdr_loss_weight * stage_fgdr_loss
            else:
                total_loss += fgdr_loss_weight * stage_fgdr_loss

    return total_loss, depth_loss, total_loss.new_tensor(0.0), fgdr_loss


def get_cur_depth_range_samples(cur_depth, ndepth, depth_inteval_pixel, shape, max_depth=192.0, min_depth=0.0):
    #shape, (B, H, W)
    #cur_depth: (B, H, W)
    #return depth_range_values: (B, D, H, W)
    cur_depth_min = (cur_depth - ndepth / 2 * depth_inteval_pixel)  # (B, H, W)
    cur_depth_max = (cur_depth + ndepth / 2 * depth_inteval_pixel)
    # cur_depth_min = (cur_depth - ndepth / 2 * depth_inteval_pixel).clamp(min=0.0)   #(B, H, W)
    # cur_depth_max = (cur_depth_min + (ndepth - 1) * depth_inteval_pixel).clamp(max=max_depth)

    assert cur_depth.shape == torch.Size(shape), "cur_depth:{}, input shape:{}".format(cur_depth.shape, shape)
    new_interval = (cur_depth_max - cur_depth_min) / (ndepth - 1)  # (B, H, W)

    depth_range_samples = cur_depth_min.unsqueeze(1) + (torch.arange(0, ndepth, device=cur_depth.device,
                                                                  dtype=cur_depth.dtype,
                                                                  requires_grad=False).reshape(1, -1, 1,
                                                                                               1) * new_interval.unsqueeze(1))

    return depth_range_samples


def get_depth_range_samples(cur_depth, ndepth, depth_inteval_pixel, device, dtype, shape,
                           max_depth=192.0, min_depth=0.0):
    #shape: (B, H, W)
    #cur_depth: (B, H, W) or (B, D)
    #return depth_range_samples: (B, D, H, W)
    if cur_depth.dim() == 2:
        cur_depth_min = cur_depth[:, 0]  # (B,)
        cur_depth_max = cur_depth[:, -1]
        new_interval = (cur_depth_max - cur_depth_min) / (ndepth - 1)  # (B, )

        depth_range_samples = cur_depth_min.unsqueeze(1) + (torch.arange(0, ndepth, device=device, dtype=dtype,
                                                                       requires_grad=False).reshape(1, -1) * new_interval.unsqueeze(1)) #(B, D)

        depth_range_samples = depth_range_samples.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, shape[1], shape[2]) #(B, D, H, W)

    else:

        depth_range_samples = get_cur_depth_range_samples(cur_depth, ndepth, depth_inteval_pixel, shape, max_depth, min_depth)

    return depth_range_samples
