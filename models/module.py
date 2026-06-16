import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
sys.path.append("..")
from utils import local_pcd

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


class ChannelAttention(nn.Module):
    """全局通道注意力 - 通过全局池化增强通道响应"""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return torch.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """CBAM空间注意力模块 - 增强重要空间位置的响应"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        return torch.sigmoid(self.conv(concat))


class AttentionGuidedCostVolumeFusion(nn.Module):
    """注意力引导的代价体融合模块 (Attention-guided Cost Volume Fusion, ACVF)

    核心设计思想：
    1. 深度维度注意力 (Depth-wise Attention): 识别最相关的深度假设
    2. 视图维度注意力 (View-wise Attention): 学习不同视图的贡献权重
    3. 空间引导注意力 (Spatial-guided Attention): 利用参考图像语义信息指导融合

    作用位置：在构建 variance cost volume 之前，对各视图特征进行加权融合
    """

    def __init__(self, feature_channels, num_views, reduction=4):
        super().__init__()
        self.num_views = num_views
        self.feat_channels = feature_channels

        # ── 视图注意力：学习每个视图的重要性 ──
        # 对每个视图分别计算注意力权重
        self.view_attention = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels // reduction, 1, bias=False),
            nn.BatchNorm2d(feature_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels // reduction, 1, 1, bias=False),
        )

        # ── 深度维度注意力：识别关键深度层 ──
        # 在构建 cost volume 时，对深度维度加权
        self.depth_attention = nn.Sequential(
            nn.Conv3d(feature_channels, feature_channels // reduction, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(feature_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv3d(feature_channels // reduction, 1, kernel_size=3, padding=1, bias=False),
        )

        # ── 空间引导注意力：利用参考图像的边缘/纹理信息 ──
        # 从参考图像提取语义引导信息
        self.spatial_guide = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),  # 输入参考图像
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1, bias=False),
        )

        # ── 特征变换层：增强特征表达 ──
        self.feature_transform = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True),
        )

        # ── 融合调制层：动态调整融合强度 ──
        self.fusion_modulator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feature_channels, feature_channels // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels // 4, 1, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, ref_feature, src_features, ref_img=None):
        """
        Args:
            ref_feature: 参考视图特征 [B, C, H, W]
            src_features: 源视图特征列表 [[B, C, H, W], ...]
            ref_img: 参考图像 [B, 3, H, W] (可选，用于空间引导)

        Returns:
            enhanced_features: 增强后的特征列表
            attention_weights: 各视图的注意力权重 [B, N, H, W]
        """
        all_features = [ref_feature] + src_features
        num_views = len(all_features)
        B, C, H, W = ref_feature.shape

        # ── Step 1: 视图注意力 - 学习每个视图的重要性 ──
        view_weights = []
        transformed_features = []
        for i, feat in enumerate(all_features):
            # 计算视图重要性
            w = self.view_attention(feat)  # [B, 1, H, W]
            view_weights.append(w)

            # 特征变换
            transformed_feat = self.feature_transform(feat)
            transformed_features.append(transformed_feat)

        # 归一化视图权重 (softmax)
        view_weights = torch.stack(view_weights, dim=1)  # [B, N, H, W]
        view_weights_softmax = F.softmax(view_weights, dim=1)

        # ── Step 2: 空间引导注意力 (如果提供了参考图像) ──
        spatial_guide_weight = None
        if ref_img is not None:
            spatial_guide_weight = self.spatial_guide(ref_img)  # [B, 1, H_img, W_img]
            spatial_guide_weight = torch.sigmoid(spatial_guide_weight)
            # 插值到特征分辨率，使空间引导与特征尺度匹配
            spatial_guide_weight = F.interpolate(
                spatial_guide_weight, size=(H, W), mode='bilinear', align_corners=False
            )

        # Step 3: feature reweighting
        raw_weights = []
        for i, transformed_feat in enumerate(transformed_features):
            w = view_weights_softmax[:, i]  # [B, 1, H, W]

            if spatial_guide_weight is not None:
                w = w * (1.0 + 0.5 * spatial_guide_weight)

            modulator = self.fusion_modulator(transformed_feat)  # [B, 1, 1, 1]
            raw_weights.append(w * modulator)

        fused_weights = torch.stack(raw_weights, dim=1)  # [B, N, 1, H, W]
        fused_weights = fused_weights / (fused_weights.sum(dim=1, keepdim=True) + 1e-8)

        enhanced_features = []
        for i, transformed_feat in enumerate(transformed_features):
            enhanced_feat = transformed_feat * fused_weights[:, i]
            enhanced_features.append(enhanced_feat)

        return enhanced_features, fused_weights[:, :, 0], spatial_guide_weight

    def compute_attention_cost_volume(self, enhanced_features, proj_matrices, depth_values, ref_proj):
        """
        使用注意力加权构建增强的 cost volume

        Args:
            enhanced_features: 注意力增强后的特征列表
            proj_matrices: 投影矩阵列表
            depth_values: 深度假设值 [B, D]
            ref_proj: 参考视图投影矩阵

        Returns:
            attn_cost_volume: 增强后的代价体 [B, C, D, H, W]
            depth_attn_weights: 深度注意力权重 [B, 1, D, H, W]
        """
        ref_feature = enhanced_features[0]
        B, C, H, W = ref_feature.shape
        D = depth_values.shape[1]

        # 构建参考体
        ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, D, 1, 1)  # [B, C, D, H, W]
        volume_sum = ref_volume
        volume_sq_sum = ref_volume ** 2

        # 对每个源视图进行 warping 和加权融合
        for idx in range(1, len(enhanced_features)):
            src_feat = enhanced_features[idx]
            src_proj = proj_matrices[idx]

            # 计算投影变换
            src_proj_new = src_proj[:, 0].clone()
            src_proj_new[:, :3, :4] = torch.matmul(src_proj[:, 1, :3, :3], src_proj[:, 0, :3, :4])
            ref_proj_new = ref_proj[:, 0].clone()
            ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])

            # 变形
            warped_volume = homo_warping(src_feat, src_proj_new, ref_proj_new, depth_values)

            # 累积
            volume_sum = volume_sum + warped_volume
            volume_sq_sum = volume_sq_sum + warped_volume ** 2

        # ── 深度注意力加权 ──
        # 计算初始 variance cost volume
        num_views = len(enhanced_features)
        cost_volume = volume_sq_sum.div_(num_views).sub_(volume_sum.div_(num_views).pow_(2))

        # 应用深度注意力
        depth_attn = self.depth_attention(cost_volume)  # [B, 1, D, H, W]
        depth_attn_weights = torch.sigmoid(depth_attn)

        # 注意力加权的 cost volume
        attn_cost_volume = cost_volume * depth_attn_weights

        return attn_cost_volume, depth_attn_weights


class CrossScaleAttentionBlock(nn.Module):
    """跨尺度注意力模块 v2 - 渐进式门控融合多尺度特征

    改进点：
    1. 无过压缩：保持各尺度通道数，通过 1x1 卷积自适应融合
    2. 渐进式融合：从小尺度(粗粒度)向大尺度(细粒度)逐步注入跨尺度信息
    3. 门控机制：学习跨尺度信息的重要性，避免干扰原始特征
    4. 位置编码：保留空间对应关系
    5. 残差稳定：所有分支都有 skip connection

    输入: features_dict = {"stage1": feat1, "stage2": feat2, "stage3": feat3}
    输出: 增强后的 features_dict
    """
    def __init__(self, feat_channels, reduction=2):
        super().__init__()
        self.feat_channels = feat_channels  # [32, 16, 8]
        self.reduction = reduction

        # 小容量信息提取层 (各尺度独立)
        # 使用 reduction=2 避免过度压缩
        proj_channels = [ch // reduction for ch in feat_channels]
        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, proj_ch, 1, bias=False),
                nn.BatchNorm2d(proj_ch),
                nn.ReLU(inplace=True)
            ) for ch, proj_ch in zip(feat_channels, proj_channels)
        ])

        # 跨尺度融合层 (双向：从粗到细 + 从细到粗)
        # stage3(小) → stage2(中)
        self.fuse_3to2 = nn.Sequential(
            nn.Conv2d(proj_channels[2] + proj_channels[1], proj_channels[1], 3, padding=1, bias=False),
            nn.BatchNorm2d(proj_channels[1]),
            nn.ReLU(inplace=True),
        )
        # stage2 + stage3 → stage1(大)
        self.fuse_2to1 = nn.Sequential(
            nn.Conv2d(proj_channels[1] + proj_channels[0], proj_channels[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(proj_channels[0]),
            nn.ReLU(inplace=True),
        )
        # stage1 → stage2 (反向引导)
        self.fuse_1to2 = nn.Sequential(
            nn.Conv2d(proj_channels[0] + proj_channels[1], proj_channels[1], 3, padding=1, bias=False),
            nn.BatchNorm2d(proj_channels[1]),
            nn.ReLU(inplace=True),
        )
        # stage1 + stage2 → stage3 (反向引导)
        self.fuse_1to3 = nn.Sequential(
            nn.Conv2d(proj_channels[0] + proj_channels[2], proj_channels[2], 3, padding=1, bias=False),
            nn.BatchNorm2d(proj_channels[2]),
            nn.ReLU(inplace=True),
        )

        # 门控层 (控制跨尺度信息的重要程度)
        self.gate_3to2 = nn.Sequential(
            nn.Conv2d(proj_channels[2] + proj_channels[1], 1, 1, bias=False),
            nn.Sigmoid()
        )
        self.gate_2to1 = nn.Sequential(
            nn.Conv2d(proj_channels[1] + proj_channels[0], 1, 1, bias=False),
            nn.Sigmoid()
        )
        self.gate_1to2 = nn.Sequential(
            nn.Conv2d(proj_channels[0] + proj_channels[1], 1, 1, bias=False),
            nn.Sigmoid()
        )
        self.gate_1to3 = nn.Sequential(
            nn.Conv2d(proj_channels[0] + proj_channels[2], 1, 1, bias=False),
            nn.Sigmoid()
        )

        # 输出投影层
        self.out_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(proj_ch, ch, 1, bias=False),
                nn.BatchNorm2d(ch)
            ) for ch, proj_ch in zip(feat_channels, proj_channels)
        ])

        # Conservative residual gating keeps the branch close to baseline early in training.
        self.stage_residual_logits = nn.Parameter(torch.full((len(feat_channels),), -2.0))

    def _upsample_cat(self, small_feat, big_feat):
        upsampled = F.interpolate(small_feat, size=big_feat.shape[2:], mode='bilinear', align_corners=False)
        return torch.cat([upsampled, big_feat], dim=1)

    def forward(self, features_dict):
        feats = [features_dict[f"stage{i+1}"] for i in range(len(self.feat_channels))]
        proj_feats = [self.proj[i](feats[i]) for i in range(len(feats))]

        # 前向路径：stage3 → stage2 → stage1（粗到细，补充大范围上下文）
        fuse_3to2_input = self._upsample_cat(proj_feats[2], proj_feats[1])
        cross_2_from_3 = self.fuse_3to2(fuse_3to2_input)
        gate_3to2 = self.gate_3to2(fuse_3to2_input)
        proj_feats[1] = proj_feats[1] + gate_3to2 * cross_2_from_3

        fuse_2to1_input = self._upsample_cat(proj_feats[1], proj_feats[0])
        cross_1_from_2 = self.fuse_2to1(fuse_2to1_input)
        gate_2to1 = self.gate_2to1(fuse_2to1_input)
        proj_feats[0] = proj_feats[0] + gate_2to1 * cross_1_from_2

        # 反向路径：stage1 → stage2 → stage3（细到粗，补充细节信息）
        proj_0_down = F.interpolate(proj_feats[0], size=feats[1].shape[2:], mode='bilinear', align_corners=False)
        fuse_1to2_input = torch.cat([proj_0_down, proj_feats[1]], dim=1)
        cross_2_from_1 = self.fuse_1to2(fuse_1to2_input)
        gate_1to2 = self.gate_1to2(fuse_1to2_input)
        proj_feats[1] = proj_feats[1] + gate_1to2 * cross_2_from_1

        proj_0_down2 = F.interpolate(proj_feats[0], size=feats[2].shape[2:], mode='bilinear', align_corners=False)
        fuse_1to3_input = torch.cat([proj_0_down2, proj_feats[2]], dim=1)
        cross_3_from_1 = self.fuse_1to3(fuse_1to3_input)
        gate_1to3 = self.gate_1to3(fuse_1to3_input)
        proj_feats[2] = proj_feats[2] + gate_1to3 * cross_3_from_1

        outputs = {}
        residual_scales = torch.sigmoid(self.stage_residual_logits)
        for i in range(len(proj_feats)):
            enhanced = self.out_proj[i](proj_feats[i])
            outputs[f"stage{i+1}"] = feats[i] + residual_scales[i] * enhanced

        return outputs


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


def depth_regression(p, depth_values):
    if depth_values.dim() <= 2:
        # print("regression dim <= 2")
        depth_values = depth_values.view(*depth_values.shape, 1, 1)
    depth = torch.sum(p * depth_values, 1)

    return depth

def cas_mvsnet_loss(inputs, depth_gt_ms, mask_ms, **kwargs):
    depth_loss_weights = kwargs.get("dlossw", None)

    total_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)

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

    return total_loss, depth_loss


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
