import torch
import torch.nn as nn
import torch.nn.functional as F
from .module import *
from .modules import ProgressiveResidualFusionViewAttention, ResidualFusionViewAttention, SinglePassReliabilityWeightedViewAttention, WarpedViewAttention

Align_Corners_Range = False


class DepthNet(nn.Module):
    def __init__(self):
        super(DepthNet, self).__init__()

    def _compose_proj(self, proj):
        proj_new = proj[:, 0].clone()
        proj_new[:, :3, :4] = torch.matmul(proj[:, 1, :3, :3], proj[:, 0, :3, :4])
        return proj_new

    def forward(self, features, proj_matrices, depth_values, num_depth, cost_regularization,
                prob_volume_init=None, view_attention=None, prev_view_weights=None, prev_confidence=None,
                feature_reliabilities=None):
        proj_matrices = torch.unbind(proj_matrices, 1)
        assert len(features) == len(proj_matrices)
        num_views = len(features)

        ref_feature = features[0]
        src_features = features[1:]
        ref_feature_reliability = feature_reliabilities[0] if feature_reliabilities is not None else None
        src_feature_reliabilities = feature_reliabilities[1:] if feature_reliabilities is not None else [None] * len(src_features)
        ref_proj = proj_matrices[0]
        src_projs = proj_matrices[1:]

        ref_proj_new = self._compose_proj(ref_proj)
        ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, num_depth, 1, 1)
        volume_sum = ref_volume
        volume_sq_sum = ref_volume ** 2
        del ref_volume

        if view_attention is None:
            for src_fea, src_proj in zip(src_features, src_projs):
                src_proj_new = self._compose_proj(src_proj)
                warped_volume = homo_warping(src_fea, src_proj_new, ref_proj_new, depth_values)
                if self.training:
                    volume_sum = volume_sum + warped_volume
                    volume_sq_sum = volume_sq_sum + warped_volume ** 2
                else:
                    volume_sum += warped_volume
                    volume_sq_sum += warped_volume.pow_(2)
                del warped_volume
            volume_variance = volume_sq_sum.div_(num_views).sub_(volume_sum.div_(num_views).pow_(2))
        elif getattr(view_attention, 'uses_single_pass_weighted_variance', False):
            src_weights = []
            total_weight = ref_feature.new_ones(ref_feature.size(0), 1, 1, ref_feature.size(2), ref_feature.size(3))
            for src_fea, src_proj, src_feature_reliability in zip(src_features, src_projs, src_feature_reliabilities):
                src_proj_new = self._compose_proj(src_proj)
                warped_volume = homo_warping(src_fea, src_proj_new, ref_proj_new, depth_values)
                warped_feature_reliability = None
                src_reliability_score = None
                if getattr(view_attention, 'use_feature_reliability', False) and src_feature_reliability is not None:
                    warped_feature_reliability = homo_warping(src_feature_reliability, src_proj_new, ref_proj_new, depth_values)
                    src_reliability_score = warped_feature_reliability.mean(dim=2)
                raw_score = view_attention.score_volume(
                    ref_feature,
                    warped_volume,
                    ref_reliability=ref_feature_reliability,
                    src_reliability=src_reliability_score,
                )
                weight = view_attention.score_to_weight(
                    raw_score,
                    prev_confidence=prev_confidence,
                    feature_reliability=src_reliability_score,
                )
                src_weights.append(weight)
                weight = weight.unsqueeze(2)
                volume_sum = volume_sum + warped_volume * weight
                volume_sq_sum = volume_sq_sum + warped_volume.pow(2) * weight
                total_weight = total_weight + weight
                del warped_volume
                if warped_feature_reliability is not None:
                    del warped_feature_reliability
            volume_mean = volume_sum / total_weight
            volume_variance = volume_sq_sum / total_weight - volume_mean.pow(2)
            src_weights = torch.cat(src_weights, dim=1) if src_weights else None
        else:
            raw_scores = []
            for src_fea, src_proj in zip(src_features, src_projs):
                src_proj_new = self._compose_proj(src_proj)
                warped_volume = homo_warping(src_fea, src_proj_new, ref_proj_new, depth_values)
                if self.training:
                    volume_sum = volume_sum + warped_volume
                    volume_sq_sum = volume_sq_sum + warped_volume ** 2
                else:
                    volume_sum += warped_volume
                    volume_sq_sum += warped_volume.pow_(2)
                raw_scores.append(view_attention.score_volume(ref_feature, warped_volume))
                del warped_volume

            base_variance = volume_sq_sum.div_(num_views).sub_(volume_sum.div_(num_views).pow_(2))
            raw_scores = torch.cat(raw_scores, dim=1)
            if getattr(view_attention, 'supports_propagation', False):
                src_weights = view_attention.normalize_scores(raw_scores, prev_weights=prev_view_weights)
            else:
                src_weights = view_attention.normalize_scores(raw_scores)

            att_volume_sum = ref_feature.new_zeros(ref_feature.size(0), ref_feature.size(1), num_depth,
                                                   ref_feature.size(2), ref_feature.size(3))
            att_volume_sq_sum = ref_feature.new_zeros(ref_feature.size(0), ref_feature.size(1), num_depth,
                                                      ref_feature.size(2), ref_feature.size(3))
            total_weight = ref_feature.new_zeros(ref_feature.size(0), 1, 1, ref_feature.size(2), ref_feature.size(3))
            ref_weight = ref_feature.new_ones(ref_feature.size(0), 1, 1, ref_feature.size(2), ref_feature.size(3))
            ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, num_depth, 1, 1)
            att_volume_sum = att_volume_sum + ref_volume * ref_weight
            att_volume_sq_sum = att_volume_sq_sum + ref_volume.pow(2) * ref_weight
            total_weight = total_weight + ref_weight
            del ref_volume

            for src_fea, src_proj, weight in zip(src_features, src_projs, torch.unbind(src_weights, dim=1)):
                src_proj_new = self._compose_proj(src_proj)
                warped_volume = homo_warping(src_fea, src_proj_new, ref_proj_new, depth_values)
                weight = weight.unsqueeze(1).unsqueeze(2)
                att_volume_sum = att_volume_sum + warped_volume * weight
                att_volume_sq_sum = att_volume_sq_sum + warped_volume.pow(2) * weight
                total_weight = total_weight + weight
                del warped_volume

            attentive_variance = att_volume_sq_sum.div(total_weight).sub_(att_volume_sum.div(total_weight).pow_(2))
            if getattr(view_attention, 'uses_baseline_fusion', False):
                volume_variance = view_attention.fuse_variance(ref_feature, base_variance, attentive_variance, src_weights)
            else:
                volume_variance = attentive_variance

        cost_reg = cost_regularization(volume_variance)
        prob_volume_pre = cost_reg.squeeze(1)

        if prob_volume_init is not None:
            prob_volume_pre += prob_volume_init

        prob_volume = F.softmax(prob_volume_pre, dim=1)
        depth = depth_regression(prob_volume, depth_values=depth_values)

        with torch.no_grad():
            prob_volume_sum4 = 4 * F.avg_pool3d(
                F.pad(prob_volume.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2)),
                (4, 1, 1),
                stride=1,
                padding=0,
            ).squeeze(1)
            depth_index = depth_regression(
                prob_volume,
                depth_values=torch.arange(num_depth, device=prob_volume.device, dtype=torch.float),
            ).long()
            depth_index = depth_index.clamp(min=0, max=num_depth - 1)
            photometric_confidence = torch.gather(prob_volume_sum4, 1, depth_index.unsqueeze(1)).squeeze(1)

            cost_mean = volume_variance.mean(dim=1, keepdim=True)
            cost_std = ((volume_variance - cost_mean) ** 2).mean(dim=1, keepdim=True).clamp(min=1e-6)
            cost_std = cost_std.squeeze(1)
            depth_idx_expanded = depth_index.unsqueeze(1)
            local_std = torch.gather(cost_std, 1, depth_idx_expanded.clamp(min=0, max=num_depth - 1)).squeeze(1)
            local_std_norm = local_std / (local_std.mean(dim=[1, 2], keepdim=True) + 1e-6)
            texture_confidence = photometric_confidence * torch.sigmoid(local_std_norm - 1.0)

        outputs = {
            'depth': depth,
            'photometric_confidence': photometric_confidence,
            'texture_confidence': texture_confidence,
        }
        if view_attention is not None:
            outputs['view_weights'] = src_weights.detach() if 'src_weights' in locals() and src_weights is not None else None
        return outputs

class CascadeMVSNet(nn.Module):
    def __init__(self, refine=False, ndepths=[48, 32, 8], depth_interals_ratio=[4, 2, 1], share_cr=False,
                 grad_method='detach', arch_mode='fpn', cr_base_chs=[8, 8, 8], use_view_attention=False,
                 view_attention_mode='legacy', use_rafe=False):
        super(CascadeMVSNet, self).__init__()
        self.refine = refine
        self.share_cr = share_cr
        self.ndepths = ndepths
        self.depth_interals_ratio = depth_interals_ratio
        self.grad_method = grad_method
        self.arch_mode = arch_mode
        self.cr_base_chs = cr_base_chs
        self.num_stage = len(ndepths)
        self.use_view_attention = use_view_attention
        self.view_attention_mode = view_attention_mode
        self.use_rafe = use_rafe
        print('**********netphs:{}, depth_intervals_ratio:{},  grad:{}, chs:{}, view_attention:{}, mode:{}, rafe:{}************'.format(
              ndepths, depth_interals_ratio, self.grad_method, self.cr_base_chs, use_view_attention, view_attention_mode, use_rafe))

        assert len(ndepths) == len(depth_interals_ratio)

        self.stage_infos = {
            'stage1': {'scale': 4.0},
            'stage2': {'scale': 2.0},
            'stage3': {'scale': 1.0},
        }

        self.feature = FeatureNet(base_channels=8, stride=4, num_stage=self.num_stage,
                                  arch_mode=self.arch_mode, use_rafe=self.use_rafe)

        if self.share_cr:
            self.cost_regularization = CostRegNet(in_channels=self.feature.out_channels, base_channels=8)
        else:
            self.cost_regularization = nn.ModuleList([
                CostRegNet(in_channels=self.feature.out_channels[i], base_channels=self.cr_base_chs[i])
                for i in range(self.num_stage)
            ])

        if self.refine:
            self.refine_network = RefineNet()

        self.DepthNet = DepthNet()
        if self.use_view_attention:
            if self.view_attention_mode == 'legacy':
                self.view_attention_modules = nn.ModuleList([
                    WarpedViewAttention(ch) for ch in self.feature.out_channels[:-1]
                ])
            elif self.view_attention_mode == 'residual_fusion':
                self.view_attention_modules = nn.ModuleList([
                    ResidualFusionViewAttention(ch, max_residual_ratio=0.25 if idx == 0 else 0.5)
                    for idx, ch in enumerate(self.feature.out_channels[:-1])
                ])
            elif self.view_attention_mode == 'progressive_residual_fusion':
                self.view_attention_modules = nn.ModuleList([
                    ProgressiveResidualFusionViewAttention(ch, max_residual_ratio=0.25 if idx == 0 else 0.5,
                                                          max_propagation_ratio=0.35 if idx == 0 else 0.55)
                    for idx, ch in enumerate(self.feature.out_channels[:-1])
                ])
            elif self.view_attention_mode == 'single_pass_reliability_weighted':
                self.view_attention_modules = nn.ModuleList([
                    SinglePassReliabilityWeightedViewAttention(
                        ch,
                        max_weight_delta=0.25 if idx == 0 else 0.35,
                        use_feature_reliability=self.use_rafe,
                    )
                    for idx, ch in enumerate(self.feature.out_channels)
                ])
            else:
                raise ValueError('unsupported view_attention_mode: {}'.format(self.view_attention_mode))

    def forward(self, imgs, proj_matrices, depth_values):
        depth_min = float(depth_values[0, 0].cpu().numpy())
        depth_max = float(depth_values[0, -1].cpu().numpy())
        depth_interval = (depth_max - depth_min) / depth_values.size(1)

        features = []
        for nview_idx in range(imgs.size(1)):
            img = imgs[:, nview_idx]
            features.append(self.feature(img))

        outputs = {}
        depth, cur_depth = None, None
        propagated_view_weights = None
        propagated_confidence = None

        for stage_idx in range(self.num_stage):
            features_stage = [feat['stage{}'.format(stage_idx + 1)] for feat in features]
            feature_reliabilities_stage = None
            reliability_key = 'stage{}_reliability'.format(stage_idx + 1)
            if self.use_rafe and all(reliability_key in feat for feat in features):
                feature_reliabilities_stage = [feat[reliability_key] for feat in features]
            proj_matrices_stage = proj_matrices['stage{}'.format(stage_idx + 1)]
            stage_scale = self.stage_infos['stage{}'.format(stage_idx + 1)]['scale']

            if depth is not None:
                if self.grad_method == 'detach':
                    cur_depth = depth.detach()
                else:
                    cur_depth = depth
                cur_depth = F.interpolate(
                    cur_depth.unsqueeze(1),
                    [img.shape[2], img.shape[3]],
                    mode='bilinear',
                    align_corners=Align_Corners_Range,
                ).squeeze(1)
            else:
                cur_depth = depth_values

            depth_range_samples = get_depth_range_samples(
                cur_depth=cur_depth,
                ndepth=self.ndepths[stage_idx],
                depth_inteval_pixel=self.depth_interals_ratio[stage_idx] * depth_interval,
                dtype=img[0].dtype,
                device=img[0].device,
                shape=[img.shape[0], img.shape[2], img.shape[3]],
                max_depth=depth_max,
                min_depth=depth_min,
            )

            view_attention = None
            if self.use_view_attention and stage_idx < len(self.view_attention_modules):
                view_attention = self.view_attention_modules[stage_idx]

            stage_prev_view_weights = None
            if propagated_view_weights is not None and view_attention is not None:
                stage_prev_view_weights = F.interpolate(
                    propagated_view_weights,
                    size=features_stage[0].shape[-2:],
                    mode='nearest'
                )

            stage_prev_confidence = None
            if propagated_confidence is not None and view_attention is not None and getattr(view_attention, 'supports_confidence_guidance', False):
                stage_prev_confidence = F.interpolate(
                    propagated_confidence.unsqueeze(1),
                    size=features_stage[0].shape[-2:],
                    mode='bilinear',
                    align_corners=Align_Corners_Range,
                ).squeeze(1).unsqueeze(1)

            outputs_stage = self.DepthNet(
                features_stage,
                proj_matrices_stage,
                depth_values=F.interpolate(
                    depth_range_samples.unsqueeze(1),
                    [self.ndepths[stage_idx], img.shape[2] // int(stage_scale), img.shape[3] // int(stage_scale)],
                    mode='trilinear',
                    align_corners=Align_Corners_Range,
                ).squeeze(1),
                num_depth=self.ndepths[stage_idx],
                cost_regularization=self.cost_regularization if self.share_cr else self.cost_regularization[stage_idx],
                view_attention=view_attention,
                prev_view_weights=stage_prev_view_weights,
                prev_confidence=stage_prev_confidence,
                feature_reliabilities=feature_reliabilities_stage,
            )

            depth = outputs_stage['depth']
            if outputs_stage.get('view_weights') is not None:
                propagated_view_weights = outputs_stage['view_weights']
            if outputs_stage.get('photometric_confidence') is not None:
                propagated_confidence = outputs_stage['photometric_confidence'].detach()
            outputs['stage{}'.format(stage_idx + 1)] = outputs_stage
            outputs.update(outputs_stage)

        if self.refine:
            refined_depth = self.refine_network(torch.cat((imgs[:, 0], depth), 1))
            outputs['refined_depth'] = refined_depth

        return outputs
