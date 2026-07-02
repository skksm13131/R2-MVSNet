import torch
import torch.nn as nn


class SinglePassReliabilityWeightedViewAttention(nn.Module):
    """SP-RWCV source-view reliability weighting used by the best R2 path."""

    uses_single_pass_weighted_variance = True
    supports_confidence_guidance = True

    def __init__(self, channels, min_hidden_channels=8, max_weight_delta=0.35,
                 confidence_floor=0.35, use_feature_reliability=False):
        super().__init__()
        hidden_channels = max(min_hidden_channels, channels // 4)
        self.use_feature_reliability = use_feature_reliability
        in_channels = channels * 3 + 2 + (2 if self.use_feature_reliability else 0)
        self.score_net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=True),
        )
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.residual_scale_logit = nn.Parameter(torch.tensor(-1.6))
        self.max_weight_delta = max_weight_delta
        self.confidence_floor = confidence_floor
        nn.init.constant_(self.score_net[-1].bias, 0.0)

    def score_volume(self, ref_feature, warped_volume, ref_reliability=None,
                     src_reliability=None):
        ref_volume = ref_feature.unsqueeze(2)
        corr = (warped_volume * ref_volume).mean(dim=1)
        best_corr = corr.max(dim=1, keepdim=True).values
        warped_mean = warped_volume.mean(dim=2)
        abs_diff = (warped_mean - ref_feature).abs()
        valid_mask = (
            warped_volume.abs().mean(dim=1).max(dim=1, keepdim=True).values > 1e-6
        ).to(ref_feature.dtype)
        score_inputs = [ref_feature, warped_mean, abs_diff, best_corr, valid_mask]
        if self.use_feature_reliability:
            if ref_reliability is None:
                ref_reliability = ref_feature.new_full(
                    (ref_feature.size(0), 1, ref_feature.size(2), ref_feature.size(3)),
                    0.5,
                )
            if src_reliability is None:
                src_reliability = ref_reliability.new_full(ref_reliability.shape, 0.5)
            score_inputs.extend([
                ref_reliability.to(ref_feature.dtype),
                src_reliability.to(ref_feature.dtype),
            ])
        return self.score_net(torch.cat(score_inputs, dim=1))

    def score_to_weight(self, raw_score, prev_confidence=None,
                        feature_reliability=None):
        residual_ratio = torch.sigmoid(self.residual_scale_logit) * self.max_weight_delta
        pixel_residual = torch.tanh(raw_score / self.temperature.clamp(min=0.5))
        if prev_confidence is not None:
            confidence = prev_confidence.to(raw_score.dtype).clamp(min=0.0, max=1.0)
            confidence = self.confidence_floor + (1.0 - self.confidence_floor) * confidence
            pixel_residual = pixel_residual * confidence
        if self.use_feature_reliability and feature_reliability is not None:
            feature_reliability = feature_reliability.to(raw_score.dtype).clamp(
                min=0.0, max=1.0
            )
            feature_reliability = (
                self.confidence_floor
                + (1.0 - self.confidence_floor) * feature_reliability
            )
            pixel_residual = pixel_residual * feature_reliability
        return (1.0 + residual_ratio * pixel_residual).clamp(min=0.25, max=2.0)
