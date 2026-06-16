import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class WarpedViewAttention(nn.Module):
    """Legacy per-view, per-pixel attention after homography warping."""

    def __init__(self, channels, min_hidden_channels=8):
        super().__init__()
        hidden_channels = max(min_hidden_channels, channels // 4)
        in_channels = channels * 3 + 2
        self.score_net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=True),
        )
        self.residual_scale_logit = nn.Parameter(torch.tensor(-2.0))
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def score_volume(self, ref_feature, warped_volume):
        ref_volume = ref_feature.unsqueeze(2)
        corr = (warped_volume * ref_volume).mean(dim=1)
        best_corr = corr.max(dim=1, keepdim=True).values
        warped_mean = warped_volume.mean(dim=2)
        abs_diff = (warped_mean - ref_feature).abs()
        valid_mask = (warped_volume.abs().mean(dim=1).max(dim=1, keepdim=True).values > 1e-6).to(ref_feature.dtype)
        score_input = torch.cat([ref_feature, warped_mean, abs_diff, best_corr, valid_mask], dim=1)
        return self.score_net(score_input)

    def normalize_scores(self, raw_scores):
        if raw_scores.numel() == 0:
            return raw_scores
        num_src_views = raw_scores.size(1)
        alpha = torch.sigmoid(self.residual_scale_logit)
        centered_scores = raw_scores - raw_scores.mean(dim=1, keepdim=True)
        score_scale = centered_scores.std(dim=1, keepdim=True).clamp(min=1e-4)
        normalized_scores = centered_scores / score_scale
        attn_weights = F.softmax(normalized_scores / self.temperature.clamp(min=0.5), dim=1)
        uniform_weights = raw_scores.new_full(raw_scores.shape, 1.0 / num_src_views)
        blended_weights = uniform_weights + alpha * (attn_weights - uniform_weights)
        return blended_weights * num_src_views

    def forward(self, ref_feature, warped_volumes):
        if len(warped_volumes) == 0:
            return ref_feature.new_zeros(ref_feature.size(0), 0, ref_feature.size(2), ref_feature.size(3))
        raw_scores = [self.score_volume(ref_feature, warped_volume) for warped_volume in warped_volumes]
        return self.normalize_scores(torch.cat(raw_scores, dim=1))


class SinglePassReliabilityWeightedViewAttention(nn.Module):
    """Single-pass reliability weighting for variance cost-volume construction.

    The module predicts a bounded per-view reliability from each warped source
    volume and uses it immediately while accumulating the variance statistics.
    It avoids the second homography-warping pass used by residual-fusion designs.
    """

    uses_single_pass_weighted_variance = True
    supports_confidence_guidance = True

    def __init__(self, channels, min_hidden_channels=8, max_weight_delta=0.35, confidence_floor=0.35,
                 use_feature_reliability=False):
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

    def score_volume(self, ref_feature, warped_volume, ref_reliability=None, src_reliability=None):
        ref_volume = ref_feature.unsqueeze(2)
        corr = (warped_volume * ref_volume).mean(dim=1)
        best_corr = corr.max(dim=1, keepdim=True).values
        warped_mean = warped_volume.mean(dim=2)
        abs_diff = (warped_mean - ref_feature).abs()
        valid_mask = (warped_volume.abs().mean(dim=1).max(dim=1, keepdim=True).values > 1e-6).to(ref_feature.dtype)
        score_inputs = [ref_feature, warped_mean, abs_diff, best_corr, valid_mask]
        if self.use_feature_reliability:
            if ref_reliability is None:
                ref_reliability = ref_feature.new_full(
                    (ref_feature.size(0), 1, ref_feature.size(2), ref_feature.size(3)), 0.5)
            if src_reliability is None:
                src_reliability = ref_reliability.new_full(ref_reliability.shape, 0.5)
            score_inputs.extend([ref_reliability.to(ref_feature.dtype), src_reliability.to(ref_feature.dtype)])
        score_input = torch.cat(score_inputs, dim=1)
        return self.score_net(score_input)

    def score_to_weight(self, raw_score, prev_confidence=None, feature_reliability=None):
        residual_ratio = torch.sigmoid(self.residual_scale_logit) * self.max_weight_delta
        pixel_residual = torch.tanh(raw_score / self.temperature.clamp(min=0.5))
        if prev_confidence is not None:
            confidence = prev_confidence.to(raw_score.dtype).clamp(min=0.0, max=1.0)
            confidence = self.confidence_floor + (1.0 - self.confidence_floor) * confidence
            pixel_residual = pixel_residual * confidence
        if self.use_feature_reliability and feature_reliability is not None:
            feature_reliability = feature_reliability.to(raw_score.dtype).clamp(min=0.0, max=1.0)
            feature_reliability = self.confidence_floor + (1.0 - self.confidence_floor) * feature_reliability
            pixel_residual = pixel_residual * feature_reliability
        return (1.0 + residual_ratio * pixel_residual).clamp(min=0.25, max=2.0)


class ResidualFusionViewAttention(nn.Module):
    """Reliability-aware residual fusion between baseline and attentive variance."""

    def __init__(self, channels, min_hidden_channels=8, max_residual_ratio=0.5):
        super().__init__()
        hidden_channels = max(min_hidden_channels, channels // 4)
        score_in_channels = channels * 3 + 2
        gate_in_channels = channels + 5

        self.score_net = nn.Sequential(
            nn.Conv2d(score_in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=True),
        )
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(gate_in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=True),
        )
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.residual_scale_logit = nn.Parameter(torch.tensor(-1.8))
        self.max_residual_ratio = max_residual_ratio
        self.uses_baseline_fusion = True
        nn.init.constant_(self.fusion_gate[-1].bias, -2.0)

    def score_volume(self, ref_feature, warped_volume):
        ref_volume = ref_feature.unsqueeze(2)
        corr = (warped_volume * ref_volume).mean(dim=1)
        best_corr = corr.max(dim=1, keepdim=True).values
        warped_mean = warped_volume.mean(dim=2)
        abs_diff = (warped_mean - ref_feature).abs()
        valid_mask = (warped_volume.abs().mean(dim=1).max(dim=1, keepdim=True).values > 1e-6).to(ref_feature.dtype)
        score_input = torch.cat([ref_feature, warped_mean, abs_diff, best_corr, valid_mask], dim=1)
        return self.score_net(score_input)

    def _compute_distribution(self, raw_scores):
        if raw_scores.numel() == 0:
            return raw_scores
        centered_scores = raw_scores - raw_scores.mean(dim=1, keepdim=True)
        score_scale = centered_scores.std(dim=1, keepdim=True).clamp(min=1e-4)
        normalized_scores = centered_scores / score_scale
        attn_weights = F.softmax(normalized_scores / self.temperature.clamp(min=0.7), dim=1)
        num_src_views = raw_scores.size(1)
        uniform_weights = raw_scores.new_full(raw_scores.shape, 1.0 / num_src_views)
        residual_ratio = torch.sigmoid(self.residual_scale_logit) * self.max_residual_ratio
        blended_weights = uniform_weights + residual_ratio * (attn_weights - uniform_weights)
        return blended_weights

    def normalize_scores(self, raw_scores, prev_weights=None):
        if raw_scores.numel() == 0:
            return raw_scores
        num_src_views = raw_scores.size(1)
        return self._compute_distribution(raw_scores) * num_src_views

    def fuse_variance(self, ref_feature, base_variance, attentive_variance, src_weights):
        num_src_views = max(src_weights.size(1), 1)
        normalized_weights = (src_weights / float(num_src_views)).clamp(min=1e-6)
        if num_src_views > 1:
            weight_entropy = -(normalized_weights * normalized_weights.log()).sum(dim=1, keepdim=True) / math.log(num_src_views)
        else:
            weight_entropy = normalized_weights.new_zeros(normalized_weights.size(0), 1, normalized_weights.size(2), normalized_weights.size(3))
        weight_peak = normalized_weights.max(dim=1, keepdim=True).values
        base_stat = base_variance.mean(dim=(1, 2), keepdim=False).unsqueeze(1)
        attentive_stat = attentive_variance.mean(dim=(1, 2), keepdim=False).unsqueeze(1)
        variance_delta = (attentive_variance - base_variance).abs().mean(dim=(1, 2), keepdim=False).unsqueeze(1)
        gate_input = torch.cat([ref_feature, base_stat, attentive_stat, variance_delta, weight_entropy, 1.0 - weight_peak], dim=1)
        fusion_gate = torch.sigmoid(self.fusion_gate(gate_input)).unsqueeze(2)
        uncertainty = (0.5 * weight_entropy + 0.5 * (1.0 - weight_peak)).unsqueeze(2).clamp(min=0.0, max=1.0)
        conservative_gate = fusion_gate * (1.0 - uncertainty)
        return base_variance + conservative_gate * (attentive_variance - base_variance)


class ProgressiveResidualFusionViewAttention(ResidualFusionViewAttention):
    """GoMVS-inspired cross-stage view-reliability propagation with residual variance fusion."""

    def __init__(self, channels, min_hidden_channels=8, max_residual_ratio=0.5, max_propagation_ratio=0.6):
        super().__init__(channels, min_hidden_channels=min_hidden_channels, max_residual_ratio=max_residual_ratio)
        hidden_channels = max(min_hidden_channels, channels // 4)
        self.propagation_gate = nn.Sequential(
            nn.Conv2d(5, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=True),
        )
        nn.init.constant_(self.propagation_gate[-1].bias, -1.5)
        self.max_propagation_ratio = max_propagation_ratio
        self.supports_propagation = True

    def normalize_scores(self, raw_scores, prev_weights=None):
        if raw_scores.numel() == 0:
            return raw_scores

        num_src_views = raw_scores.size(1)
        current_weights = self._compute_distribution(raw_scores)
        if prev_weights is None:
            return current_weights * num_src_views

        prev_weights = prev_weights.to(raw_scores.dtype)
        prev_weights = prev_weights.clamp(min=1e-6)
        prev_weights = prev_weights / prev_weights.sum(dim=1, keepdim=True).clamp(min=1e-6)

        if num_src_views > 1:
            current_entropy = -(current_weights.clamp(min=1e-6) * current_weights.clamp(min=1e-6).log()).sum(dim=1, keepdim=True) / math.log(num_src_views)
            prev_entropy = -(prev_weights * prev_weights.log()).sum(dim=1, keepdim=True) / math.log(num_src_views)
        else:
            zeros = current_weights.new_zeros(current_weights.size(0), 1, current_weights.size(2), current_weights.size(3))
            current_entropy = zeros
            prev_entropy = zeros

        current_peak = current_weights.max(dim=1, keepdim=True).values
        prev_peak = prev_weights.max(dim=1, keepdim=True).values
        agreement = 1.0 - (current_weights - prev_weights).abs().mean(dim=1, keepdim=True)
        agreement = agreement.clamp(min=0.0, max=1.0)
        gate_input = torch.cat([current_entropy, prev_entropy, current_peak, prev_peak, agreement], dim=1)
        propagation_ratio = torch.sigmoid(self.propagation_gate(gate_input)) * self.max_propagation_ratio

        blended_weights = current_weights + propagation_ratio * (prev_weights - current_weights)
        blended_weights = blended_weights / blended_weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
        return blended_weights * num_src_views
