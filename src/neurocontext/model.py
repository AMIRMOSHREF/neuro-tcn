from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class CausalConv1d(nn.Conv1d):
    def __init__(self, *args, dilation: int = 1, **kwargs):
        super().__init__(*args, dilation=dilation, padding=0, **kwargs)
        self.left_padding = dilation * (self.kernel_size[0] - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(F.pad(x, (self.left_padding, 0)))


class ResidualDCC(nn.Module):
    def __init__(self, hidden: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        self.depthwise = CausalConv1d(
            hidden, hidden, kernel, groups=hidden, dilation=dilation
        )
        self.pointwise = nn.Conv1d(hidden, hidden * 2, 1)
        self.norm = nn.GroupNorm(1, hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        value, gate = self.pointwise(self.depthwise(x)).chunk(2, dim=1)
        return self.norm(residual + self.dropout(value * torch.sigmoid(gate)))


class STFTBranch(nn.Module):
    """Compact log-STFT summary computed independently for every neuron."""

    def __init__(self, delay_bins: int, hidden: int):
        super().__init__()
        self.n_fft = min(16, 2 ** int(math.floor(math.log2(delay_bins))))
        self.hop = max(1, self.n_fft // 4)
        self.project = nn.Sequential(
            nn.Linear(self.n_fft // 2 + 1, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original = x.shape[:-1]
        flat = x.reshape(-1, x.shape[-1])
        window = torch.hann_window(self.n_fft, device=x.device, dtype=x.dtype)
        spectrum = torch.stft(
            flat,
            n_fft=self.n_fft,
            hop_length=self.hop,
            window=window,
            return_complex=True,
            center=False,
        )
        energy = torch.log1p(spectrum.abs().square()).mean(dim=-1)
        return self.project(energy).reshape(*original, -1)


class ContextForecaster(nn.Module):
    """DCC-STFT model with sparse neuron and temporal context attention."""

    def __init__(self, config: dict):
        super().__init__()
        model = config["model"]
        hidden = int(model["hidden_dim"])
        self.hidden = hidden
        self.response_bins = int(config["response_bins"])
        self.input_projection = nn.Conv1d(1, hidden, 1)
        self.tcn = nn.Sequential(
            *[
                ResidualDCC(
                    hidden,
                    int(model["kernel_size"]),
                    2**layer,
                    float(model["dropout"]),
                )
                for layer in range(int(model["tcn_layers"]))
            ]
        )
        self.temporal_score = nn.Conv1d(hidden, 1, 1)
        self.stft = STFTBranch(int(config["delay_bins"]), hidden)
        self.unit_fusion = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.GELU(), nn.LayerNorm(hidden)
        )
        self.neuron_score = nn.Linear(hidden, 1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=int(model["attention_heads"]),
            dim_feedforward=hidden * 3,
            dropout=float(model["dropout"]),
            batch_first=True,
            norm_first=True,
        )
        self.region_attention = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.response_decoder = nn.Sequential(
            nn.Linear(hidden * 2, hidden * 2),
            nn.GELU(),
            nn.Dropout(float(model["dropout"])),
            nn.Linear(hidden * 2, self.response_bins),
            nn.Softplus(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 4, hidden * 2),
            nn.GELU(),
            nn.Dropout(float(model["dropout"])),
            nn.Linear(hidden * 2, 3),
        )

    def forward(self, delay: torch.Tensor, unit_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, regions, units, time = delay.shape
        x = delay.reshape(batch * regions * units, 1, time)
        temporal_features = self.tcn(self.input_projection(torch.log1p(x)))
        temporal_logits = self.temporal_score(temporal_features).squeeze(1)
        temporal_attention = torch.softmax(temporal_logits, dim=-1)
        temporal_embedding = torch.sum(
            temporal_features * temporal_attention.unsqueeze(1), dim=-1
        ).reshape(batch, regions, units, self.hidden)
        spectral_embedding = self.stft(torch.log1p(delay))
        unit_embedding = self.unit_fusion(
            torch.cat([temporal_embedding, spectral_embedding], dim=-1)
        )

        neuron_gate = torch.sigmoid(self.neuron_score(unit_embedding).squeeze(-1))
        neuron_gate = neuron_gate * unit_mask.float()
        denominator = neuron_gate.sum(dim=2, keepdim=True).clamp_min(1e-6)
        region_tokens = (unit_embedding * neuron_gate.unsqueeze(-1)).sum(dim=2)
        region_tokens = region_tokens / denominator
        region_context = self.region_attention(region_tokens)

        expanded_context = region_context.unsqueeze(2).expand(-1, -1, units, -1)
        rates = self.response_decoder(
            torch.cat([unit_embedding, expanded_context], dim=-1)
        )
        rates = rates * unit_mask.unsqueeze(-1)
        logits = self.classifier(region_context.reshape(batch, -1))
        return {
            "logits": logits,
            "rates": rates,
            "neuron_gate": neuron_gate,
            "temporal_attention": temporal_attention.reshape(
                batch, regions, units, time
            ),
            "region_attention": region_context,
        }


def multitask_loss(
    output: dict[str, torch.Tensor],
    response: torch.Tensor,
    labels: torch.Tensor,
    unit_mask: torch.Tensor,
    config: dict,
) -> tuple[torch.Tensor, dict[str, float]]:
    weights = config["training"]
    class_loss = F.cross_entropy(output["logits"], labels)
    mask = unit_mask.unsqueeze(-1).expand_as(response)
    forecast_elements = F.poisson_nll_loss(
        output["rates"].clamp_min(1e-6), response, log_input=False, reduction="none"
    )
    forecast_loss = forecast_elements[mask].mean()
    active_gates = output["neuron_gate"][unit_mask]
    sparsity = active_gates.mean()
    attention = output["temporal_attention"]
    temporal_tv = (attention[..., 1:] - attention[..., :-1]).abs()
    temporal_tv = temporal_tv[unit_mask.unsqueeze(-1).expand_as(temporal_tv)].mean()
    total = (
        float(weights["classification_weight"]) * class_loss
        + float(weights["forecast_weight"]) * forecast_loss
        + float(weights["sparsity_weight"]) * sparsity
        + float(weights["temporal_tv_weight"]) * temporal_tv
    )
    metrics = {
        "loss": float(total.detach()),
        "classification_loss": float(class_loss.detach()),
        "forecast_loss": float(forecast_loss.detach()),
        "mean_gate": float(sparsity.detach()),
    }
    return total, metrics
