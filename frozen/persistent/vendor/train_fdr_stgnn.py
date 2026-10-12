"""Train the duration-aware dynamic graph model with optional reliability input."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


NO_DURATION_VARIANTS = {"no_duration", "no_duration_target_gate"}
NO_SOURCE_RELIABILITY_VARIANTS = {
    "no_reliability_graph",
    "graph_base",
    "target_staleness_gate",
    "no_duration_target_gate",
    "no_self_graph",
    "observed_source_mask",
    "conditional_reliability_graph",
    "conditional_reliability_no_self",
    "spatial_repair",
    "spatial_repair_no_aux",
    "spatial_repair_binary",
    "node_norm_spatial_repair",
    "node_norm_spatial_repair_no_aux",
    "graph_no_duration_channel",
    "graph_no_reliability_channel",
    "graph_mask_only",
    "graph_staleness_only",
    "graph_duration_staleness",
    "graph_duration_staleness_gate",
}
NO_RECONSTRUCTION_VARIANTS = {
    "no_reconstruction",
    "graph_base",
    "target_staleness_gate",
    "no_duration_target_gate",
    "no_self_graph",
    "observed_source_mask",
    "conditional_reliability_graph",
    "conditional_reliability_no_self",
    "spatial_repair",
    "spatial_repair_no_aux",
    "spatial_repair_binary",
    "node_norm_spatial_repair",
    "node_norm_spatial_repair_no_aux",
    "graph_no_duration_channel",
    "graph_no_reliability_channel",
    "graph_mask_only",
    "graph_staleness_only",
    "graph_duration_staleness",
    "graph_duration_staleness_gate",
}
TARGET_STALENESS_VARIANTS = {"target_staleness_gate", "no_duration_target_gate"}
NO_SELF_GRAPH_VARIANTS = {"no_self_graph", "conditional_reliability_no_self"}
CONDITIONAL_RELIABILITY_VARIANTS = {
    "conditional_reliability_graph",
    "conditional_reliability_no_self",
}
SPATIAL_REPAIR_VARIANTS = {
    "spatial_repair",
    "spatial_repair_no_aux",
    "spatial_repair_binary",
    "node_norm_spatial_repair",
    "node_norm_spatial_repair_no_aux",
}
BINARY_SPATIAL_REPAIR_VARIANTS = {"spatial_repair_binary"}
NODE_NORMALIZED_REPAIR_VARIANTS = {
    "node_norm_spatial_repair",
    "node_norm_spatial_repair_no_aux",
}
NO_SPATIAL_AUX_VARIANTS = {
    "spatial_repair_no_aux",
    "node_norm_spatial_repair_no_aux",
}
NO_DURATION_CHANNEL_VARIANTS = {"graph_no_duration_channel", "graph_mask_only"}
NO_RELIABILITY_CHANNEL_VARIANTS = {"graph_no_reliability_channel", "graph_mask_only"}
STALENESS_ONLY_VARIANTS = {"graph_staleness_only"}
DURATION_STALENESS_VARIANTS = {
    "graph_duration_staleness",
    "graph_duration_staleness_gate",
}
STALENESS_VARIANTS = STALENESS_ONLY_VARIANTS | DURATION_STALENESS_VARIANTS
STALENESS_GATE_VARIANTS = {"graph_duration_staleness_gate"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def elapsed_steps(observed: np.ndarray) -> np.ndarray:
    delta = np.zeros_like(observed, dtype=np.float32)
    for step in range(1, len(observed)):
        delta[step] = np.where(observed[step], 0.0, delta[step - 1] + 1.0)
    return delta


def forward_fill(normalized: np.ndarray, observed: np.ndarray) -> np.ndarray:
    filled = np.empty_like(normalized, dtype=np.float32)
    last = np.zeros(normalized.shape[1], dtype=np.float32)
    for step in range(len(normalized)):
        last[observed[step]] = normalized[step, observed[step]]
        filled[step] = last
    return filled


class ForecastWindowDataset(Dataset):
    def __init__(
        self,
        flow: np.ndarray,
        observed: np.ndarray,
        mean: float,
        std: float,
        input_steps: int,
        output_steps: int,
        max_delta_scale: float,
    ) -> None:
        normalized = (flow.astype(np.float32) - mean) / std
        filled = forward_fill(normalized, observed)
        delta = elapsed_steps(observed)
        delta = np.minimum(delta / max(input_steps, 1), max_delta_scale).astype(np.float32)
        self.inputs = np.stack(
            [filled, observed.astype(np.float32), delta],
            axis=-1,
        )
        self.complete_history = normalized.astype(np.float32)
        self.targets = normalized.astype(np.float32)
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.length = len(flow) - input_steps - output_steps + 1
        if self.length <= 0:
            raise ValueError("Split is too short for the requested windows")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        end = index + self.input_steps
        x = torch.from_numpy(self.inputs[index:end])
        history_target = torch.from_numpy(self.complete_history[index:end])
        future_target = torch.from_numpy(self.targets[end : end + self.output_steps])
        return x, history_target, future_target


class FDRSTGNN(nn.Module):
    """Fault-duration-aware reliability-gated spatio-temporal graph network."""

    def __init__(
        self,
        nodes: int,
        hidden_dim: int,
        embedding_dim: int,
        graph_dim: int,
        output_steps: int,
        layers: int,
        top_k: int,
        dropout: float,
        graph_dropout: float,
        graph_temperature: float,
        variant: str,
        node_mean_normalized: np.ndarray,
        node_std_normalized: np.ndarray,
    ) -> None:
        super().__init__()
        self.nodes = nodes
        self.hidden_dim = hidden_dim
        self.output_steps = output_steps
        self.top_k = top_k
        self.graph_dropout = graph_dropout
        self.graph_temperature = graph_temperature
        self.variant = variant
        self.last_adjacency: torch.Tensor | None = None
        self.last_repair_gate: torch.Tensor | None = None
        self.last_spatial_estimate: torch.Tensor | None = None
        self.last_duration_channel: torch.Tensor | None = None
        self.last_reliability_channel: torch.Tensor | None = None
        self.register_buffer(
            "node_mean_normalized",
            torch.as_tensor(node_mean_normalized, dtype=torch.float32),
        )
        self.register_buffer(
            "node_std_normalized",
            torch.as_tensor(node_std_normalized, dtype=torch.float32),
        )

        self.node_embedding = nn.Embedding(nodes, embedding_dim)
        self.decay_head = nn.Linear(embedding_dim, 1)
        temporal_input_dim = 5 + embedding_dim
        self.temporal_encoder = nn.GRU(
            temporal_input_dim,
            hidden_dim,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.reconstruction_head = nn.Linear(hidden_dim, 1)

        self.static_query = nn.Linear(embedding_dim, graph_dim, bias=False)
        self.static_key = nn.Linear(embedding_dim, graph_dim, bias=False)
        self.dynamic_query = nn.Linear(hidden_dim, graph_dim, bias=False)
        self.dynamic_key = nn.Linear(hidden_dim, graph_dim, bias=False)
        self.repair_query = nn.Linear(embedding_dim, graph_dim, bias=False)
        self.repair_key = nn.Linear(embedding_dim, graph_dim, bias=False)
        self.repair_rate_raw = nn.Parameter(torch.zeros(nodes))
        self.graph_value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.message_projection = nn.Linear(hidden_dim, hidden_dim)
        self.fusion_gate = nn.Linear(hidden_dim * 2 + 1, hidden_dim)
        self.target_staleness_scale = nn.Parameter(torch.tensor(-1.0))
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.forecast_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_steps),
        )

    def repair_history(
        self,
        values: torch.Tensor,
        observed: torch.Tensor,
        delta: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query = self.repair_query(embeddings)
        key = self.repair_key(embeddings)
        logits = torch.matmul(query, key.transpose(0, 1)) / math.sqrt(query.shape[-1])
        diagonal = torch.eye(self.nodes, dtype=torch.bool, device=logits.device)
        logits = logits.masked_fill(diagonal, -1e9)
        k = min(max(self.top_k, 1), max(self.nodes - 1, 1))
        selected_values, selected_indices = torch.topk(logits, k=k, dim=-1)
        sparse_logits = torch.full_like(logits, -1e9)
        sparse_logits.scatter_(-1, selected_indices, selected_values)
        adjacency = torch.softmax(sparse_logits / self.graph_temperature, dim=-1)

        batch, history, nodes = values.shape
        flat_values = values.reshape(batch * history, nodes)
        flat_observed = observed.reshape(batch * history, nodes)
        if self.variant in NODE_NORMALIZED_REPAIR_VARIANTS:
            source_values = (
                (flat_values - self.node_mean_normalized[None, :])
                / self.node_std_normalized[None, :].clamp_min(1e-4)
            )
        else:
            source_values = flat_values
        weighted_values = (source_values * flat_observed) @ adjacency.transpose(0, 1)
        observed_weight = flat_observed @ adjacency.transpose(0, 1)
        estimate = weighted_values / observed_weight.clamp_min(1e-6)
        if self.variant in NODE_NORMALIZED_REPAIR_VARIANTS:
            estimate = (
                estimate * self.node_std_normalized[None, :]
                + self.node_mean_normalized[None, :]
            )
        estimate = torch.where(observed_weight > 1e-6, estimate, flat_values)
        estimate = estimate.reshape(batch, history, nodes)

        missing = 1.0 - observed
        if self.variant in BINARY_SPATIAL_REPAIR_VARIANTS:
            gate = missing
        else:
            rate = F.softplus(self.repair_rate_raw) + 1e-4
            gate = missing * (1.0 - torch.exp(-delta * rate[None, None, :]))
        repaired = values + gate * (estimate - values)
        return repaired, estimate, gate

    def reliability(self, observed: torch.Tensor, delta: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        alpha = F.softplus(self.decay_head(embeddings)).squeeze(-1) + 1e-4
        if self.variant in NO_DURATION_VARIANTS:
            effective_delta = (~observed.bool()).to(delta.dtype)
        else:
            effective_delta = delta
        decay = torch.exp(-effective_delta * alpha[None, None, :])
        return observed + (1.0 - observed) * decay

    def dynamic_graph(
        self,
        hidden: torch.Tensor,
        source_reliability: torch.Tensor,
        source_observed: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        static_q = self.static_query(embeddings)
        static_k = self.static_key(embeddings)
        static_logits = torch.matmul(static_q, static_k.transpose(0, 1)) / math.sqrt(static_q.shape[-1])
        dynamic_q = self.dynamic_query(hidden)
        dynamic_k = self.dynamic_key(hidden)
        dynamic_logits = torch.matmul(dynamic_q, dynamic_k.transpose(1, 2)) / math.sqrt(dynamic_q.shape[-1])
        logits = dynamic_logits + static_logits[None, :, :]
        if self.variant in CONDITIONAL_RELIABILITY_VARIANTS:
            target_missing = 1.0 - source_observed
            source_penalty = torch.log(source_reliability.clamp_min(1e-5))[:, None, :]
            logits = logits + target_missing[:, :, None] * source_penalty
        elif self.variant not in NO_SOURCE_RELIABILITY_VARIANTS:
            logits = logits + torch.log(source_reliability.clamp_min(1e-5))[:, None, :]
        if self.variant == "observed_source_mask":
            target_missing = source_observed < 0.5
            source_missing = source_observed < 0.5
            has_observed_source = source_observed.bool().any(dim=-1)[:, None, None]
            disallowed = (
                target_missing[:, :, None]
                & source_missing[:, None, :]
                & has_observed_source
            )
            logits = logits.masked_fill(disallowed, -1e9)
        if self.variant in NO_SELF_GRAPH_VARIANTS:
            diagonal = torch.eye(self.nodes, dtype=torch.bool, device=logits.device)
            logits = logits.masked_fill(diagonal[None, :, :], -1e9)
        k = min(max(self.top_k, 1), self.nodes)
        values, indices = torch.topk(logits, k=k, dim=-1)
        sparse_logits = torch.full_like(logits, -1e9)
        sparse_logits.scatter_(-1, indices, values)
        adjacency = torch.softmax(sparse_logits / self.graph_temperature, dim=-1)
        return F.dropout(adjacency, p=self.graph_dropout, training=self.training)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: batch, history, nodes, [forward-filled value, mask, elapsed duration]
        batch, history, nodes, _ = x.shape
        if nodes != self.nodes:
            raise ValueError(f"Expected {self.nodes} nodes, received {nodes}")
        values = x[..., 0]
        observed = x[..., 1]
        delta = x[..., 2]
        node_ids = torch.arange(nodes, device=x.device)
        embeddings = self.node_embedding(node_ids)
        reliability = self.reliability(observed, delta, embeddings)
        staleness = (1.0 - observed) * (1.0 - reliability)
        if self.variant in STALENESS_ONLY_VARIANTS:
            duration_channel = staleness
            reliability_channel = torch.zeros_like(reliability)
            fusion_context = staleness
        elif self.variant in DURATION_STALENESS_VARIANTS:
            duration_channel = delta
            reliability_channel = staleness
            fusion_context = staleness
        else:
            duration_channel = (
                torch.zeros_like(delta)
                if self.variant in NO_DURATION_VARIANTS | NO_DURATION_CHANNEL_VARIANTS
                else delta
            )
            reliability_channel = (
                torch.zeros_like(reliability)
                if self.variant in NO_RELIABILITY_CHANNEL_VARIANTS
                else reliability
            )
            fusion_context = reliability_channel
        self.last_duration_channel = duration_channel.detach()
        self.last_reliability_channel = reliability_channel.detach()
        if self.variant in SPATIAL_REPAIR_VARIANTS:
            repaired_values, spatial_estimate, repair_gate = self.repair_history(
                values,
                observed,
                delta,
                embeddings,
            )
        else:
            repaired_values = values
            spatial_estimate = values
            repair_gate = torch.zeros_like(values)
        self.last_repair_gate = repair_gate.detach()
        self.last_spatial_estimate = spatial_estimate.detach()
        repair_delta = repaired_values - values
        temporal_features = torch.stack(
            [repaired_values, repair_delta, observed, duration_channel, reliability_channel],
            dim=-1,
        )
        expanded_embeddings = embeddings[None, None, :, :].expand(batch, history, -1, -1)
        temporal_features = torch.cat([temporal_features, expanded_embeddings], dim=-1)
        sequences = temporal_features.permute(0, 2, 1, 3).reshape(batch * nodes, history, -1)
        encoded, _ = self.temporal_encoder(sequences)
        reconstruction = self.reconstruction_head(encoded).reshape(batch, nodes, history)
        reconstruction = reconstruction.permute(0, 2, 1)
        hidden = encoded[:, -1].reshape(batch, nodes, self.hidden_dim)

        if self.variant == "no_graph":
            self.last_adjacency = None
            fused = hidden
        else:
            reliability_last = reliability[:, -1]
            fusion_context_last = fusion_context[:, -1]
            observed_last = observed[:, -1]
            adjacency = self.dynamic_graph(hidden, reliability_last, observed_last, embeddings)
            self.last_adjacency = adjacency.detach()
            message = torch.matmul(adjacency, self.graph_value(hidden))
            gate_features = torch.cat(
                [hidden, message, fusion_context_last[..., None]], dim=-1
            )
            gate_logits = self.fusion_gate(gate_features)
            if self.variant in STALENESS_GATE_VARIANTS:
                scale = F.softplus(self.target_staleness_scale)
                gate_logits = gate_logits + scale * fusion_context_last[..., None]
            if self.variant in TARGET_STALENESS_VARIANTS:
                if self.variant == "no_duration_target_gate":
                    target_staleness = 1.0 - observed[:, -1]
                else:
                    target_staleness = delta[:, -1]
                scale = F.softplus(self.target_staleness_scale)
                gate_logits = gate_logits + scale * target_staleness[..., None]
            gate = torch.sigmoid(gate_logits)
            fused = self.fusion_norm(hidden + gate * self.message_projection(message))

        base = repaired_values[:, -1]
        prediction = self.forecast_head(fused).permute(0, 2, 1) + base[:, None, :]
        return prediction, reconstruction, reliability, spatial_estimate, repair_gate


def aggregate_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = prediction - target
    absolute = np.abs(error)
    return {
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "wape": float(absolute.sum() / max(np.abs(target).sum(), 1e-8) * 100.0),
    }


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_arguments(args: argparse.Namespace) -> None:
    positive_integer_names = (
        "input_steps",
        "output_steps",
        "hidden_dim",
        "embedding_dim",
        "graph_dim",
        "layers",
        "top_k",
        "batch_size",
        "epochs",
        "patience",
        "latency_batch_size",
        "latency_repetitions",
        "lr_patience",
    )
    for name in positive_integer_names:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.num_workers < 0 or args.latency_warmup < 0:
        raise ValueError("--num-workers and --latency-warmup must be non-negative")
    if args.graph_temperature <= 0 or args.max_delta_scale <= 0:
        raise ValueError("--graph-temperature and --max-delta-scale must be positive")
    if args.reconstruction_weight < 0 or args.spatial_repair_weight < 0 or args.weight_decay < 0:
        raise ValueError("Loss weights and --weight-decay must be non-negative")
    if args.learning_rate <= 0 or args.min_learning_rate <= 0:
        raise ValueError("Learning rates must be positive")
    if not 0.0 < args.lr_factor < 1.0:
        raise ValueError("--lr-factor must be in (0, 1)")
    for name in ("dropout", "graph_dropout"):
        value = getattr(args, name)
        if not 0.0 <= value < 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1)")


def validate_data(
    flow: np.ndarray,
    masks: np.lib.npyio.NpzFile,
    train_end: int,
    val_end: int,
    total: int,
    std: float,
) -> None:
    if flow.ndim != 2:
        raise ValueError(f"Expected flow with shape [time, nodes], received {flow.shape}")
    if total != len(flow) or not 0 < train_end < val_end < total:
        raise ValueError(
            f"Invalid chronological split: train_end={train_end}, val_end={val_end}, "
            f"total={total}, flow_length={len(flow)}"
        )
    if not np.isfinite(flow).all() or not math.isfinite(std) or std <= 0:
        raise ValueError("Flow values and normalization statistics must be finite, with std > 0")
    expected_shapes = {
        "train": (train_end, flow.shape[1]),
        "val": (val_end - train_end, flow.shape[1]),
        "test": (total - val_end, flow.shape[1]),
    }
    for name, expected_shape in expected_shapes.items():
        if name not in masks:
            raise KeyError(f"Mask archive does not contain split '{name}'")
        if masks[name].shape != expected_shape:
            raise ValueError(
                f"Mask shape mismatch for {name}: expected {expected_shape}, "
                f"received {masks[name].shape}"
            )
    if "observed_is_one" in masks and not bool(masks["observed_is_one"].item()):
        raise ValueError("Mask semantics must use observed=1 and missing=0")


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    mean: float,
    std: float,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, dict[str, float]]:
    model.eval()
    predictions, targets = [], []
    reconstruction_absolute = 0.0
    reconstruction_squared = 0.0
    reconstruction_count = 0
    spatial_absolute = 0.0
    spatial_squared = 0.0
    spatial_count = 0
    repair_gate_sum = 0.0
    reliability_sum = 0.0
    reliability_count = 0
    for x, history_target, future_target in loader:
        x = x.to(device)
        prediction, reconstruction, reliability, spatial_estimate, repair_gate = model(x)
        predictions.append(prediction.cpu().numpy())
        targets.append(future_target.numpy())
        missing = x[..., 1] < 0.5
        if missing.any():
            reconstruction_error = (reconstruction - history_target.to(device))[missing] * std
            reconstruction_absolute += float(reconstruction_error.abs().sum().item())
            reconstruction_squared += float(reconstruction_error.square().sum().item())
            reconstruction_count += int(missing.sum().item())
            reliability_sum += float(reliability[missing].sum().item())
            reliability_count += int(missing.sum().item())
            spatial_error = (spatial_estimate - history_target.to(device))[missing] * std
            spatial_absolute += float(spatial_error.abs().sum().item())
            spatial_squared += float(spatial_error.square().sum().item())
            spatial_count += int(missing.sum().item())
            repair_gate_sum += float(repair_gate[missing].sum().item())
    prediction = np.concatenate(predictions) * std + mean
    target = np.concatenate(targets) * std + mean
    reconstruction_metrics = {
        "missing_history_mae": reconstruction_absolute / max(reconstruction_count, 1),
        "missing_history_rmse": math.sqrt(reconstruction_squared / max(reconstruction_count, 1)),
        "missing_history_count": reconstruction_count,
        "mean_missing_reliability": reliability_sum / max(reliability_count, 1),
        "spatial_missing_history_mae": spatial_absolute / max(spatial_count, 1),
        "spatial_missing_history_rmse": math.sqrt(spatial_squared / max(spatial_count, 1)),
        "spatial_missing_history_count": spatial_count,
        "mean_missing_repair_gate": repair_gate_sum / max(spatial_count, 1),
    }
    return aggregate_metrics(prediction, target), prediction, target, reconstruction_metrics


@torch.no_grad()
def measure_latency(
    model: nn.Module,
    sample: torch.Tensor,
    device: torch.device,
    warmup: int,
    repetitions: int,
) -> dict[str, float]:
    model.eval()
    sample = sample.to(device)
    for _ in range(warmup):
        model(sample)
    if device.type == "cuda":
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(repetitions)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(repetitions)]
        for index in range(repetitions):
            starts[index].record()
            model(sample)
            ends[index].record()
        torch.cuda.synchronize()
        times = np.array([start.elapsed_time(end) for start, end in zip(starts, ends)], dtype=np.float64)
    else:
        measurements = []
        for _ in range(repetitions):
            started = time.perf_counter()
            model(sample)
            measurements.append((time.perf_counter() - started) * 1000.0)
        times = np.asarray(measurements)
    return {
        "latency_batch_size": int(len(sample)),
        "latency_mean_ms": float(times.mean()),
        "latency_p50_ms": float(np.quantile(times, 0.50)),
        "latency_p90_ms": float(np.quantile(times, 0.90)),
        "latency_per_sample_ms": float(times.mean() / len(sample)),
        "latency_repetitions": repetitions,
        "latency_warmup": warmup,
    }


@torch.no_grad()
def collect_model_diagnostics(
    model: FDRSTGNN,
    loader: DataLoader,
    device: torch.device,
    input_steps: int,
) -> dict[str, object]:
    model.eval()
    node_ids = torch.arange(model.nodes, device=device)
    embeddings = model.node_embedding(node_ids)
    alpha = (F.softplus(model.decay_head(embeddings)).squeeze(-1) + 1e-4).cpu().numpy()
    repair_rate = (F.softplus(model.repair_rate_raw) + 1e-4).cpu().numpy()
    reliability_sum: dict[int, float] = {}
    reliability_count: dict[int, int] = {}
    repair_gate_sum: dict[int, float] = {}
    repair_gate_count: dict[int, int] = {}
    self_mass_sum = 0.0
    observed_source_mass_sum = 0.0
    graph_target_count = 0
    missing_self_mass_sum = 0.0
    missing_observed_source_mass_sum = 0.0
    missing_target_count = 0
    missing_duration_channel_sum = 0.0
    missing_reliability_channel_sum = 0.0
    missing_channel_count = 0
    for x, _, _ in loader:
        x = x.to(device)
        observed = x[..., 1]
        delta = x[..., 2]
        reliability = model.reliability(observed, delta, embeddings)
        _, _, _, _, repair_gate = model(x)
        missing = observed < 0.5
        if missing.any():
            if model.last_duration_channel is None or model.last_reliability_channel is None:
                raise RuntimeError("Temporal missingness channels were not recorded")
            missing_duration_channel_sum += float(model.last_duration_channel[missing].sum().item())
            missing_reliability_channel_sum += float(
                model.last_reliability_channel[missing].sum().item()
            )
            missing_channel_count += int(missing.sum().item())
        duration_steps = torch.round(delta * input_steps).to(torch.int64)
        for duration in duration_steps[missing].unique().tolist():
            selected = missing & (duration_steps == duration)
            reliability_sum[duration] = reliability_sum.get(duration, 0.0) + float(
                reliability[selected].sum().item()
            )
            reliability_count[duration] = reliability_count.get(duration, 0) + int(selected.sum().item())
            repair_gate_sum[duration] = repair_gate_sum.get(duration, 0.0) + float(
                repair_gate[selected].sum().item()
            )
            repair_gate_count[duration] = repair_gate_count.get(duration, 0) + int(selected.sum().item())
        adjacency = model.last_adjacency
        if adjacency is not None:
            observed_last = observed[:, -1]
            self_mass = adjacency.diagonal(dim1=1, dim2=2)
            observed_source_mass = (adjacency * observed_last[:, None, :]).sum(dim=-1)
            self_mass_sum += float(self_mass.sum().item())
            observed_source_mass_sum += float(observed_source_mass.sum().item())
            graph_target_count += int(self_mass.numel())
            missing_targets = observed_last < 0.5
            if missing_targets.any():
                missing_self_mass_sum += float(self_mass[missing_targets].sum().item())
                missing_observed_source_mass_sum += float(
                    observed_source_mass[missing_targets].sum().item()
                )
                missing_target_count += int(missing_targets.sum().item())
    reliability_by_duration = [
        {
            "duration_steps": int(duration),
            "mean_reliability": reliability_sum[duration] / reliability_count[duration],
            "count": reliability_count[duration],
        }
        for duration in sorted(reliability_sum)
    ]
    repair_gate_by_duration = [
        {
            "duration_steps": int(duration),
            "mean_repair_gate": repair_gate_sum[duration] / repair_gate_count[duration],
            "count": repair_gate_count[duration],
        }
        for duration in sorted(repair_gate_sum)
    ]
    return {
        "alpha_min": float(np.min(alpha)),
        "alpha_mean": float(np.mean(alpha)),
        "alpha_median": float(np.median(alpha)),
        "alpha_p90": float(np.quantile(alpha, 0.9)),
        "alpha_max": float(np.max(alpha)),
        "repair_rate_min": float(np.min(repair_rate)),
        "repair_rate_mean": float(np.mean(repair_rate)),
        "repair_rate_median": float(np.median(repair_rate)),
        "repair_rate_p90": float(np.quantile(repair_rate, 0.9)),
        "repair_rate_max": float(np.max(repair_rate)),
        "target_staleness_scale": float(F.softplus(model.target_staleness_scale).item()),
        "all_target_self_mass": self_mass_sum / max(graph_target_count, 1),
        "all_target_observed_source_mass": observed_source_mass_sum / max(graph_target_count, 1),
        "missing_target_self_mass": missing_self_mass_sum / max(missing_target_count, 1),
        "missing_target_observed_source_mass": (
            missing_observed_source_mass_sum / max(missing_target_count, 1)
        ),
        "graph_target_count": graph_target_count,
        "missing_target_count": missing_target_count,
        "mean_missing_duration_channel": (
            missing_duration_channel_sum / max(missing_channel_count, 1)
        ),
        "mean_missing_reliability_channel": (
            missing_reliability_channel_sum / max(missing_channel_count, 1)
        ),
        "reliability_by_duration": reliability_by_duration,
        "repair_gate_by_duration": repair_gate_by_duration,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["PEMS04", "PEMS08"])
    parser.add_argument("--mask-file", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/fdr_stgnn"))
    parser.add_argument(
        "--variant",
        choices=[
            "full",
            "no_duration",
            "no_reliability_graph",
            "no_graph",
            "no_reconstruction",
            "graph_base",
            "target_staleness_gate",
            "no_duration_target_gate",
            "no_self_graph",
            "observed_source_mask",
            "conditional_reliability_graph",
            "conditional_reliability_no_self",
            "spatial_repair",
            "spatial_repair_no_aux",
            "spatial_repair_binary",
            "node_norm_spatial_repair",
            "node_norm_spatial_repair_no_aux",
            "graph_no_duration_channel",
            "graph_no_reliability_channel",
            "graph_mask_only",
            "graph_staleness_only",
            "graph_duration_staleness",
            "graph_duration_staleness_gate",
        ],
        default="full",
    )
    parser.add_argument("--input-steps", type=int, default=12)
    parser.add_argument("--output-steps", type=int, default=12)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--graph-dim", type=int, default=32)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--graph-dropout", type=float, default=0.1)
    parser.add_argument("--graph-temperature", type=float, default=1.0)
    parser.add_argument("--max-delta-scale", type=float, default=4.0)
    parser.add_argument("--reconstruction-weight", type=float, default=0.1)
    parser.add_argument("--spatial-repair-weight", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--lr-patience", type=int, default=8)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--latency-batch-size", type=int, default=1)
    parser.add_argument("--latency-warmup", type=int, default=30)
    parser.add_argument("--latency-repetitions", type=int, default=200)
    parser.add_argument("--validation-only", action="store_true")
    args = parser.parse_args()
    validate_arguments(args)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    complete = np.load(args.processed_dir / args.dataset / "complete.npz")
    flow = complete["flow"].astype(np.float32)
    train_end, val_end, total = (int(value) for value in complete["split"])
    mean = float(complete["mean"][0])
    std = float(complete["std"][0])
    masks = np.load(args.mask_file)
    validate_data(flow, masks, train_end, val_end, total, std)
    slices = {
        "train": (flow[:train_end], masks["train"].astype(bool)),
        "val": (flow[train_end:val_end], masks["val"].astype(bool)),
        "test": (flow[val_end:total], masks["test"].astype(bool)),
    }
    datasets = {
        name: ForecastWindowDataset(
            values,
            observed,
            mean,
            std,
            args.input_steps,
            args.output_steps,
            args.max_delta_scale,
        )
        for name, (values, observed) in slices.items()
    }
    loaders = {
        name: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=name == "train",
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        for name, dataset in datasets.items()
    }

    model = FDRSTGNN(
        nodes=flow.shape[1],
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        graph_dim=args.graph_dim,
        output_steps=args.output_steps,
        layers=args.layers,
        top_k=args.top_k,
        dropout=args.dropout,
        graph_dropout=args.graph_dropout,
        graph_temperature=args.graph_temperature,
        variant=args.variant,
        node_mean_normalized=(flow[:train_end].mean(axis=0) - mean) / std,
        node_std_normalized=np.maximum(flow[:train_end].std(axis=0) / std, 1e-4),
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
        min_lr=args.min_learning_rate,
    )
    forecast_loss_fn = nn.L1Loss()
    reconstruction_weight = (
        0.0 if args.variant in NO_RECONSTRUCTION_VARIANTS else args.reconstruction_weight
    )
    spatial_repair_weight = (
        args.spatial_repair_weight
        if args.variant in SPATIAL_REPAIR_VARIANTS
        and args.variant not in NO_SPATIAL_AUX_VARIANTS
        else 0.0
    )
    run_name = (
        f"{args.dataset}_{args.variant}_h{args.hidden_dim}_g{args.graph_dim}_"
        f"k{args.top_k}_seed{args.seed}"
    )
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "best.pt"
    history_rows: list[dict[str, object]] = []
    best_val = float("inf")
    stale = 0
    total_training_seconds = 0.0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    print(
        f"device={device}, variant={args.variant}, train_windows={len(datasets['train'])}, "
        f"parameters={parameter_count}"
    )
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        model.train()
        forecast_sum = 0.0
        reconstruction_sum = 0.0
        spatial_repair_sum = 0.0
        total_sum = 0.0
        count = 0
        for x, history_target, future_target in loaders["train"]:
            x = x.to(device)
            history_target = history_target.to(device)
            future_target = future_target.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction, reconstruction, _, spatial_estimate, _ = model(x)
            forecast_loss = forecast_loss_fn(prediction, future_target)
            missing = x[..., 1] < 0.5
            if reconstruction_weight > 0 and missing.any():
                reconstruction_loss = F.l1_loss(reconstruction[missing], history_target[missing])
            else:
                reconstruction_loss = torch.zeros((), device=device)
            if spatial_repair_weight > 0 and missing.any():
                spatial_repair_loss = F.l1_loss(spatial_estimate[missing], history_target[missing])
            else:
                spatial_repair_loss = torch.zeros((), device=device)
            loss = (
                forecast_loss
                + reconstruction_weight * reconstruction_loss
                + spatial_repair_weight * spatial_repair_loss
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            batch_size = len(x)
            forecast_sum += float(forecast_loss.item()) * batch_size
            reconstruction_sum += float(reconstruction_loss.item()) * batch_size
            spatial_repair_sum += float(spatial_repair_loss.item()) * batch_size
            total_sum += float(loss.item()) * batch_size
            count += batch_size
        val_metrics, _, _, val_reconstruction = evaluate(model, loaders["val"], device, mean, std)
        scheduler.step(val_metrics["mae"])
        seconds = time.perf_counter() - started
        total_training_seconds += seconds
        row = {
            "epoch": epoch,
            "train_forecast_loss_norm": forecast_sum / count,
            "train_reconstruction_loss_norm": reconstruction_sum / count,
            "train_spatial_repair_loss_norm": spatial_repair_sum / count,
            "train_total_loss_norm": total_sum / count,
            "val_mae_raw": val_metrics["mae"],
            "val_missing_history_mae": val_reconstruction["missing_history_mae"],
            "val_spatial_missing_history_mae": val_reconstruction[
                "spatial_missing_history_mae"
            ],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": seconds,
        }
        history_rows.append(row)
        print(
            f"epoch={epoch:03d} train_forecast={row['train_forecast_loss_norm']:.4f} "
            f"train_recon={row['train_reconstruction_loss_norm']:.4f} "
            f"train_spatial={row['train_spatial_repair_loss_norm']:.4f} "
            f"val_MAE={row['val_mae_raw']:.4f} lr={row['learning_rate']:.2e} "
            f"seconds={seconds:.1f}"
        )
        if val_metrics["mae"] < best_val - 1e-5:
            best_val = val_metrics["mae"]
            stale = 0
            torch.save(model.state_dict(), best_path)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early_stop epoch={epoch}, best_val_MAE={best_val:.4f}")
                break

    write_rows(run_dir / "history.csv", history_rows)
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    peak_training_memory_mb = (
        float(torch.cuda.max_memory_allocated(device) / (1024**2)) if device.type == "cuda" else 0.0
    )
    base_config = vars(args) | {
        "device": str(device),
        "best_val_mae": best_val,
        "parameter_count": parameter_count,
        "effective_reconstruction_weight": reconstruction_weight,
        "effective_spatial_repair_weight": spatial_repair_weight,
        "total_training_seconds": total_training_seconds,
        "peak_training_memory_mb": peak_training_memory_mb,
    }
    if args.validation_only:
        with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
            json.dump(base_config, handle, indent=2, default=str)
        print(f"validation_only saved: {run_dir}")
        return

    test_metrics, prediction, target, reconstruction_metrics = evaluate(
        model,
        loaders["test"],
        device,
        mean,
        std,
    )
    model_diagnostics = collect_model_diagnostics(
        model,
        loaders["test"],
        device,
        args.input_steps,
    )
    result_rows: list[dict[str, object]] = []
    for horizon in sorted(set([3, 6, 9, args.output_steps])):
        row = {
            "dataset": args.dataset,
            "variant": args.variant,
            "horizon": horizon,
        }
        row.update(aggregate_metrics(prediction[:, horizon - 1], target[:, horizon - 1]))
        result_rows.append(row)
    overall = {
        "dataset": args.dataset,
        "variant": args.variant,
        "horizon": "all",
    }
    overall.update(test_metrics)
    result_rows.append(overall)
    write_rows(run_dir / "test_metrics.csv", result_rows)

    latency_source = next(iter(loaders["test"]))[0]
    latency_sample = latency_source[: args.latency_batch_size]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    latency = measure_latency(
        model,
        latency_sample,
        device,
        args.latency_warmup,
        args.latency_repetitions,
    )
    peak_inference_memory_mb = (
        float(torch.cuda.max_memory_allocated(device) / (1024**2)) if device.type == "cuda" else 0.0
    )
    efficiency = {
        "parameter_count": parameter_count,
        "checkpoint_size_mb": best_path.stat().st_size / (1024**2),
        "peak_training_memory_mb": peak_training_memory_mb,
        "peak_inference_memory_mb": peak_inference_memory_mb,
        "total_training_seconds": total_training_seconds,
    } | latency
    with (run_dir / "efficiency.json").open("w", encoding="utf-8") as handle:
        json.dump(efficiency, handle, indent=2)
    with (run_dir / "reconstruction_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(reconstruction_metrics, handle, indent=2)
    with (run_dir / "model_diagnostics.json").open("w", encoding="utf-8") as handle:
        json.dump(model_diagnostics, handle, indent=2)
    with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(base_config | efficiency, handle, indent=2, default=str)

    for row in result_rows:
        print(
            f"test horizon={str(row['horizon']):>3} MAE={row['mae']:.4f} "
            f"RMSE={row['rmse']:.4f} WAPE={row['wape']:.2f}%"
        )
    print(
        f"latency mean={latency['latency_mean_ms']:.3f}ms "
        f"per_sample={latency['latency_per_sample_ms']:.3f}ms "
        f"peak_inference_memory={peak_inference_memory_mb:.1f}MB"
    )
    print(f"saved: {run_dir}")


if __name__ == "__main__":
    main()
