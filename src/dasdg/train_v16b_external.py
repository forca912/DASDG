"""Run frozen external baselines on the independent v16B test split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


MODELS = (
    "gru_value",
    "gru_mask_duration",
    "stgcn_corr",
    "dcrnn_corr",
    "graph_wavenet_adaptive",
)

MODEL_INPUT_CHANNELS = {
    "gru_value": ["causal_filled_value"],
    "gru_mask_duration": ["causal_filled_value", "observation_mask", "elapsed_missing_duration"],
    "stgcn_corr": ["causal_filled_value", "observation_mask"],
    "dcrnn_corr": ["causal_filled_value", "observation_mask"],
    "graph_wavenet_adaptive": ["causal_filled_value", "observation_mask"],
}


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


def row_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.maximum(matrix.astype(np.float32), 0.0)
    denominator = matrix.sum(axis=1, keepdims=True)
    return matrix / np.maximum(denominator, 1e-8)


def build_correlation_adjacency(
    train_flow: np.ndarray,
    top_k: int,
    observed: np.ndarray | None = None,
) -> np.ndarray:
    available = train_flow.astype(np.float32)
    if observed is not None:
        if observed.shape != available.shape:
            raise ValueError("Observed mask does not match correlation input")
        available = forward_fill(available, observed.astype(bool))
    centered = available.astype(np.float64) - available.mean(axis=0, keepdims=True)
    scale = available.std(axis=0, keepdims=True).astype(np.float64)
    standardized = centered / np.maximum(scale, 1e-6)
    correlation = standardized.T @ standardized / max(len(standardized) - 1, 1)
    correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
    correlation = np.maximum(correlation, 0.0)
    np.fill_diagonal(correlation, 0.0)
    nodes = correlation.shape[0]
    k = min(max(top_k, 1), max(nodes - 1, 1))
    sparse = np.zeros_like(correlation, dtype=np.float32)
    indices = np.argpartition(correlation, -k, axis=1)[:, -k:]
    rows = np.arange(nodes)[:, None]
    sparse[rows, indices] = correlation[rows, indices]
    sparse = np.maximum(sparse, sparse.T)
    sparse += np.eye(nodes, dtype=np.float32)
    return row_normalize(sparse)


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
        self.inputs = np.stack([filled, observed.astype(np.float32), delta], axis=-1)
        self.targets = normalized.astype(np.float32)
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.length = len(flow) - input_steps - output_steps + 1
        if self.length <= 0:
            raise ValueError("Split is too short for the requested windows")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        end = index + self.input_steps
        x = torch.from_numpy(self.inputs[index:end])
        y = torch.from_numpy(self.targets[end : end + self.output_steps])
        return x, y


class SharedNodeGRU(nn.Module):
    def __init__(
        self,
        nodes: int,
        input_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        output_steps: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.node_embedding = nn.Embedding(nodes, embedding_dim)
        self.gru = nn.GRU(
            input_dim + embedding_dim,
            hidden_dim,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_steps),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, history, nodes, _ = x.shape
        values = x[..., : self.input_dim].permute(0, 2, 1, 3)
        node_ids = torch.arange(nodes, device=x.device)
        embeddings = self.node_embedding(node_ids)[None, :, None, :].expand(batch, -1, history, -1)
        sequences = torch.cat([values, embeddings], dim=-1).reshape(batch * nodes, history, -1)
        encoded, _ = self.gru(sequences)
        prediction = self.head(encoded[:, -1]).reshape(batch, nodes, -1).permute(0, 2, 1)
        return prediction + x[:, -1, :, 0][:, None, :]


class CausalGatedTemporal(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.filter_conv = nn.Conv2d(input_dim, output_dim, kernel_size=(kernel_size, 1))
        self.gate_conv = nn.Conv2d(input_dim, output_dim, kernel_size=(kernel_size, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        padded = F.pad(x, (0, 0, self.kernel_size - 1, 0))
        return self.dropout(torch.tanh(self.filter_conv(padded)) * torch.sigmoid(self.gate_conv(padded)))


class STGCNCorrelation(nn.Module):
    def __init__(
        self,
        adjacency: np.ndarray,
        hidden_dim: int,
        output_steps: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.register_buffer("adjacency", torch.as_tensor(adjacency, dtype=torch.float32))
        self.temporal_in = CausalGatedTemporal(2, hidden_dim, 3, dropout)
        self.graph_projection = nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1)
        self.temporal_out = CausalGatedTemporal(hidden_dim, hidden_dim, 3, dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, output_steps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = x[..., :2].permute(0, 3, 1, 2)
        hidden = self.temporal_in(features)
        propagated = torch.einsum("nm,bctm->bctn", self.adjacency, hidden)
        hidden = F.relu(self.graph_projection(torch.cat([hidden, propagated], dim=1)))
        hidden = self.temporal_out(hidden)[:, :, -1].permute(0, 2, 1)
        hidden = self.norm(hidden)
        prediction = self.head(hidden).permute(0, 2, 1)
        return prediction + x[:, -1, :, 0][:, None, :]


class DiffusionGRUCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, supports: list[np.ndarray], steps: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.steps = steps
        for index, support in enumerate(supports):
            self.register_buffer(f"support_{index}", torch.as_tensor(support, dtype=torch.float32))
        self.support_count = len(supports)
        multiplier = 1 + self.support_count * steps
        feature_dim = (input_dim + hidden_dim) * multiplier
        self.gates = nn.Linear(feature_dim, hidden_dim * 2)
        self.candidate = nn.Linear(feature_dim, hidden_dim)

    def diffusion(self, values: torch.Tensor) -> torch.Tensor:
        features = [values]
        for index in range(self.support_count):
            support = getattr(self, f"support_{index}")
            state = values
            for _ in range(self.steps):
                state = torch.einsum("nm,bmf->bnf", support, state)
                features.append(state)
        return torch.cat(features, dim=-1)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        gate_features = self.diffusion(torch.cat([x, hidden], dim=-1))
        reset, update = torch.sigmoid(self.gates(gate_features)).chunk(2, dim=-1)
        candidate_features = self.diffusion(torch.cat([x, reset * hidden], dim=-1))
        candidate = torch.tanh(self.candidate(candidate_features))
        return update * hidden + (1.0 - update) * candidate


class DCRNNCorrelation(nn.Module):
    def __init__(
        self,
        adjacency: np.ndarray,
        hidden_dim: int,
        output_steps: int,
        diffusion_steps: int,
        dropout: float,
    ) -> None:
        super().__init__()
        forward = row_normalize(adjacency)
        backward = row_normalize(adjacency.T)
        self.cell = DiffusionGRUCell(2, hidden_dim, [forward, backward], diffusion_steps)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, output_steps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, history, nodes, _ = x.shape
        hidden = torch.zeros(batch, nodes, self.cell.hidden_dim, device=x.device, dtype=x.dtype)
        for step in range(history):
            hidden = self.cell(x[:, step, :, :2], hidden)
        prediction = self.head(self.dropout(hidden)).permute(0, 2, 1)
        return prediction + x[:, -1, :, 0][:, None, :]


class GraphWaveNetAdaptive(nn.Module):
    def __init__(
        self,
        nodes: int,
        adjacency: np.ndarray,
        hidden_dim: int,
        embedding_dim: int,
        output_steps: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.register_buffer("fixed_adjacency", torch.as_tensor(adjacency, dtype=torch.float32))
        self.node_source = nn.Parameter(torch.randn(nodes, embedding_dim) * 0.1)
        self.node_target = nn.Parameter(torch.randn(embedding_dim, nodes) * 0.1)
        self.start = nn.Conv2d(2, hidden_dim, kernel_size=1)
        self.dilations = (1, 2, 4)
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.graph_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        for dilation in self.dilations:
            self.filter_convs.append(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 2), dilation=(1, dilation))
            )
            self.gate_convs.append(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 2), dilation=(1, dilation))
            )
            self.graph_convs.append(nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1))
            self.residual_convs.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1))
            self.skip_convs.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1))
        self.dropout = nn.Dropout(dropout)
        self.end_one = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.end_two = nn.Conv2d(hidden_dim, output_steps, kernel_size=1)

    def adjacency(self) -> torch.Tensor:
        adaptive = torch.softmax(F.relu(self.node_source @ self.node_target), dim=-1)
        mixed = 0.5 * adaptive + 0.5 * self.fixed_adjacency
        return mixed / mixed.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.start(x[..., :2].permute(0, 3, 2, 1))
        skip: torch.Tensor | None = None
        adjacency = self.adjacency()
        for index, dilation in enumerate(self.dilations):
            residual = hidden
            padded = F.pad(hidden, (dilation, 0, 0, 0))
            gated = torch.tanh(self.filter_convs[index](padded)) * torch.sigmoid(
                self.gate_convs[index](padded)
            )
            propagated = torch.einsum("nm,bcmt->bcnt", adjacency, gated)
            graph_hidden = self.graph_convs[index](torch.cat([gated, propagated], dim=1))
            graph_hidden = self.dropout(graph_hidden)
            hidden = self.residual_convs[index](graph_hidden) + residual
            current_skip = self.skip_convs[index](graph_hidden)
            skip = current_skip if skip is None else skip + current_skip
        output = F.relu(skip)
        output = F.relu(self.end_one(output))
        prediction = self.end_two(output)[..., -1]
        return prediction + x[:, -1, :, 0][:, None, :]


def build_model(args: argparse.Namespace, nodes: int, adjacency: np.ndarray) -> nn.Module:
    if args.model == "gru_value":
        return SharedNodeGRU(
            nodes, 1, args.hidden_dim, args.embedding_dim, args.output_steps,
            args.layers, args.dropout,
        )
    if args.model == "gru_mask_duration":
        return SharedNodeGRU(
            nodes, 3, args.hidden_dim, args.embedding_dim, args.output_steps,
            args.layers, args.dropout,
        )
    if args.model == "stgcn_corr":
        return STGCNCorrelation(adjacency, args.hidden_dim, args.output_steps, args.dropout)
    if args.model == "dcrnn_corr":
        return DCRNNCorrelation(
            adjacency, args.hidden_dim, args.output_steps, args.diffusion_steps, args.dropout
        )
    if args.model == "graph_wavenet_adaptive":
        return GraphWaveNetAdaptive(
            nodes, adjacency, args.hidden_dim, args.embedding_dim,
            args.output_steps, args.dropout,
        )
    raise ValueError(args.model)


def aggregate_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = prediction - target
    absolute = np.abs(error)
    return {
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "wape": float(absolute.sum() / max(np.abs(target).sum(), 1e-8) * 100.0),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    mean: float,
    std: float,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    predictions, targets = [], []
    for x, y in loader:
        predictions.append(model(x.to(device)).cpu().numpy())
        targets.append(y.numpy())
    prediction = np.concatenate(predictions) * std + mean
    target = np.concatenate(targets) * std + mean
    return aggregate_metrics(prediction, target), prediction, target


def metric_rows(
    model_name: str,
    dataset: str,
    prediction: np.ndarray,
    target: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for horizon in sorted(set([3, 6, 9, prediction.shape[1]])):
        row: dict[str, object] = {"dataset": dataset, "model": model_name, "horizon": horizon}
        row.update(aggregate_metrics(prediction[:, horizon - 1], target[:, horizon - 1]))
        rows.append(row)
    overall: dict[str, object] = {"dataset": dataset, "model": model_name, "horizon": "all"}
    overall.update(aggregate_metrics(prediction, target))
    rows.append(overall)
    return rows


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def measure_latency(
    model: nn.Module,
    sample: torch.Tensor,
    device: torch.device,
    warmup: int,
    repetitions: int,
) -> dict[str, float | int]:
    model.eval()
    sample = sample.to(device)
    for _ in range(warmup):
        model(sample)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    times = []
    for _ in range(repetitions):
        started = time.perf_counter()
        model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append((time.perf_counter() - started) * 1000.0)
    array = np.asarray(times, dtype=np.float64)
    return {
        "latency_batch_size": len(sample),
        "latency_mean_ms": float(array.mean()),
        "latency_p90_ms": float(np.quantile(array, 0.90)),
        "latency_per_sample_ms": float(array.mean() / len(sample)),
        "latency_repetitions": repetitions,
        "latency_warmup": warmup,
    }


def validate_inputs(
    flow: np.ndarray,
    masks: np.lib.npyio.NpzFile,
    train_end: int,
    val_end: int,
    total: int,
    std: float,
) -> None:
    if flow.ndim != 2 or not np.isfinite(flow).all():
        raise ValueError("Flow must be a finite [time, node] array")
    if std <= 0 or not np.isfinite(std):
        raise ValueError("Invalid training standard deviation")
    expected = {
        "train": (train_end, flow.shape[1]),
        "val": (val_end - train_end, flow.shape[1]),
        "test": (total - val_end, flow.shape[1]),
    }
    for split, shape in expected.items():
        if split not in masks or masks[split].shape != shape:
            raise ValueError(f"Mask {split} shape mismatch: expected {shape}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["PEMS04", "PEMS08"])
    parser.add_argument("--mask-file", type=Path, required=True)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/v16b_primary"))
    parser.add_argument("--input-steps", type=int, default=12)
    parser.add_argument("--output-steps", type=int, default=12)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--diffusion-steps", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-delta-scale", type=float, default=4.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--latency-warmup", type=int, default=10)
    parser.add_argument("--latency-repetitions", type=int, default=30)
    parser.add_argument("--independent-test", action="store_true")
    args = parser.parse_args()

    if args.epochs <= 0 or args.patience <= 0 or args.batch_size <= 0:
        raise ValueError("epochs, patience, and batch size must be positive")
    if not args.independent_test:
        raise RuntimeError("v16B requires the explicit --independent-test flag")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    complete = np.load(args.processed_dir / args.dataset / "complete.npz")
    flow = complete["flow"].astype(np.float32)
    train_end, val_end, total = (int(value) for value in complete["split"])
    mean = float(complete["mean"][0])
    std = float(complete["std"][0])
    masks = np.load(args.mask_file)
    validate_inputs(flow, masks, train_end, val_end, total, std)
    normalized_train = (flow[:train_end] - mean) / std
    adjacency = build_correlation_adjacency(
        normalized_train,
        args.top_k,
        masks["train"].astype(bool),
    )
    adjacency_hash = hashlib.sha256(adjacency.tobytes()).hexdigest()

    slices = {
        "train": (flow[:train_end], masks["train"].astype(bool)),
        "val": (flow[train_end:val_end], masks["val"].astype(bool)),
        "test": (flow[val_end:total], masks["test"].astype(bool)),
    }
    datasets = {
        name: ForecastWindowDataset(
            values, observed, mean, std, args.input_steps, args.output_steps, args.max_delta_scale
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

    model = build_model(args, flow.shape[1], adjacency).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(args.patience // 3, 2), min_lr=1e-5
    )
    loss_fn = nn.L1Loss()
    run_name = (
        f"{args.dataset}_{args.model}_h{args.hidden_dim}_lr{args.learning_rate:g}_"
        f"d{args.dropout:g}_seed{args.seed}"
    )
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        run_dir / "adjacency.npz",
        adjacency=adjacency,
        source=np.array("masked_training_split_causal_fill_positive_correlation_topk"),
        sha256=np.array(adjacency_hash),
    )
    best_path = run_dir / "best.pt"
    history_rows: list[dict[str, object]] = []
    best_val = float("inf")
    stale = 0
    total_training_seconds = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    print(
        f"device={device}, dataset={args.dataset}, model={args.model}, "
        f"train_windows={len(datasets['train'])}, val_windows={len(datasets['val'])}, "
        f"parameters={parameter_count}, adjacency=train_correlation"
    )
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        model.train()
        total_loss, count = 0.0, 0
        for x, y in loaders["train"]:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            loss = loss_fn(prediction, y)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(x)
            count += len(x)
        val_metrics, _, _ = evaluate(model, loaders["val"], device, mean, std)
        elapsed = time.perf_counter() - started
        total_training_seconds += elapsed
        scheduler.step(val_metrics["mae"])
        row = {
            "epoch": epoch,
            "train_loss_norm": total_loss / max(count, 1),
            "val_mae_raw": val_metrics["mae"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": elapsed,
        }
        history_rows.append(row)
        print(
            f"epoch={epoch:03d} train_loss_norm={row['train_loss_norm']:.5f} "
            f"val_MAE_raw={row['val_mae_raw']:.5f} seconds={elapsed:.1f}"
        )
        if val_metrics["mae"] < best_val - 1e-5:
            best_val = val_metrics["mae"]
            stale = 0
            torch.save(model.state_dict(), best_path)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early_stop epoch={epoch}, best_val_MAE={best_val:.5f}")
                break

    write_rows(run_dir / "history.csv", history_rows)
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    _, val_prediction, val_target = evaluate(model, loaders["val"], device, mean, std)
    write_rows(
        run_dir / "validation_metrics.csv",
        metric_rows(args.model, args.dataset, val_prediction, val_target),
    )

    _, test_prediction, test_target = evaluate(model, loaders["test"], device, mean, std)
    write_rows(
        run_dir / "test_metrics.csv",
        metric_rows(args.model, args.dataset, test_prediction, test_target),
    )

    latency_source = next(iter(loaders["test"]))[0][:1]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    latency = measure_latency(
        model, latency_source, device, args.latency_warmup, args.latency_repetitions
    )
    peak_inference_memory_mb = (
        float(torch.cuda.max_memory_allocated(device) / (1024**2)) if device.type == "cuda" else 0.0
    )
    efficiency = {
        "parameter_count": parameter_count,
        "checkpoint_size_mb": best_path.stat().st_size / (1024**2),
        "total_training_seconds": total_training_seconds,
        "peak_inference_memory_mb": peak_inference_memory_mb,
    } | latency
    with (run_dir / "efficiency.json").open("w", encoding="utf-8") as handle:
        json.dump(efficiency, handle, indent=2)

    config = vars(args) | {
        "device": str(device),
        "selection_split": "validation",
        "evaluation_split": "independent_test",
        "frozen_from": "v16a_validation_only",
        "best_val_mae": best_val,
        "parameter_count": parameter_count,
        "input_channels": MODEL_INPUT_CHANNELS[args.model],
        "adjacency_source": "masked_training_split_causal_fill_positive_correlation_topk",
        "adjacency_sha256": adjacency_hash,
    } | efficiency
    with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, default=str)

    print(
        f"completed model={args.model}, best_val_MAE={best_val:.5f}, "
        f"latency_per_sample_ms={latency['latency_per_sample_ms']:.4f}, saved={run_dir}"
    )


if __name__ == "__main__":
    main()
