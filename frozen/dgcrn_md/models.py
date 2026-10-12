"""Controlled adapters for the published implementations, not official upstream baselines."""
from __future__ import annotations

import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from vendor.train_fdr_stgnn import FDRSTGNN
from vendor.train_v16b_external import GraphWaveNetAdaptive, SharedNodeGRU


class GraphWaveNetInputs(GraphWaveNetAdaptive):
    """Only the first convolution and its input slice change for 3-channel input."""
    def __init__(self, *args, input_dim=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_dim = input_dim
        if input_dim != 2:
            self.start = nn.Conv2d(input_dim, self.start.out_channels, kernel_size=1)

    def forward(self, x):
        hidden = self.start(x[..., :self.input_dim].permute(0, 3, 2, 1))
        skip = None
        adjacency = self.adjacency()
        for index, dilation in enumerate(self.dilations):
            residual = hidden
            padded = F.pad(hidden, (dilation, 0, 0, 0))
            gated = torch.tanh(self.filter_convs[index](padded)) * torch.sigmoid(
                self.gate_convs[index](padded))
            propagated = torch.einsum('nm,bcmt->bcnt', adjacency, gated)
            graph_hidden = self.dropout(self.graph_convs[index](torch.cat([gated, propagated], dim=1)))
            hidden = self.residual_convs[index](graph_hidden) + residual
            current_skip = self.skip_convs[index](graph_hidden)
            skip = current_skip if skip is None else skip + current_skip
        prediction = self.end_two(F.relu(self.end_one(F.relu(skip))))[..., -1]
        return prediction + x[:, -1, :, 0][:, None, :]


class RevisionDASDG(FDRSTGNN):
    """Forecast path of graph_no_reliability_channel with single-factor switches.

    Retains the published registered parameters for weight compatibility. Unused
    research heads remain registered but are not executed; report both registered
    and forecast-gradient-active parameter counts, not just the former.
    """
    def __init__(self, *args, graph_mode='combined', gated=True, duration=True, **kwargs):
        super().__init__(*args, variant='graph_no_reliability_channel', **kwargs)
        self.graph_mode = graph_mode
        self.gated = gated
        self.duration = duration

    def dynamic_graph(self, hidden, source_reliability, source_observed, embeddings):
        if self.graph_mode == 'combined':
            return super().dynamic_graph(hidden, source_reliability, source_observed, embeddings)
        if self.graph_mode in ('static', 'dense'):
            static_q = self.static_query(embeddings)
            static_k = self.static_key(embeddings)
            static = (static_q @ static_k.T) / math.sqrt(static_q.shape[-1])
        if self.graph_mode in ('dynamic', 'dense'):
            dq, dk = self.dynamic_query(hidden), self.dynamic_key(hidden)
            dynamic = (dq @ dk.transpose(1, 2)) / math.sqrt(dq.shape[-1])
        if self.graph_mode == 'static':
            logits = static[None].expand(len(hidden), -1, -1)
        elif self.graph_mode == 'dynamic':
            logits = dynamic
        elif self.graph_mode == 'dense':
            logits = dynamic + static[None]
        else:
            raise ValueError(self.graph_mode)
        if self.graph_mode != 'dense':
            values, indices = logits.topk(min(self.top_k, self.nodes), dim=-1)
            logits = torch.full_like(logits, -1e9).scatter(-1, indices, values)
        adjacency = torch.softmax(logits / self.graph_temperature, dim=-1)
        return F.dropout(adjacency, self.graph_dropout, training=self.training)

    def forward(self, x):
        batch, history, nodes, _ = x.shape
        if nodes != self.nodes:
            raise ValueError('Node count mismatch')
        values, observed = x[..., 0], x[..., 1]
        duration = x[..., 2] if self.duration else torch.zeros_like(values)
        embeddings = self.node_embedding(torch.arange(nodes, device=x.device))
        zeros = torch.zeros_like(values)
        features = torch.stack([values, zeros, observed, duration, zeros], dim=-1)
        features = torch.cat([features, embeddings[None, None].expand(batch, history, -1, -1)], dim=-1)
        sequences = features.permute(0, 2, 1, 3).reshape(batch * nodes, history, -1)
        encoded, _ = self.temporal_encoder(sequences)
        hidden = encoded[:, -1].reshape(batch, nodes, self.hidden_dim)
        if self.graph_mode == 'none':
            self.last_adjacency = None
            # Keep LayerNorm: only remove spatial message, not normalization.
            fused = self.fusion_norm(hidden)
        else:
            adjacency = self.dynamic_graph(hidden, zeros[:, -1], observed[:, -1], embeddings)
            self.last_adjacency = adjacency.detach()
            message = adjacency @ self.graph_value(hidden)
            if self.gated:
                gate = torch.sigmoid(self.fusion_gate(torch.cat([hidden, message, zeros[:, -1, :, None]], dim=-1)))
            else:
                gate = 1.0
            fused = self.fusion_norm(hidden + gate * self.message_projection(message))
        return self.forecast_head(fused).permute(0, 2, 1) + values[:, -1, None, :]


def build_model(name, nodes, adjacency, cfg):
    shared = dict(nodes=nodes, hidden_dim=cfg['hidden_dim'], embedding_dim=cfg['embedding_dim'],
                  output_steps=cfg['output_steps'], dropout=cfg['dropout'])
    if name in ('gwn_mask', 'gwn_duration'):
        return GraphWaveNetInputs(**shared, adjacency=adjacency, input_dim=2 if name == 'gwn_mask' else 3)
    if name == 'gru_duration':
        return SharedNodeGRU(**shared, input_dim=3, layers=1)
    switches = {
        'dasdg': {}, 'dasdg_no_duration': {'duration': False},
        'dasdg_no_graph': {'graph_mode': 'none'}, 'dasdg_static': {'graph_mode': 'static'},
        'dasdg_dynamic': {'graph_mode': 'dynamic'}, 'dasdg_dense': {'graph_mode': 'dense'},
        'dasdg_no_gate': {'gated': False},
    }
    if name not in switches:
        raise ValueError(f'Unknown model: {name}')
    return RevisionDASDG(**shared, graph_dim=cfg['graph_dim'], layers=1,
                        top_k=cfg['top_k'], graph_dropout=cfg['graph_dropout'], graph_temperature=1.0,
                        node_mean_normalized=np.zeros(nodes, dtype=np.float32),
                        node_std_normalized=np.ones(nodes, dtype=np.float32), **switches[name])


def input_channels(name):
    base = ['causal_filled_value', 'observation_mask']
    return base if name in ('gwn_mask', 'dasdg_no_duration') else base + ['elapsed_missing_duration']
