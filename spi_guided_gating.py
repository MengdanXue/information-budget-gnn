"""
SPI-Guided Soft Gating Module
============================

Core innovation: Use SPI (Structural Predictability Index) to guide
the fusion of GNN and MLP outputs.

Key formula:
    beta = Sigmoid((SPI - tau) / T)
    Z_final = beta * Z_GCN + (1 - beta) * Z_MLP

Where:
    - SPI = |2h - 1| (Structural Predictability Index)
    - tau = 0.67 (Trust Region threshold)
    - T = Temperature parameter (learnable or fixed)

Author: [Your Name]
Date: 2024-12
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


def compute_edge_homophily(edge_index: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute edge homophily h = fraction of edges connecting same-class nodes.

    Args:
        edge_index: [2, E] edge indices
        labels: [N] node labels

    Returns:
        h: edge homophily in [0, 1]
    """
    src, dst = edge_index[0], edge_index[1]
    same_class = (labels[src] == labels[dst]).float()
    h = same_class.mean().item()
    return h


def compute_spi(h: float) -> float:
    """
    Compute Structural Predictability Index.

    SPI = |2h - 1|

    Properties:
        - SPI = 0 when h = 0.5 (random, no structural signal)
        - SPI = 1 when h = 0 or h = 1 (maximum structural signal)
    """
    return abs(2 * h - 1)


class SPIGuidedGating(nn.Module):
    """
    SPI-Guided Soft Gating Module.

    Uses pre-computed SPI to dynamically weight GNN vs MLP outputs.

    Key innovation: Theory-informed gating (not learned from scratch)
        - Embeds R^2=0.82 correlation finding
        - Uses tau=0.67 threshold from Trust Region analysis
    """

    def __init__(
        self,
        tau: float = 0.67,
        T_init: float = 0.1,
        learnable_T: bool = True,
        learnable_tau: bool = False
    ):
        """
        Args:
            tau: Trust Region threshold (default 0.67 from our analysis)
            T_init: Initial temperature (controls sharpness)
            learnable_T: Whether T is learnable
            learnable_tau: Whether tau is learnable (usually False to preserve theory)
        """
        super().__init__()

        if learnable_tau:
            self.tau = nn.Parameter(torch.tensor(tau))
        else:
            self.register_buffer('tau', torch.tensor(tau))

        if learnable_T:
            # Use log-scale for numerical stability
            self.log_T = nn.Parameter(torch.tensor(np.log(T_init)))
        else:
            self.register_buffer('log_T', torch.tensor(np.log(T_init)))

    @property
    def T(self) -> torch.Tensor:
        """Temperature parameter (always positive via exp)."""
        return torch.exp(self.log_T)

    def compute_beta(self, spi: torch.Tensor) -> torch.Tensor:
        """
        Compute gating coefficient beta.

        beta = Sigmoid((SPI - tau) / T)

        When SPI > tau: beta -> 1 (trust GNN)
        When SPI < tau: beta -> 0 (trust MLP)
        """
        return torch.sigmoid((spi - self.tau) / self.T)

    def forward(
        self,
        z_gnn: torch.Tensor,
        z_mlp: torch.Tensor,
        spi: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: fuse GNN and MLP outputs based on SPI.

        Args:
            z_gnn: [N, D] GNN node embeddings
            z_mlp: [N, D] MLP node embeddings
            spi: scalar or [N] SPI values

        Returns:
            z_fused: [N, D] fused embeddings
            beta: gating coefficient (for visualization)
        """
        beta = self.compute_beta(spi)

        # Handle scalar vs per-node SPI
        if beta.dim() == 0:
            beta = beta.unsqueeze(0)
        if beta.dim() == 1:
            beta = beta.unsqueeze(-1)  # [N, 1] for broadcasting

        z_fused = beta * z_gnn + (1 - beta) * z_mlp

        return z_fused, beta.squeeze()

    def get_interpretable_params(self) -> dict:
        """Return interpretable parameter values."""
        return {
            'tau': self.tau.item(),
            'T': self.T.item(),
            'sharpness': 1.0 / self.T.item()  # Higher = sharper transition
        }


class SPIGuidedGNN(nn.Module):
    """
    Complete SPI-Guided Graph Neural Network.

    Architecture:
        - GNN branch: Standard GCN/GAT for structure-aware features
        - MLP branch: Feature-only MLP (ignores structure)
        - Fusion: SPI-guided soft gating
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        gnn_type: str = 'gcn',
        tau: float = 0.67,
        T_init: float = 0.1,
        learnable_T: bool = True
    ):
        super().__init__()

        self.dropout = dropout
        self.gnn_type = gnn_type

        # GNN branch
        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = in_channels if i == 0 else hidden_channels
            out_dim = hidden_channels if i < num_layers - 1 else out_channels

            if gnn_type == 'gcn':
                from torch_geometric.nn import GCNConv
                self.gnn_layers.append(GCNConv(in_dim, out_dim))
            elif gnn_type == 'gat':
                from torch_geometric.nn import GATConv
                self.gnn_layers.append(GATConv(in_dim, out_dim, heads=4, concat=False))
            elif gnn_type == 'sage':
                from torch_geometric.nn import SAGEConv
                self.gnn_layers.append(SAGEConv(in_dim, out_dim))
            else:
                raise ValueError(f"Unknown GNN type: {gnn_type}")

        # MLP branch (same architecture, no graph structure)
        self.mlp_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = in_channels if i == 0 else hidden_channels
            out_dim = hidden_channels if i < num_layers - 1 else out_channels
            self.mlp_layers.append(nn.Linear(in_dim, out_dim))

        # SPI-guided gating
        self.gating = SPIGuidedGating(
            tau=tau,
            T_init=T_init,
            learnable_T=learnable_T
        )

    def forward_gnn(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward through GNN branch."""
        for i, layer in enumerate(self.gnn_layers):
            x = layer(x, edge_index)
            if i < len(self.gnn_layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward_mlp(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through MLP branch."""
        for i, layer in enumerate(self.mlp_layers):
            x = layer(x)
            if i < len(self.mlp_layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        spi: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass with SPI-guided fusion.

        Args:
            x: [N, F] node features
            edge_index: [2, E] edge indices
            spi: Pre-computed SPI (if None, computed from labels)
            labels: Node labels (for computing SPI if not provided)

        Returns:
            logits: [N, C] class logits
            info: dict with gating info for analysis
        """
        # Compute SPI if not provided
        if spi is None:
            if labels is not None:
                h = compute_edge_homophily(edge_index, labels)
                spi = torch.tensor(compute_spi(h), device=x.device)
            else:
                # Default to tau (boundary case)
                spi = torch.tensor(0.67, device=x.device)

        # GNN branch
        z_gnn = self.forward_gnn(x, edge_index)

        # MLP branch
        z_mlp = self.forward_mlp(x)

        # SPI-guided fusion
        z_fused, beta = self.gating(z_gnn, z_mlp, spi)

        # Info for analysis
        info = {
            'spi': spi.item() if spi.dim() == 0 else spi.mean().item(),
            'beta': beta.mean().item() if beta.dim() > 0 else beta.item(),
            'gating_params': self.gating.get_interpretable_params()
        }

        return z_fused, info


class NumericalBiasModule(nn.Module):
    """
    Numerical Bias Module (extracted from Doc4/Doc5 idea).

    Computes relative numerical features between node pairs
    to provide additional signal when structure is unreliable.

    Features computed:
        - Ratio: v_i / v_j
        - Log ratio: log(v_i / v_j)
        - Absolute difference: |v_i - v_j|
        - Relative difference: |v_i - v_j| / max(v_i, v_j)
    """

    def __init__(
        self,
        numerical_dim: int,
        hidden_dim: int = 32,
        output_dim: int = 1
    ):
        """
        Args:
            numerical_dim: Number of numerical features per node
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (1 for scalar bias)
        """
        super().__init__()

        # Input: 4 relation features per numerical dimension
        relation_dim = numerical_dim * 4

        self.mlp = nn.Sequential(
            nn.Linear(relation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def compute_relation_features(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        eps: float = 1e-8
    ) -> torch.Tensor:
        """
        Compute relation features between node pairs.

        Args:
            x_i: [E, D] features of source nodes
            x_j: [E, D] features of target nodes

        Returns:
            rel: [E, D*4] relation features
        """
        # Ratio (with numerical stability)
        ratio = x_i / (x_j + eps)

        # Log ratio
        log_ratio = torch.log(x_i + eps) - torch.log(x_j + eps)

        # Absolute difference
        abs_diff = torch.abs(x_i - x_j)

        # Relative difference
        max_val = torch.maximum(x_i, x_j) + eps
        rel_diff = abs_diff / max_val

        # Concatenate
        rel = torch.cat([ratio, log_ratio, abs_diff, rel_diff], dim=-1)

        return rel

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        numerical_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute numerical bias for each edge.

        Args:
            x: [N, F] node features
            edge_index: [2, E] edge indices
            numerical_mask: [F] boolean mask for numerical features

        Returns:
            edge_bias: [E] numerical bias per edge
        """
        src, dst = edge_index[0], edge_index[1]

        # Extract numerical features
        if numerical_mask is not None:
            x_num = x[:, numerical_mask]
        else:
            x_num = x

        x_i = x_num[src]  # [E, D]
        x_j = x_num[dst]  # [E, D]

        # Compute relation features
        rel = self.compute_relation_features(x_i, x_j)

        # MLP to scalar bias
        edge_bias = self.mlp(rel).squeeze(-1)

        return edge_bias


# ============================================================
# Ablation Study Models
# ============================================================

class StructureOnlyGNN(nn.Module):
    """Baseline: Standard GNN (no fusion)."""

    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2, dropout=0.5, gnn_type='gcn'):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dropout = dropout

        for i in range(num_layers):
            in_dim = in_channels if i == 0 else hidden_channels
            out_dim = hidden_channels if i < num_layers - 1 else out_channels

            if gnn_type == 'gcn':
                from torch_geometric.nn import GCNConv
                self.layers.append(GCNConv(in_dim, out_dim))
            elif gnn_type == 'gat':
                from torch_geometric.nn import GATConv
                self.layers.append(GATConv(in_dim, out_dim, heads=4, concat=False))

    def forward(self, x, edge_index):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class FeatureOnlyMLP(nn.Module):
    """Baseline: MLP (ignores structure)."""

    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2, dropout=0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dropout = dropout

        for i in range(num_layers):
            in_dim = in_channels if i == 0 else hidden_channels
            out_dim = hidden_channels if i < num_layers - 1 else out_channels
            self.layers.append(nn.Linear(in_dim, out_dim))

    def forward(self, x, edge_index=None):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class NaiveFusionGNN(nn.Module):
    """Baseline: Simple average fusion (no SPI guidance)."""

    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2, dropout=0.5, fusion_weight=0.5):
        super().__init__()
        self.gnn = StructureOnlyGNN(in_channels, hidden_channels, out_channels,
                                     num_layers, dropout)
        self.mlp = FeatureOnlyMLP(in_channels, hidden_channels, out_channels,
                                   num_layers, dropout)
        self.fusion_weight = fusion_weight

    def forward(self, x, edge_index):
        z_gnn = self.gnn(x, edge_index)
        z_mlp = self.mlp(x)
        return self.fusion_weight * z_gnn + (1 - self.fusion_weight) * z_mlp


# ============================================================
# Visualization Utilities
# ============================================================

def plot_gating_curve(
    gating_module: SPIGuidedGating,
    save_path: Optional[str] = None
):
    """
    Plot the SPI-to-beta gating curve.

    Shows how the model transitions from trusting MLP (low SPI)
    to trusting GNN (high SPI).
    """
    import matplotlib.pyplot as plt

    spi_values = torch.linspace(0, 1, 100)

    with torch.no_grad():
        beta_values = gating_module.compute_beta(spi_values).numpy()

    params = gating_module.get_interpretable_params()
    tau = params['tau']

    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot curve
    ax.plot(spi_values.numpy(), beta_values, 'b-', linewidth=2, label='Gating curve')

    # Mark threshold
    ax.axvline(x=tau, color='r', linestyle='--', linewidth=1.5, label=f'tau={tau:.2f}')
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.7)

    # Shade regions
    ax.axvspan(0, tau, alpha=0.1, color='red', label='MLP dominant')
    ax.axvspan(tau, 1, alpha=0.1, color='green', label='GNN dominant')

    # Labels
    ax.set_xlabel('SPI (Structural Predictability Index)', fontsize=12)
    ax.set_ylabel('beta (GNN weight)', fontsize=12)
    ax.set_title(f'SPI-Guided Gating Curve (T={params["T"]:.3f})', fontsize=14)
    ax.legend(loc='lower right')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close()
    return fig


if __name__ == "__main__":
    # Demo usage
    print("SPI-Guided Gating Module Demo")
    print("="*50)

    # Create gating module
    gating = SPIGuidedGating(tau=0.67, T_init=0.1, learnable_T=True)
    print(f"Initial params: {gating.get_interpretable_params()}")

    # Test with different SPI values
    test_spi = torch.tensor([0.0, 0.3, 0.5, 0.67, 0.8, 1.0])
    betas = gating.compute_beta(test_spi)

    print("\nSPI -> Beta mapping:")
    for s, b in zip(test_spi.tolist(), betas.tolist()):
        trust = "GNN" if b > 0.5 else "MLP"
        print(f"  SPI={s:.2f} -> beta={b:.4f} (trust {trust})")

    # Plot gating curve
    from pathlib import Path
    output_dir = Path(__file__).parent / "figures"
    output_dir.mkdir(exist_ok=True)
    plot_gating_curve(gating, output_dir / "gating_curve.png")

    print("\nDemo complete!")
