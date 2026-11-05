"""SIGReg : Sketched Isotropic Gaussian Regularization (Balestriero & LeCun, 2025).

Pousse la distribution des embeddings vers N(0, I). Par Cramér-Wold, il suffit que
toutes les projections 1D soient N(0, 1) : on tire `num_slices` directions uniformes
sur la sphère, renouvelées à chaque appel, et on applique à chaque projection le test
d'Epps-Pulley, qui compare la fonction caractéristique empirique à exp(-t²/2) :

    EP = N ∫ |φ_N(t) - exp(-t²/2)|² w(t) dt,   w(t) = exp(-t²/2)

L'intégrande est paire, d'où l'intégration sur [0, domain] seulement (règle des
trapèzes, `knots` points). Le gradient est borné et la complexité linéaire en N.
"""

import torch
import torch.nn as nn
from torch import Tensor


class SIGReg(nn.Module):
    """(V, N, D) -> statistique d'Epps-Pulley moyennée sur les V vues et les directions."""

    def __init__(self, knots: int = 17, num_slices: int = 256, domain: float = 3.0):
        super().__init__()
        self.num_slices = num_slices
        t = torch.linspace(0, domain, knots)
        dt = domain / (knots - 1)
        trapezoid = torch.full((knots,), 2 * dt)
        trapezoid[[0, -1]] = dt
        phi = torch.exp(-t.square() / 2)  # fonction caractéristique de N(0, 1)
        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        self.register_buffer("weights", trapezoid * phi)  # quadrature x poids w(t)

    def forward(self, z: Tensor) -> Tensor:
        directions = torch.randn(z.size(-1), self.num_slices, device=z.device)
        directions = directions / directions.norm(dim=0)
        x_t = (z @ directions).unsqueeze(-1) * self.t  # (V, N, S, T)

        # |φ_N(t) - φ(t)|², φ_N(t) = E[cos tx] + i E[sin tx]
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()  # (V, S, T)
        return ((err @ self.weights) * z.size(-2)).mean()
