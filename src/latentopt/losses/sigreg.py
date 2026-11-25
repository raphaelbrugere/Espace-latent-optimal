"""SIGReg : Sketched Isotropic Gaussian Regularization (Balestriero & LeCun, 2025).

Pousse la distribution des embeddings vers N(0, I). Par Cramer-Wold, il suffit que
toutes les projections 1D soient N(0, 1) : on tire `num_slices` directions uniformes
sur la sphere, renouvelees a chaque appel, et on applique a chaque projection le test
d'Epps-Pulley, qui compare la fonction caracteristique empirique a exp(-t^2/2) :

    EP = N * int |phi_N(t) - exp(-t^2/2)|^2 w(t) dt,   w(t) = exp(-t^2/2)

L'integrande est paire, d'ou l'integration sur [0, domain] seulement (regle des
trapezes, `knots` points). Le gradient est borne et la complexite lineaire en N.
"""

import torch
import torch.nn as nn
from torch import Tensor


class SIGReg(nn.Module):
    """(V, N, D) -> statistique d'Epps-Pulley moyennee sur les V vues et les directions."""

    def __init__(self, knots: int = 17, num_slices: int = 256, domain: float = 3.0) -> None:
        """`knots` : points de la grille t (discretisation de l'integrale, regle des trapezes).
        `num_slices` : directions 1D aleatoires. `domain` : domaine d'integration, t dans
        [0, domain]."""
        super().__init__()
        self.num_slices = num_slices
        t = torch.linspace(0, domain, knots)
        dt = domain / (knots - 1)
        trapezoid = torch.full((knots,), 2 * dt)
        trapezoid[[0, -1]] = dt
        phi = torch.exp(-t.square() / 2)  # fonction caracteristique de N(0, 1)
        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        self.register_buffer("weights", trapezoid * phi)  # quadrature x poids w(t)

    def forward(self, z: Tensor) -> Tensor:
        directions = torch.randn(z.size(-1), self.num_slices, device=z.device)
        directions = directions / directions.norm(dim=0)
        x_t = (z @ directions).unsqueeze(-1) * self.t  # (V, N, S, T)

        # |phi_N(t) - phi(t)|^2, phi_N(t) = E[cos tx] + i E[sin tx]
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()  # (V, S, T)
        return ((err @ self.weights) * z.size(-2)).mean()
