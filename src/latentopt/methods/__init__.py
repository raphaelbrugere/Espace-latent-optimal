"""Methodes d'entrainement : LeJEPA (auto-supervise) et sondage sur backbone gele."""

from latentopt.methods.frozen import FrozenBackboneModule
from latentopt.methods.lejepa import LeJEPA, LeJEPAModule, MultiViewTransform, build_optimizer

__all__ = [
    "FrozenBackboneModule",
    "LeJEPA",
    "LeJEPAModule",
    "MultiViewTransform",
    "build_optimizer",
]
