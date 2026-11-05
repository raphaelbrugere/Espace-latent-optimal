"""Poids externes : un module par jeu de poids, qui charge dans une architecture d'archs/.

`DOWNLOADS` : nom -> téléchargement, utilisé par tools/prepare_data.py.
"""

from latentopt.models.backbones.external import dino

DOWNLOADS = {dino.NAME: dino.download}
