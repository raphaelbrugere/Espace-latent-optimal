"""python -m latentopt.cli <commande> : équivalent de `lo`, sans le script installé."""

import sys

from latentopt.cli import main

sys.exit(main())
