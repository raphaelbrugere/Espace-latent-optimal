#!/bin/bash
# Pretraining LeJEPA d'une soumission de l'arène, sur un GPU du DCE.
#
# À lancer depuis la racine du dépôt, sur le frontal :
#   mkdir -p logslurms
#   sbatch slurm/arena_pretrain.sbatch flavien --epochs 100 --resume
#   sbatch --time=12:00:00 slurm/arena_pretrain.sbatch flavien --resume   # 800 epochs
#   sbatch slurm/arena_pretrain.sbatch baseline                           # la baseline n'a pas --resume
#
# Le 1er argument est le dossier de la soumission, les suivants sont passés tels quels
# à submissions/<soumission>/train/pretrain_lejepa.py.
# Les données doivent être téléchargées avant (sur le frontal), voir docs/CLUSTER.md.
#
#SBATCH --job-name=arena-pretrain
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_long
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=logslurms/%x-%j.out
#SBATCH --error=logslurms/%x-%j.err

set -euo pipefail

SUB=${1:?usage: sbatch slurm/arena_pretrain.sbatch <soumission> [arguments du script]}
shift

REPO=$SLURM_SUBMIT_DIR
cd "$REPO/projects/lejepa_arena"
source "$REPO/.venv/bin/activate"
export ARENA_DATA=${ARENA_DATA:-$PWD/data}

echo "job $SLURM_JOB_ID sur $(hostname), soumission $SUB, arguments: $*"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c "import torch; assert torch.cuda.is_available(), 'pas de GPU visible'; print('torch', torch.__version__, torch.cuda.get_device_name())"

python "submissions/$SUB/train/pretrain_lejepa.py" --workers "${SLURM_CPUS_PER_TASK:-8}" "$@"
