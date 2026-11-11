#!/bin/bash
# Entraîne des têtes de l'arène sur le backbone gelé d'une soumission, puis l'évalue sur val.
#
# À lancer depuis la racine du dépôt, sur le frontal, une fois backbone.pt écrit :
#   sbatch slurm/arena_heads.sbatch flavien pets_seg pets_det     # quelques têtes
#   sbatch slurm/arena_heads.sbatch flavien                       # les 5 têtes
#   sbatch --export=ALL,TAG=100ep slurm/arena_heads.sbatch flavien pets_seg pets_det
#
# Le 1er argument est le dossier de la soumission, les suivants les tâches (toutes par défaut).
# Chaque tête est sauvée dans submissions/<soumission>/heads/<tâche>.pt (l'ancienne est écrasée).
# evaluate note TOUTES les têtes présentes dans heads/, pas seulement celles de ce job.
# Résultats : projects/lejepa_arena/results/<TAG>/val/<soumission>.json (TAG par défaut : latest).
#
#SBATCH --job-name=arena-heads
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_long
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
# bf16 obligatoire (voir arena_pretrain.sbatch) : pas de GTX 1080 Ti ni de RTX 2080 Ti.
#SBATCH --exclude=sh[10-16],tx[00-16]
#SBATCH --output=logslurms/%x-%j.out
#SBATCH --error=logslurms/%x-%j.err

set -euo pipefail

SUB=${1:?usage: sbatch slurm/arena_heads.sbatch <soumission> [tâches...]}
shift
TASKS=("$@")
if [ ${#TASKS[@]} -eq 0 ]; then
    TASKS=(imagenette_cls flowers102_cls galaxy10_cls pets_seg pets_det)
fi
TAG=${TAG:-latest}

REPO=$SLURM_SUBMIT_DIR
cd "$REPO/projects/lejepa_arena"
source "$REPO/.venv/bin/activate"
export ARENA_DATA=${ARENA_DATA:-$PWD/data}
WORKERS=${SLURM_CPUS_PER_TASK:-8}

echo "job $SLURM_JOB_ID sur $(hostname), soumission $SUB, tâches: ${TASKS[*]}, tag $TAG"
python -c "
import torch
assert torch.cuda.is_available(), 'pas de GPU visible'
name, cc = torch.cuda.get_device_name(), torch.cuda.get_device_capability()
print('torch', torch.__version__, name, 'compute capability', cc)
assert cc >= (8, 0), f'{name} ne gère pas le bf16 (il faut Ampere ou plus récent)'
"

for task in "${TASKS[@]}"; do
    echo "=== $task ==="
    python "submissions/$SUB/train/train_head.py" --task "$task" --workers "$WORKERS"
done

echo "=== evaluate (val) ==="
python -m arena.evaluate "submissions/$SUB" --out "results/$TAG" --workers "$WORKERS"
