python submissions/test_team/train/export_timm_weights.py >> 01.log
python submissions/test_team/train/train_head.py --task imagenette_cls >> 02.log
python submissions/test_team/train/train_head.py --task flowers102_cls >> 03.log
python submissions/test_team/train/train_head.py --task galaxy10_cls>> 04.log
python submissions/test_team/train/train_head.py --task pets_seg>> 05.log
python submissions/test_team/train/train_head.py --task pets_det>> 06.log
python -m arena.evaluate submissions/test_team >> 07.log