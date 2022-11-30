#!/usr/bin/env bash
# ResNet50, Office-Home, Single Source

(
CUDA_VISIBLE_DEVICES=0 python src/CALE.py data/office-home -d OfficeHome -s Ar -t Cl -a resnet50 -b 64 --epochs 20 --seed 0 --trade-off2 1.0 --trade-off3 0.5 --log logs/OfficeHome_Ar2Cl
CUDA_VISIBLE_DEVICES=0 python src/CALE.py data/office-home -d OfficeHome -s Ar -t Pr -a resnet50 -b 64 --epochs 20 --seed 0 --trade-off2 1.0 --trade-off3 0.5 --log logs/OfficeHome_Ar2Pr
) & (
CUDA_VISIBLE_DEVICES=1 python src/CALE.py data/office-home -d OfficeHome -s Ar -t Rw -a resnet50 -b 64 --epochs 20 --seed 0 --trade-off2 1.0 --trade-off3 0.5 --log logs/OfficeHome_Ar2Rw
CUDA_VISIBLE_DEVICES=1 python src/CALE.py data/office-home -d OfficeHome -s Cl -t Ar -a resnet50 -b 64 --epochs 20 --seed 0 --trade-off2 1.0 --trade-off3 0.5 --log logs/OfficeHome_Cl2Ar
) & (
CUDA_VISIBLE_DEVICES=2 python src/CALE.py data/office-home -d OfficeHome -s Cl -t Pr -a resnet50 -b 64 --epochs 20 --seed 0 --trade-off2 1.0 --trade-off3 0.5 --log logs/OfficeHome_Cl2Pr
CUDA_VISIBLE_DEVICES=2 python src/CALE.py data/office-home -d OfficeHome -s Cl -t Rw -a resnet50 -b 64 --epochs 20 --seed 0 --trade-off2 1.0 --trade-off3 0.5 --log logs/OfficeHome_Cl2Rw
) & (
CUDA_VISIBLE_DEVICES=4 python src/CALE.py data/office-home -d OfficeHome -s Pr -t Ar -a resnet50 -b 64 --epochs 20 --seed 0 --trade-off2 1.0 --trade-off3 0.5 --log logs/OfficeHome_Pr2Ar
CUDA_VISIBLE_DEVICES=4 python src/CALE.py data/office-home -d OfficeHome -s Pr -t Cl -a resnet50 -b 64 --epochs 20 --seed 0 --trade-off2 1.0 --trade-off3 0.5 --log logs/OfficeHome_Pr2Cl
) & (
CUDA_VISIBLE_DEVICES=5 python src/CALE.py data/office-home -d OfficeHome -s Pr -t Rw -a resnet50 -b 64 --epochs 20 --seed 0 --trade-off2 1.0 --trade-off3 0.5 --log logs/OfficeHome_Pr2Rw
CUDA_VISIBLE_DEVICES=5 python src/CALE.py data/office-home -d OfficeHome -s Rw -t Ar -a resnet50 -b 64 --epochs 20 --seed 0 --trade-off2 1.0 --trade-off3 0.5 --log logs/OfficeHome_Rw2Ar
) & (
CUDA_VISIBLE_DEVICES=6 python src/CALE.py data/office-home -d OfficeHome -s Rw -t Cl -a resnet50 -b 64 --epochs 20 --seed 0 --trade-off2 1.0 --trade-off3 0.5 --log logs/OfficeHome_Rw2Cl
CUDA_VISIBLE_DEVICES=6 python src/CALE.py data/office-home -d OfficeHome -s Rw -t Pr -a resnet50 -b 64 --epochs 20 --seed 0 --trade-off2 1.0 --trade-off3 0.5 --log logs/OfficeHome_Rw2Pr
) & 



