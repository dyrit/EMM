#!/bin/bash -l
export CUDA_VISIBLE_DEVICES=1
conda activate torch2

python EMLC/EMLC_main/main.py --seed 25 --fname 'runtest'
python EMLC/EMLC_main/main.py --seed 25 --wnew 10 \
--pretrain_epochs 500 \
--fname 'runtest' 
#
python EMLC/EMLC_main/main.py --seed 25 --wnew 10 \
--pretrain_epochs 500 --ds 'eron' \
--fname 'runtest' 

python EMLC/EMLC_main/main.py --seed 25 --wnew 10 \
--pretrain_epochs 500 --ds 'Col5k' \
--fname 'runtest' 