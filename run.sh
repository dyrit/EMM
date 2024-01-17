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

python EMLC/EMLC_main/main.py --seed 25 --wnew 10 \
--pretrain_epochs 500 --ds 'nus' --tr_rounds 30 \
--train 0.0003 --test 0.005 --pool 0.01 \
--fname 'runtest1' 

python EMLC/EMLC_main/main.py --seed 25 --wnew 10 \
--pretrain_epochs 500 --ds 'bibTex' --tr_rounds 30 \
--train 0.002 --test 0.2 --pool 0.6 \
--fname 'runtest' 

python EMLC/EMLC_main/main.py --seed 25 --wnew 10 \
--pretrain_epochs 500 --ds 'bibTex' --tr_rounds 30 \
--train 0.02 --test 0.2 --pool 0.6 \
--al_mtd 'evicov1' --wnew 100 \
--l_epochs 10 --pi_epochs 5 \
--fname '_evicov1_wnew100_freeze_e_g1_l_10_pi_5_bk'  

python EMLC/EMLC_main/main.py --seed 25 --wnew 10 \
--pretrain_epochs 500 --ds 'bibTex' --tr_rounds 300 \
--train 0.02 --test 0.2 --pool 0.6 \
--al_mtd 'evicov1' --wnew 100 \
--l_epochs 1 --pi_epochs 1 \
--opt_mu True \
--fname '_opt_evicov1_wnew100_freeze_e_g1_l_1_pi_1_bk'  

python EMLC/EMLC_main/main.py --seed 25 --wnew 10 \
--pretrain_epochs 500 --ds 'Delicious' --tr_rounds 300 \
--train 0.02 --test 0.2 --pool 0.6 \
--al_mtd 'evicov1' --wnew 100 \
--l_epochs 100 --pi_epochs 100 \
--opt_mu True \
--fname '_opt_evicov1_wnew100_freeze_e_g1_l_100_pi_100_bk' 

python EMLC/EMLC_main/main.py --seed 25 --wnew 10 \
--pretrain_epochs 500 --ds 'Delicious' --tr_rounds 300 \
--train 0.02 --test 0.2 --pool 0.6 \
--al_mtd 'evicov1' --wnew 100 \
--l_epochs 100 --pi_epochs 100 \
--opt_mu True --AL_rounds 100 \
--fname '_mlnntest' 