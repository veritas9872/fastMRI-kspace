#!/usr/bin/env bash
python3.7 eval/evaluate_recons.py \
--predictions-path /home/user/PycharmProjects/fastMRI-kspace/recons/I2R_T30_cp15_v4_ensemble_rot/ \
--target-path /media/user/Data/compFastMRI/multicoil_val \
--challenge multicoil