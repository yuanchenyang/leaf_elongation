#!/bin/bash

# Debug:
# && python -m cellpose \
# 'b cellpose/core.py:876' _train_net

#
# ./smaller/train                 : folder with images and masks
# ./smaller/train/models/CP_model : trained model
# ./smaller/test                  : folder with images to run model on

rm -f smaller/train/models/* \
&& python -m pdb -c 'b cellpose/models.py:783' -c 'c' -c 'nimg_per_epoch = 8' -c 'c' -c 'q' -m cellpose \
       --verbose \
       --use_gpu \
       --train \
       --dir smaller/train \
       --pretrained_model cyto \
       --chan 2 \
       --n_epochs 200 \
       --learning_rate 0.001 \
       --weight_decay 0.0001 \
       --min_train_masks 1 \
       --mask_filter '_cp_masks' \
&& mv smaller/train/models/* smaller/train/models/CP_model \
&& python -m cellpose \
       --verbose \
       --use_gpu \
       --dir smaller/test \
       --pretrained_model smaller/train/models/CP_model \
       --diameter 0. \
       --chan 2 \
       --flow_threshold 0.45 \
       --cellprob_threshold '-0.2' \
       --save_png \
       --save_txt \
       --no_npy \
&& wc -l smaller/test/*.txt
