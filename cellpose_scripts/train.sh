#!/bin/bash

# Debug:
# && python -m cellpose \
# 'b cellpose/core.py:876' _train_net

#
# ./selected/train                 : folder with images and masks
# ./selected/train/models/CP_model : trained model
# ./selected/test                  : folder with images to run model on

rm -f selected/train/models/* \
&& python -m cellpose \
       --verbose \
       --use_gpu \
       --train \
       --dir selected/train \
       --pretrained_model cyto \
       --chan 0 \
       --n_epochs 1000 \
       --learning_rate 0.04 \
       --weight_decay 0.0001 \
       --nimg_per_epoch 8 \
       --min_train_masks 1 \
       --mask_filter '_cp_masks' \
&& mv selected/train/models/* selected/train/models/CP_model \
&& ./test.sh
