#!/bin/bash

python -m cellpose \
       --verbose \
       --use_gpu \
       --dir test \
       --pretrained_model training-data/big/models/CP_big_n=3_500_0.1 \
       --diameter 0. \
       --chan 0 \
       --min_size 100 \
       --save_outlines \
       --save_txt \
       --save_png \
       --no_npy \
