#!/bin/bash

python -m cellpose \
       --verbose \
       --use_gpu \
       --dir selected/test \
       --pretrained_model selected/train/models/CP_model \
       --diameter 0. \
       --chan 0 \
       --flow_threshold 0.6 \
       --cellprob_threshold '-0.5' \
       --save_mpl \
       --save_png \
       --save_txt \
       --no_npy \
&& wc -l selected/test/*.txt
