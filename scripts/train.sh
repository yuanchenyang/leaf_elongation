#!\bin\bash
python -m cellpose \
       --verbose \
       --use_gpu \
       --train \
       --dir big \
       --pretrained_model cyto \
       --chan 0 \
       --n_epochs 500 \
       --learning_rate 0.1 \
       --weight_decay .0001 \
       --mask_filter="_cp_masks" \
