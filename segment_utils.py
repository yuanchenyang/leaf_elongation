import torch
import numpy as np
import cv2
import pprint
from itertools import product
from tqdm import tqdm
from PIL import Image
from itertools import pairwise

from segment_anything import (
    sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
)
from segment_anything.utils.amg import (
    build_point_grid, build_all_layer_point_grids, rle_to_mask
)

from utils import *

def get_sam_model(checkpoint, model_type='vit_b', device='cuda'):
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=device)
    return sam

def generate_and_save_masks(sam, img, save_to, **kwargs):
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        **kwargs,
    )
    masks = mask_generator.generate(pil_to_cv(img))
    masks_array = np.stack([mask["segmentation"] for mask in masks])
    mask_on_img = draw_masks_on_img(img, masks_array)
    to_pil_image(mask_on_img).save(save_to)

class Sweep(list):
    pass

def get_sweep(kwargs):
    sweeps = {key:value for key,value in kwargs.items() if type(value) == Sweep}
    for values in product(*sweeps.values()):
        new_kwargs = kwargs.copy()
        new_kwargs.update(zip(sweeps, values))
        yield new_kwargs

def param_sweep(img_path, sam_path, save_img='tmp/image{:03}.jpg',
                save_config='tmp/{:03}.txt'):
    sam = get_sam_model(sam_path)
    img = Image.open(img_path)
    kwargs = dict(
        points_per_side                = None,
        point_grids                    = [get_grid(30, 60) for _ in range(2)],
        pred_iou_thresh                = Sweep([0.2, 0.25, 0.3, 0.35, 0.4]),
        stability_score_thresh         = Sweep([0.55, 0.60, 0.65, 0.7, 0.75]),
        stability_score_offset         = Sweep([0.9, 1.0, 1.1]),
        box_nms_thresh                 = 0.2,
        crop_n_layers                  = 1,
        crop_nms_thresh                = 0.2,
        crop_overlap_ratio             = 0.3,
        crop_n_points_downscale_factor = 2,
        min_mask_region_area           = 0,  # Requires open-cv to run post-processing
    )
    for i, cur_kwargs in tqdm(enumerate(get_sweep(kwargs))):
        with open(save_config.format(i), 'w') as f:
            pprint.pprint(cur_kwargs, f)
        generate_and_save_masks(sam, img, save_img.format(i), **cur_kwargs)

if __name__=='__main__':
    param_sweep('../Wheat Z W 4 control/2021-02-19-tris_minimal-0307_2023-05-22_046.jpg',
                '../segment_anything/vit_b_lm.pth')
