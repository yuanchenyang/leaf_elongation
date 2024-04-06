import torch
import numpy as np
import cv2
from PIL import Image
from itertools import pairwise
from torchvision.utils import draw_segmentation_masks
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

from segment_anything import (
    sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
)
from segment_anything.utils.amg import (
    build_point_grid, build_all_layer_point_grids, rle_to_mask
)

def show_anns(anns):
    img = anns_img(anns)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    ax.imshow(img)

def anns_img(anns, rgba=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1],
                   4 if rgba else 3))
    if rgba:
        img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        if rgba:
            color_mask = np.concatenate([np.random.random(3), [0.35]])
        else:
            color_mask = np.random.randint(0,255, size=(3,))
        img[m] = color_mask
    return img

def get_sam_model(checkpoint, model_type='vit_b', device='cuda'):
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=device)
    return sam

def show_masks(image, masks, modes=('mask',), thickness=5):
    img_np = pil_to_cv(image)
    for mask in masks:
        if 'bb' in modes:
            x1, y1, w, h = map(int, mask['bbox'])
            cv2.rectangle(img_np, (x1, y1), (x1+w, y1+h), color=(255,0,0), thickness=thickness)
        elif 'outline' in modes:
            outlines = cv2.findContours(mask['segmentation'].astype(np.uint8), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)[-2]
            cv2.drawContours(img_np, outlines, -1, (255, 0, 0), thickness)

    if 'mask' in modes:
        mask_img = np.zeros_like(img_np)
        for mask in masks:
            color = np.array(np.random.randint(0,255, size=(3,)), dtype='uint8')
            mask_img[mask['segmentation']] = color
        img_np = cv2.addWeighted(img_np, 0.8, mask_img, 0.2, 0)
    return img_np

def img_crops(img, nx, ny):
    w, h = img.size
    wcrops = np.linspace(0, w, nx+1).astype(int)
    hcrops = np.linspace(0, h, ny+1).astype(int)
    return [[img.crop((x0,y0, x1, y1))
             for x0, x1 in pairwise(wcrops)]
            for y0, y1 in pairwise(hcrops)]

def get_grid(nx, ny):
    x, y = np.meshgrid(np.linspace(0,1,nx), np.linspace(0,1,ny))
    return np.stack([x.flatten(), y.flatten()], axis=1)

def pil_to_cv(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)

def save_masks_png(filename, masks):
    masks_png = np.zeros(masks[0]["segmentation"].shape, dtype=np.int32)
    for i, mask in enumerate(masks):
        masks_png[mask["segmentation"]] = i+1
    im = Image.fromarray(masks_png)
    im.save(filename)

def generate_and_save_masks(img, sam, **kwargs):
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side = None, #50,
        point_grids = [get_grid(30, 60) for _ in range(2)],
        pred_iou_thresh=0.2,
        stability_score_thresh=0.65,
        crop_nms_thresh=0.2,
        crop_overlap_ratio = 0.3,
        box_nms_thresh=0.2,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=0,  # Requires open-cv to run post-processing
    )
    masks = mask_generator.generate(pil_to_cv(img))
    masks_array = np.stack([mask["segmentation"] for mask in masks])
    mask_on_img = draw_segmentation_masks(
        pil_to_tensor(image.convert('L').convert('RGB')),
        torch.tensor(masks_array),
        alpha=0.4
    )
