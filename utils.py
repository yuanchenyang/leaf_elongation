import argparse
import os
import cv2
import glob
import csv
import numpy as np

from functools import partial
from itertools import tee, product, chain
from scipy.stats import gaussian_kde
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors, NearestCentroid
from PIL import Image
from roifile import ImagejRoi, roiwrite
from matplotlib import pyplot as plt

import torch
from torchvision.utils import draw_segmentation_masks, save_image
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def get_filenames(args):
    return glob.glob(os.path.join(args.dir, f'*{args.ext}'))

def get_full_path(args, filename, mkdir=False):
    path = os.path.join(args.dir, filename)
    if mkdir and not os.path.exists(path):
        os.makedirs(path)
    return path

def get_name(path):
    return os.path.split(path)[-1]

def replace_ext(filename, new_ext):
    return os.path.splitext(filename)[0] + new_ext

def open_masks(filename):
    return np.array(Image.open(filename))

def save_masks(filename, masks):
    im = Image.fromarray(masks)
    im.save(filename)

def get_default_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dir',
                        help='Base directory')
    parser.add_argument("-v", "--verbose",
                        help="Increase output verbosity",
                        action="store_true")
    return parser

def get_outlines(masks, min_size, verbose=False):
    """ get outlines of masks as a list to loop over for plotting """
    indices, counts = np.unique(masks, return_counts=True)
    if verbose:
        print('Total ROIs:', len(indices) - 1)
    for i, count in zip(indices, counts):
        if i == 0 or count < min_size:
            continue
        mn = masks==i
        contours = cv2.findContours(mn.astype(np.uint8), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
        contours = contours[-2]
        cmax = np.argmax([c.shape[0] for c in contours])
        pix = contours[cmax].astype(int).squeeze()
        if len(pix) > 4:
            yield pix

def rot_matrix(t):
    return np.array([[np.cos(t), -np.sin(t)],[np.sin(t), np.cos(t)]])

def scale(x, y, angle, factor):
    x1, y1 = rot_matrix(-angle) @ np.array([x, y])
    x2, y2 = rot_matrix(angle) @ np.array([x1, y1 * factor])
    return x2, y2

def get_anchor(measure_from, rect):
    if measure_from == 'center':
        return rect[0]
    ps = cv2.boxPoints(rect)
    centers = np.array([(p1 + p2)/2 for p1, p2 in pairwise(np.vstack([ps, ps[0]]))])
    if measure_from == 'left':
        return centers[np.argmin(centers[:, 0])]
    if measure_from == 'right':
        return centers[np.argmax(centers[:, 0])]
    if measure_from == 'top':
        return centers[np.argmin(centers[:, 1])]
    if measure_from == 'bottom':
        return centers[np.argmax(centers[:, 1])]
    raise ValueError('Invalid `measure_from`!')

def find_pairs(outlines, img, outfile, angle,
               max_dist=1000, neighbors=2, scaling_factor=10, measure_from='center'):
    rects = [cv2.minAreaRect(o) for o in outlines]
    rect_anchors = [get_anchor(measure_from, r) for r in rects]
    scaled_anchors = [scale(x, y, angle, scaling_factor) for x, y in rect_anchors]
    if len(scaled_anchors) <= neighbors:
        return # Too few rects to find neighbors
    nbrs = NearestNeighbors(n_neighbors=neighbors+1, algorithm='ball_tree').fit(scaled_anchors)
    distances, indices = nbrs.kneighbors(scaled_anchors)
    cv2.drawContours(img, outlines, -1, (0, 255, 0), 3)
    cv2.drawContours(img, [np.intp(cv2.boxPoints(r)) for r in rects],
                     -1, (255, 0, 0), 3)
    for i, ((_, *idx), (_, *dist)) in enumerate(zip(indices, distances)):
        for j, d in zip(idx, dist):
            if d < max_dist: # Reduce to filter out far apart cells
                c1, c2 = np.intp(rect_anchors[i]), np.intp(rect_anchors[j])
                yield c1, c2
                cv2.line(img, c1, c2, (0, 0, 255), 3)
    plt.imshow(img)
    plt.tight_layout()
    plt.savefig(outfile)

def save_rois(filename, outlines, verbose=False):
    rois = [ImagejRoi.frompoints(outline) for outline in outlines]
    if verbose:
        print(f'Saving {len(rois)} ROIs')

    # Delete file if it exists; the roifile lib appends to existing zip files.
    if os.path.exists(filename):
        os.remove(filename)
    roiwrite(filename, rois)

def filter_masks(masks, min_size, max_size, verbose=False):
    indices, counts = np.unique(masks, return_counts=True)
    if verbose:
        print(f'{len(counts) - 1} total masks of sizes')
        print(np.sort(counts[1:]))
    for i, count in zip(indices, counts):
        if count < min_size or count > max_size:
            masks[masks == i] = 0
    if verbose:
        print(f'{len(np.unique(masks)) - 1} masks after filtering')
    return masks

def directionality(im, bins=500, kde_bw_method=0.10, ksize=7,
                   verbose=False, plot=False):
    grad_x = cv2.Sobel(src=im, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=ksize)
    grad_y = cv2.Sobel(src=im, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=ksize)
    magnitude = np.sqrt(grad_x**2 + grad_y**2).astype(np.float64)
    magnitude /= np.max(magnitude)
    angles = np.arctan2(grad_y, grad_x)
    # Range from -pi/4 to pi/4, so center at 0
    angles_shifted = ((angles + np.pi/4) % (np.pi/2) - np.pi/4)
    data = angles_shifted.flatten()
    # Remove 0 values from data
    data = data[data!=0]
    weights, p = np.histogram(data, bins=bins)
    kde = gaussian_kde(p[1:], weights=weights, bw_method=kde_bw_method)
    x = np.linspace(-np.pi/4, np.pi/4, bins*20)
    y = kde.pdf(x)
    angle = x[np.argmax(y)]
    if verbose: print(f'Found angle: {angle*180/np.pi} degrees')
    if plot:
        plt.hist(data, bins=bins, density=True)
        plt.plot(x, y)
    return angle

def draw_slope_lines(im_gray, angle, nlines=10):
    h, w = im_gray.shape
    x = np.linspace(0, w-1, 2)
    slope = np.tan(angle)
    for dy in np.linspace(0, h-slope*w-1, nlines):
        plt.plot(x, slope*x + dy, 'c:')

def round_int(s):
    return int(round(float(s),0))

def read_img(filename):
    return np.float64(cv2.imread(filename, cv2.IMREAD_GRAYSCALE))

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

def draw_masks_on_img(img: Image, masks_array:np.ndarray, alpha=0.4, grey_img=True) -> torch.tensor:
    if grey_img:
        img = img.convert('L').convert('RGB')
    return draw_segmentation_masks(
        pil_to_tensor(img),
        torch.tensor(masks_array),
        alpha=alpha
    )

def masks_to_one_hot_array(masks: np.ndarray):
    # (H, W) integer array with to (N, H, W) boolean array
    return (np.arange(masks.max()) == masks[...,None]).astype(bool).transpose(2, 0, 1)
